import hashlib
import re
import time
from typing import Any

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from pinecone import Pinecone, ServerlessSpec
from pinecone_text.hybrid import hybrid_convex_scale
from pinecone_text.sparse import BM25Encoder
from sentence_transformers import CrossEncoder
from lawApp_LangGraph.FastAPI.logging import rag as rag_log
from pinecone_text.sparse import SparseVector

load_dotenv()


class RAG_service:
    def __init__(
        self,
        index_name: str,
        api_key: str,
        cloud: str,
        region: str,
        dimension: int = 1024,
    ):
        """
        创建初始化类
        包含:
        初始化Pinecone
        md文档分割工具
        递归分块(段落 句子)
        BGA向量化
        BM25稀疏矩阵向量化
        BAAI重排序模型

        :param index_name:
        :param api_key:
        :param cloud:
        :param region:
        :param dimension:
        """
        self.index_name = index_name
        self.api_key = api_key
        self.cloud = cloud
        self.region = region
        self.dimension = dimension
        self.pc = Pinecone(api_key=api_key)
        self.index = None

        # 初始化工具
        headers_to_split_on = [("#", "Header_1")]

        # md文档分割
        self.md_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on
        )
        # 段落句子分割
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,
            chunk_overlap=50,
            separators=["\n\n", "\n", ".", "；", " "],
            add_start_index=True,
        )

        # 重排序模型
        self.reranker = CrossEncoder("BAAI/bge-reranker-large", max_length=512)

        # 稀疏向量
        self.bm25 = BM25Encoder().load(
            "E:\\LangChain\\lawApp_LangGraph\\RAG_service\\bm25_law_params.json"
        )

        # 密集向量
        model_name = "BAAI/bge-large-zh-v1.5"
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name, encode_kwargs={"normalize_embeddings": True}
        )

    def create_index(self, wait_for_completion: bool = True) -> bool:
        """
        创建索引
        创建混合索引
        指定的索引方式
        :param wait_for_completion:
        :return:
        """

        # 混合索引的强制要求：metric 必须为 dotproduct,vector_type 为 dense
        target_metric = "dotproduct"

        # 检查索引是否存在
        if not self.pc.has_index(self.index_name):
            print(f"正在创建混合检索索引: {self.index_name}...")
            self.pc.create_index(
                name=self.index_name,
                dimension=self.dimension,
                metric=target_metric,
                spec=ServerlessSpec(cloud=self.cloud, region=self.region),
                vector_type="dense",
            )
        else:
            print(f"索引 '{self.index_name}' 已存在.")
        if wait_for_completion:
            while not self.pc.describe_index(self.index_name).status.get(
                "ready", False
            ):
                time.sleep(2)

        self.index = self.pc.Index(self.index_name)
        print(f"索引 '{self.index_name}' 已就绪.")
        return True

    def get_index_stats(self):
        stats = self.index.describe_index_stats()
        print("当前索引状态:", stats)
        return stats

    def get_Documents(self, file_path: str) -> list[Any] | None:
        """
        添加文本到数据库中需要:
        加载文本,
        文本分块,

        具体实施:
        (正则表达式)
        将清洗好的文件加载
        以每一个案例为单位
        先提取元数据
        (分块)
        从[基本案情]到最后的内容提取
        以句子为单位,合成一大段

        数据清洗:
        metadata:{year,case_number,case_cause,chunk_text}


        :param file_path:str
        :return 符合输入格式的列表
        """

        loader = TextLoader(file_path, encoding="utf-8")

        # 插入Pinecone数据容器
        Pinecone_records = []

        # 默认读取的为Document形式
        documents = loader.load()

        # 统一换行符 (\r\n → \n), 避免跨平台正则匹配问题
        raw_text = documents[0].page_content
        raw_text = raw_text.replace("\r\n", "\n").replace("\r", "\n")

        # 生成文件唯一标识 (取路径 MD5 前 8 位)
        file_hash = hashlib.md5(file_path.encode()).hexdigest()[:8]

        # 按一级标题分割成多个案例
        articles = self.md_splitter.split_text(raw_text)

        # 获取元数据和正文内容
        for DocuID, article in enumerate(articles):
            print(f">> 第{DocuID}篇文章获取中...")
            # 元数据容器
            metadata = {}

            # 提取年份：假设文件名或路径包含 202X
            year_match = re.search(r"20\d{2}", article.page_content)
            metadata["year"] = year_match.group(0) if year_match else "Unknown"

            # 提取裁判书字号：匹配如（2023）最高法民终...号
            case_num_pattern = (
                r"裁判书字号\s+((?:(?!裁判书字号)[\s\S])+?法院[\s\S]+?书)"
            )
            case_num_match = re.search(case_num_pattern, article.page_content)

            metadata["case_number"] = (
                case_num_match.group(1).strip() if case_num_match else "未识别"
            )

            # 提取案由：通常在字号之后,或者是特定的段落
            case_cause_pattern = r"案由[:：]\s*([\u4e00-\u9fa5]+)"
            cause_match = re.search(case_cause_pattern, article.page_content)

            metadata["case_cause"] = cause_match.group(1) if cause_match else "通用"

            # 最终的metadata示例: 'metadata': {'case_cause': ,'case_number': , 'chunk_index': 2,'chunk_text': }

            # 提取基本案情
            facts_pattern = r"【基本案情】\s*([\s\S]+?)(?=\n【|$)"
            facts_match = re.search(facts_pattern, article.page_content)

            if not facts_match:
                continue
            # 获取捕获组内容（不含【基本案情】）
            raw_content = facts_match.group(1)
            # 去除空格、换行、制表符等所有空白字符,以及 # 符号
            facts_cleaned = re.sub(
                r"\n+", "\n", raw_content
            ).strip()  # 去除所有空白（空格、换行等）
            facts_cleaned = facts_cleaned.replace("#", "")  # 去除所有 # 字符

            print("元数据和原文解析完毕...")

            chunks = self.text_splitter.split_text(facts_cleaned)

            for i, chunk in enumerate(chunks):
                print(f"第{i}个记录创建中...")
                # 独有的
                record_id = f"lawCase_{file_hash}_{DocuID}_chunk{i}"

                # 密集向量
                dense_vector = self.embeddings.embed_query(chunk)

                # 返回 {"indices": [...], "values": [...]}
                sparse_vector = self.bm25.encode_documents(chunk)

                # 过滤空稀疏向量 (BM25 对极短文本可能返回空)
                if not sparse_vector["values"] or not sparse_vector["indices"]:
                    continue

                """
                添加内容: id 向量数据 元数据:{年份 判决书 案由 文档切片}
                符合Pinecone的输入格式
                """
                record = {
                    "id": record_id,
                    "values": dense_vector,  # 稠密向量列表
                    "sparse_values": sparse_vector,  # 稀疏向量字典
                    "metadata": {
                        **metadata,
                        "chunk_index": i,
                        "chunk_text": chunk,
                    },
                }
                Pinecone_records.append(record)

                print("记录插入完毕")

        return Pinecone_records

    def add_document(
        self,
        records: list,
        namespace: str,
        batch_size: int = 50,
        pause: float = 1.0,
        max_retries: int = 3,
        show_stats: bool = False,
    ):
        """分批 upsert 到 Pinecone, 批次间暂停 pause 秒避免 API 限流。

        :param records: 待上传的记录列表
        :param namespace: Pinecone 命名空间
        :param batch_size: 每批上传条数
        :param pause: 批次间等待秒数
        :param max_retries: 单批失败最大重试次数
        :param show_stats: 是否在完成后打印索引统计
        """
        if self.index is None:
            raise RuntimeError("索引未初始化, 请先调用 create_index()")

        if not records:
            print("  记录列表为空, 跳过上传")
            return True

        total = len(records)
        success_count = 0
        fail_count = 0

        for i in range(0, total, batch_size):
            batch = records[i : i + batch_size]
            uploaded = min(i + batch_size, total)

            for attempt in range(1, max_retries + 1):
                try:
                    self.index.upsert(vectors=batch, namespace=namespace)
                    success_count += len(batch)
                    print(f"  已上传 {min(uploaded, total)}/{total} 条")
                    break
                except Exception as e:
                    if attempt < max_retries:
                        wait = 2**attempt
                        print(
                            f"  批次 [{i}-{uploaded}] 失败, {wait}s 后重试 ({attempt}/{max_retries}): {e}"
                        )
                        time.sleep(wait)
                    else:
                        fail_count += len(batch)
                        print(f"  批次 [{i}-{uploaded}] 最终失败: {e}")

            if uploaded < total:
                time.sleep(pause)

        print(f"  上传完成: 成功 {success_count} 条, 失败 {fail_count} 条")
        if show_stats:
            self.get_index_stats()
        return fail_count == 0

    def search_withDenseSparse(
        self,
        query: str,
        namespace: str,
        top_k: int = 50,
        rerank_top_n: int = 10,
        alpha: float = 0.5,
    ) -> list:
        """
        分别获取问题的稀疏和密集向量化矩阵
        双路查询
        对结果和文字重排序
        分别检索的向量对文本意思没有关联
        交叉编码器同时接收查询‑文档对作为输入
        通过 Transformer 的全注意力机制（Self‑Attention）让查询和文档的每个词充分交互
        最终输出一个相关性分数
        :param query:
        :param namespace:
        :param top_k:
        :param rerank_top_n:
        :param alpha:
        :return:
        """
        t_total = time.time()

        # 步骤0：入参校验
        if self.index is None:
            raise RuntimeError("索引未初始化, 请先调用 create_index()")

        effective_n = max(1, int(rerank_top_n))
        effective_top_k = max(effective_n, int(top_k))

        rag_log.debug(
            "RAG 检索开始",
            detail=f"query={query[:60]} | top_k={effective_top_k} | alpha={alpha} | ns={namespace}",
        )

        # 步骤1：获取查询的密集和稀疏向量
        t_embed = time.time()
        dense_vec = self.embeddings.embed_query(query)
        sparse_vec = self.bm25.encode_queries(query)
        rag_log.debug(
            "密集+稀疏向量编码完成",
            detail=f"dense_dim={len(dense_vec)}",
            result=f"elapsed={time.time() - t_embed:.2f}s",
        )
        
        # 将 dict 显式转换为 SparseVector 对象
        sparse_vec_obj = SparseVector(
            indices=sparse_vec["indices"],
            values=sparse_vec["values"],
        )

        # 使用官方混合凸组合函数
        weighted_dense, weighted_sparse = hybrid_convex_scale(
            dense_vec, sparse_vec_obj, alpha
        )

        # 步骤2：混合召回
        t_query = time.time()
        results = self.index.query(
            vector=weighted_dense,
            sparse_vector=weighted_sparse,
            namespace=namespace,
            top_k=effective_top_k,
            include_metadata=True,
        )
        matches = results.matches
        rag_log.debug(
            "Pinecone 混合召回完成",
            detail=f"matches={len(matches)}",
            result=f"elapsed={time.time() - t_query:.2f}s",
        )

        if not matches:
            rag_log.info(
                "RAG 检索结束",
                detail="未检索到任何结果",
                result=f"total={time.time() - t_total:.2f}s",
            )
            return []

        # 步骤3：提取文本对
        texts = [m.metadata["chunk_text"] for m in matches]
        pairs = [[query, t] for t in texts]

        # 步骤4：重排序
        t_rerank = time.time()
        try:
            scores = self.reranker.predict(pairs)
            scores = list(scores) if not isinstance(scores, list) else scores
            scores = [float(s) for s in scores]
        except Exception as e:
            rag_log.warning(
                "重排序失败,降级为原始混合排序",
                detail=str(e),
                result=f"返回前{effective_n}条",
            )
            return matches[:effective_n]

        rag_log.debug(
            "CrossEncoder 重排序完成",
            detail=f"pairs={len(pairs)} | scores_range=[{min(scores):.3f}, {max(scores):.3f}]",
            result=f"elapsed={time.time() - t_rerank:.2f}s",
        )

        if len(scores) != len(matches):
            rag_log.warning(
                "重排序结果数量不匹配,降级为原始排序",
                detail=f"scores={len(scores)} vs matches={len(matches)}",
            )
            return matches[:effective_n]

        # 组合得分与匹配对象成元组，按得分排序，再取出匹配对象
        reranked = [
            match
            for match, _ in sorted(
                zip(matches, scores), key=lambda pair: pair[1], reverse=True
            )
        ]
        top_score = max(scores) if scores else 0
        rag_log.info(
            "RAG 检索完成",
            detail=f"召回{len(matches)}条 → 重排序 → 返回{min(effective_n, len(reranked))}条",
            result=f"top_score={top_score:.3f} | total={time.time() - t_total:.2f}s",
        )
        return reranked[:effective_n]


