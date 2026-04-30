import os
import re
import time
from typing import Any

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec
from pinecone_text.sparse import BM25Encoder
from sentence_transformers import CrossEncoder


class RAG_service:

    def __init__(
            self,
            index_name: str,
            api_key: str,
            cloud: str,
            region: str,
            dimension: int = 1024
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
        self.index = self.pc.Index(self.index_name)

        # 初始化工具
        headers_to_split_on = [("#", "Header_1")]

        self.md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,
            chunk_overlap=50,
            separators=["\n\n", "\n", "。", "；", " "],
            add_start_index=True
        )

        # 重排序模型
        self.reranker = CrossEncoder('BAAI/bge-reranker-large', max_length=512)

        bm25_path = os.getenv("BM25_PATH")
        # 稀疏向量
        self.bm25 = BM25Encoder().load(bm25_path)

        # 密集向量
        model_name = "BAAI/bge-large-zh-v1.5"
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            encode_kwargs={'normalize_embeddings': True}
        )

    def create_index(self, wait_for_completion: bool = True) -> bool:
        """
        创建索引
        创建混合索引
        指定的索引方式
        :param wait_for_completion:
        :return:
        """
        self.pc = Pinecone(api_key=self.api_key)

        # 混合索引的强制要求：metric 必须为 dotproduct，vector_type 为 dense
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
            print(f"索引 '{self.index_name}' 已存在。")
        # 等待索引就绪 异步处理
        if wait_for_completion:
            while not self.pc.describe_index(self.index_name).status.get("ready", False):
                time.sleep(2)
            print(f"索引 '{self.index_name}' 已就绪。")

        self.index = self.pc.Index(self.index_name)
        print(f"索引 '{self.index_name}' 已就绪。")

        return True

    def get_index_stats(self):
        """
        获取索引是否创建成功
        测试索引是否创建成功并打印统计信息
        :return: 是否创建成功
        """
        stats = self.index.describe_index_stats()
        print("当前索引的状态: ", stats)
        return True

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

        loader = TextLoader(
            file_path,
            encoding="utf-8"
        )

        # 插入Pinecone数据容器
        Pinecone_records = []

        # 默认读取的为Document形式
        documents = loader.load()

        # 按一级标题分割成多个案例
        articles = self.md_splitter.split_text(documents[0].page_content)

        # 获取元数据和正文内容
        for DocuID, article in enumerate(articles):
            print(f"第{DocuID}篇文章获取中...")
            # 元数据容器
            metadata = {}

            # 提取年份：假设文件名或路径包含 202X
            year_match = re.search(r"20\d{2}", article.page_content)
            metadata["year"] = year_match.group(0) if year_match else "Unknown"

            # 提取裁判书字号：匹配如（2023）最高法民终...号
            case_num_pattern = r'裁判书字号[\s\\n]+((?:(?!裁判书字号)[\s\S])+?法院[\s\S]+?书)'
            case_num_match = re.search(case_num_pattern, article.page_content)

            metadata["case_number"] = case_num_match.group(1).strip() if case_num_match else "未识别"

            # 提取案由：通常在字号之后，或者是特定的段落
            case_cause_pattern = r"案由[:：]\s*([\u4e00-\u9fa5]+)"
            cause_match = re.search(case_cause_pattern, article.page_content)

            metadata["case_cause"] = cause_match.group(1) if cause_match else "通用"

            # 最终的metadata示例:
            # 'metadata': {'case_cause': ,'case_number': ,
            # 'chunk_index': ,'chunk_text': }
            # 提取基本案情
            facts_pattern = r'【基本案情】\s*([\s\S]+?)(?=\n【|$)'
            facts_match = re.search(facts_pattern, article.page_content)

            if not facts_match:
                continue
            # 获取捕获组内容（不含【基本案情】）
            raw_content = facts_match.group(1)
            # 去除空格、换行、制表符等所有空白字符，以及 # 符号
            facts_cleaned = re.sub(r'\n+', '\n', raw_content).strip()  # 去除所有空白（空格、换行等）
            facts_cleaned = facts_cleaned.replace('#', '')  # 去除所有 # 字符

            print("元数据和原文解析完毕...")

            chunks = self.text_splitter.split_text(facts_cleaned)

            for i, chunk in enumerate(chunks):
                print(f"第{i}个记录创建中...")
                # 独有的
                record_id = f"annualCases{metadata['year']}_Docu{DocuID}_chunk{i}"

                # 密集向量
                dense_vector = self.embeddings.embed_query(chunk)

                # 返回 {"indices": [...], "values": [...]}
                sparse_vector = self.bm25.encode_documents(chunk)

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
                    }
                }
                Pinecone_records.append(record)

                print("记录插入完毕")

        return Pinecone_records

    def add_document_namespace(self, Pinecone_records, namespace: str, ):
        """
        分批次上传
        :param Pinecone_records:
        :param namespace:
        :return:
        """

        # 分批上传，每批最多 50 条向量
        batch_size = 50
        total = len(Pinecone_records)

        for i in range(0, total, batch_size):
            batch = Pinecone_records[i:i + batch_size]
            self.index.upsert(vectors=batch, namespace=namespace)
            uploaded = min(i + batch_size, total)
            print(f"已上传 {uploaded}/{total} 条记录")

        print(self.get_index_stats())

        return True

    def search_withDenseSparse(
            self,
            query: str,
            namespace: str,
            top_k: int = 50,
            rerank_top_n=10,
            alpha: float = 0.5
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
        :return: 重排序后的结果列表
        """
        # 密集向量
        dense_vec = self.embeddings.embed_query(query)

        # 稀疏向量
        sparse_vec = self.bm25.encode_documents(query)

        # 混合召回
        results = self.index.query(
            vector=dense_vec,
            sparse_vector=sparse_vec,
            alpha=alpha,  # 控制权重：关键词 (0 <====> 1) 语义
            namespace=namespace,
            top_k=top_k,
            include_metadata=True
        )
        print("混合召回中...")

        #  提取文本，准备重排序
        matches = results.matches
        texts = [m.metadata['chunk_text'] for m in matches]
        # 问题与原文的键值对
        pairs = [[query, t] for t in texts]

        # 计算重排序分数
        scores = self.reranker.predict(pairs)

        # 重组并排序
        for match, score in zip(matches, scores):
            match.rerank_score = score

        # 按分数高低排序
        reranked = sorted(matches, key=lambda x: x.rerank_score, reverse=True)

        print("重排序中...")

        print("重排序结果为:{}".format(reranked))

        return reranked[:rerank_top_n]


load_dotenv()

# 实例化测试
service = RAG_service(
    index_name="pinecone-test-lawapp",
    api_key=os.getenv("PINECONE_API_KEY"),
    cloud="aws",
    region="us-east-1",
)
result = service.get_Documents("../MarkDownFiles/中国法院2020年度案例：婚姻家庭与继承纠纷.md")
