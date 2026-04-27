import os
from typing import Any

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
import re
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone_text.sparse import BM25Encoder
import time
from pinecone import Pinecone, ServerlessSpec
from FlagEmbedding import FlagReranker

"""
RAG流程:
加载文件
文档切分
嵌入模型
存入向量数据库
混合检索(语义检索 BM25检索 RRF融合)
重排序(CROSS-ENCODER)
上下文组装

定义RAG_service类:
    配置文件
    配置数据库

    定义方法:
        添加文档
        检索文档

"""


class RAG_service:
    def __init__(
            self,
            index_name: str,
            api_key: str,
            cloud: str,
            region: str,
            dimension: int = 1024
    ):
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

        self.reranker = FlagReranker('BAAI/bge-reranker-large', use_fp16=True)

        self.bm25 = BM25Encoder().load("../lawApp_LangGraph/RAG_service/bm25_law_params.json")

        # 密集向量
        model_name = "BAAI/bge-large-zh-v1.5"
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            encode_kwargs={'normalize_embeddings': True}
        )

    """
    创建索引
    创建混合索引
    指定的索引方式
    @:param wait_for_completion
    @:return boolean
    """

    def create_index(self, wait_for_completion: bool = True) -> bool:
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

    """
    获取索引是否创建成功
    测试索引是否创建成功并打印统计信息
    """

    def get_index_stats(self):

        stats = self.index.describe_index_stats()
        print("当前索引的状态: ", stats)
        return True

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
    
    @:param file_path:str
    @:return 符合输入格式的列表
    """

    def get_Documents(self, file_path: str) -> list[Any] | None:

        # 加载获取
        global articles, metadata, facts_cleaned

        # 插入Pinecone数据容器
        Pinecone_records = []

        loader = TextLoader(
            file_path,
            encoding="utf-8"
        )

        # 加载一次之后可以反复使用
        # 稀疏向量
        bm25 = BM25Encoder().load("lawApp_LangGraph/RAG_service/bm25_law_params.json")

        # 密集向量
        model_name = "BAAI/bge-large-zh-v1.5"
        embeddings = HuggingFaceEmbeddings(model_name=model_name, encode_kwargs={'normalize_embeddings': True})

        # 默认读取的为Document形式
        documents = loader.load()

        for doc in documents:
            # split_text 返回 List[Document]，每个 Document 对应一个一级标题下的内容块
            articles = self.md_splitter.split_text(doc.page_content)

            # 获取元数据和正文内容
            for i, article in enumerate(articles):

                print(f'正在处理第{i}文章')

                metadata = {}

                # 提取年份：假设文件名或路径包含 202X
                year_match = re.search(r"20\d{2}", article.page_content)
                if year_match:
                    metadata["year"] = year_match.group(0) if year_match else "Unknown"

                    # 提取裁判书字号：匹配如（2023）最高法民终...号
                    case_num_pattern = r'裁判书字号[\s\\n]+((?:(?!裁判书字号)[\s\S])+?法院[\s\S]+?书)'
                    case_num_match = re.search(case_num_pattern, article.page_content)

                    metadata["case_number"] = case_num_match.group(1).strip() if case_num_match else "未识别"

                    # 提取案由：通常在字号之后，或者是特定的段落
                    case_cause_pattern = r"案由[:：]\s*([\u4e00-\u9fa5]+)"
                    cause_match = re.search(case_cause_pattern, article.page_content)

                    metadata["case_cause"] = cause_match.group(1) if cause_match else "通用"

                    # 提取基本案情
                    facts_pattern = r'【基本案情】\s*([\s\S]+?)(?=\n【|$)'
                    facts_match = re.search(facts_pattern, article.page_content)

                    # 获取捕获组内容（不含【基本案情】）
                    raw_content = facts_match.group(1)
                    # 去除空格、换行、制表符等所有空白字符，以及 # 符号
                    facts_cleaned = re.sub(r'\n+', '\n', raw_content).strip()  # 去除所有空白（空格、换行等）
                    facts_cleaned = facts_cleaned.replace('#', '')  # 去除所有 # 字符

                    chunks = self.text_splitter.split_text(facts_cleaned)

                    for chunk_id, chunk in enumerate(chunks):
                        record_id = f"Docu{i}_chunk{chunk_id}"

                        # 密集向量和稀疏向量
                        dense_vector = embeddings.embed_query(chunk)

                        # 返回 {"indices": [...], "values": [...]}
                        sparse_vector = bm25.encode_documents(chunk)

                        """
                        添加内容: id 向量数据 元数据:{年份 判决书 案由 文档切片}
                        符合Pinecone的输入格式
                        """
                        record = {
                            "id": record_id,

                            "values": dense_vector,  # 稠密向量列表 [0.12, -0.34, ...]
                            "sparse_values": sparse_vector,  # 稀疏向量字典
                            "metadata": {
                                "chunk_index": chunk_id,
                                **metadata,
                                "chunk_text": chunk,
                            }
                        }
                        print(f'第{chunk_id}个record组装完毕')

                        Pinecone_records.append(record)
                print("\n")
                print(f'第{i}文章处理完毕,已经添加至record中\n')
            return Pinecone_records

    """
        添加分块好的文本到数据库中    
        @:param 
            file_path: 文件路径
            namespace: 指定命名空间
        @:return 
        """

    def add_document(
            self,
            file_path: str,
            namespace: str,
    ):

        Pinecone_records = self.get_Documents(file_path=file_path)

        # 分批上传，每批最多 50 条向量
        batch_size = 50
        total = len(Pinecone_records)

        for i in range(0, total, batch_size):
            batch = Pinecone_records[i:i + batch_size]
            self.index.upsert(vectors=batch, namespace="Law_test_namespace")
            uploaded = min(i + batch_size, total)
            print(f"已上传 {uploaded}/{total} 条记录")

        print(self.pc.describe_index("pinecone-test-lawapp"))

        return True

    """
    接受问题
    实现检索
    混合检索(语义+关键字)
    实现重排序
    
    
    @:param query
    @:return
    
    """

    def dense_search(self, query, namespace, top_k=50):
        query_vec = self.embeddings.embed_query(query)
        return self.index.query(vector=query_vec, top_k=top_k, namespace=namespace, include_metadata=True).matches

    def sparse_search(self, query, namespace, top_k=50):
        query_sparse = self.bm25.encode_queries(query)
        return self.index.query(sparse_vector=query_sparse, top_k=top_k, namespace=namespace,
                                include_metadata=True).matches

    """
    results_lists: 多个 match 列表，每个 match 对象必须有 id 属性
    k: RRF 常数，通常取 60
    返回：按融合分数降序排列的 match 列表
    """

    def rrf_fusion(
            self,
            results_lists: list[list],
            k: int = 60,
            top_n: int = 10
    ):

        score_dict = {}
        id_to_match = {}

        for lst in results_lists:
            for rank, match in enumerate(lst):
                doc_id = match.id
                # RRF计算公式 分数累加： 1 / (k + rank + 1)
                rrf_score = 1.0 / (k + rank + 1)
                score_dict[doc_id] = score_dict.get(doc_id, 0.0) + rrf_score
                id_to_match[doc_id] = match

        # 按融合分数降序排序
        sorted_ids = sorted(
            score_dict.keys(),
            key=lambda x: score_dict[x],
            reverse=True
        )
        final_matches = []
        for doc_id in sorted_ids[:top_n]:
            match = id_to_match[doc_id]
            match.rrf_score = score_dict[doc_id]
            final_matches.append(match)

        return final_matches

    def search_documents(
            self,
            query: str,
            namespace: str,
            top_k: int = 50,
            top_n: int = 10
    ):

        # 密集向量语义查询
        dense_matches = self.dense_search(query, namespace, top_k)

        # 稀疏向量 关键字查询
        sparse_matches = self.sparse_search(query, namespace, top_k)

        # RRF融合算法
        fused = self.rrf_fusion([dense_matches, sparse_matches], k=60, top_n=top_k)

        # 融合后重排序结果
        rerank_result = self.reranker(query, fused, top_n)


        return rerank_result


load_dotenv()

# 实例化测试
service = RAG_service(
    index_name="pinecone-test-lawapp",
    api_key=os.getenv("PINECONE_API_KEY"),
    cloud="aws",
    region="us-east-1",
)
service.create_index()

service.get_index_stats()
