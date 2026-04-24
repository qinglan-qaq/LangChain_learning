import os
import re
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader

from pinecone import Pinecone, ServerlessSpec
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter

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

import time
from pinecone import Pinecone, ServerlessSpec


class RAG_service:
    def __init__(
            self,
            index_name: str,
            api_key: str,
            metric: str,
            cloud: str,
            region: str,
            dimension: int = 1024
    ):
        self.index_name = index_name
        self.api_key = api_key
        self.metric = metric
        self.cloud = cloud
        self.region = region
        self.dimension = dimension

        self.pc = None
        self.index = None

        headers_to_split_on = [("#", "Header_1")]
        self.md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=120,
            separators=["\n\n", "\n", "。", "；", " "],
            add_start_index=True
        )

    def create_index(self) -> bool:
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
                vector_type="dense",  # 存储稠密向量，同时接受稀疏向量
            )

        else:
            print(f"索引 '{self.index_name}' 已存在。")

        self.index = self.pc.Index(self.index_name)
        print(f"索引 '{self.index_name}' 已就绪。")

        return True

    # def get_index_stats(self):
    #     """测试索引是否创建成功"""
    #     print(self.index.get_index_stats())
    #     return True

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

        records = self.get_Documents(file_path=file_path)

        # 指定命名空间添加数据
        self.index.upsert_records(self, namespace, records)

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
    
    @:param file_path:
    
    @:return
    
    """

    def get_Documents(self, file_path: str):
        # 加载获取
        loader = TextLoader(file_path=file_path, encoding="utf-8")
        documents = loader.load()

        for doc in documents:
            # split_text 返回 List[Document]，每个 Document 对应一个一级标题下的内容块
            articles = self.md_splitter.split_text(doc.page_content)

        for i, article in enumerate(articles):

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

                # 获取捕获组内容
                raw_content = facts_match.group(1)
                # 去除空格、换行、制表符等所有空白字符，以及 # 符号
                facts_cleaned = re.sub(r'\n+', '\n', raw_content).strip()  # 去除所有空白（空格、换行等）
                facts_cleaned = facts_cleaned.replace('#', '')

                # 插入Pinecone数据容器
                Pinecone_records = []

                chunks = self.text_splitter.split_text(facts_cleaned)

                for i, chunk in enumerate(chunks):
                    record_id = f"doc_chunk_{i}"

                    record = {
                        "id": record_id,
                        "values": [],  # 这里后续需要填入 embedding_model.embed_query(chunk) 的结果
                        "metadata": {
                            **metadata,  # 展开原始元数据
                            "text": chunk,  # 必须把原始文本存入 metadata，否则检索后拿不到原文
                            "chunk_index": i  # 记录分块序号
                        }
                    }
                    Pinecone_records.append(record)

            return Pinecone_records

    """
    接受问题
    实现检索
    混合检索(语义+关键字)
    实现重排序
    
    
    @:param query
    @:return
    
    """

    def search_documents(self, query: str):

        return


load_dotenv()

# 实例化测试
service = RAG_service(
    index_name="Pinecone_test_lawApp",
    api_key=os.getenv("PINECONE_API_KEY"),
    metric="cosine",
    cloud="aws",
    region="us-east-1",
)
service.create_index()

# service.add_document(
#     "lawApp_LangGraph/MarkDownFiles/中国法院2019年度案例：婚姻家庭与继承纠纷.md",
#     encoding="utf-8"
# )
