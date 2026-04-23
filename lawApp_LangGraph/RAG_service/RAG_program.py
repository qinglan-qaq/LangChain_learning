import os
import re
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from pinecone import Pinecone, ServerlessSpec
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownTextSplitter, MarkdownHeaderTextSplitter

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
            metric: str,
            cloud: str,
            region: str,
            dimension: int,
            embeding_model: str,
    ):
        # 配置数据库信息
        self.index_name = index_name
        self.api_key = api_key
        self.metric = metric
        self.cloud = cloud
        self.region = region
        self.dimension = dimension
        # 选择嵌入模型
        self.embeding_model = embeding_model

        # 初始化 Pinecone 客户端
        self.pc = Pinecone(api_key=self.api_key)

        # 3. 检查索引是否存在，如果不存在则创建
        if self.index_name not in self.pc.list_indexes().names():
            print(f"正在创建索引: {self.index_name}...")
            self.pc.create_index(
                name=self.index_name,
                dimension=self.dimension,
                metric=self.metric,
                spec=ServerlessSpec(
                    cloud=self.cloud,
                    region=self.region
                )
            )

        # 4. 初始化并保存索引操作对象
        self.index = self.pc.Index(self.index_name)

        print(f"索引 '{self.index_name}' 已就绪。")

        # md文档按照标题层级读取
        headers_to_split_on = [
            ("#", "Header_1")
        ]
        self.md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

        # 递归分块
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=120,
            separators=["\n\n", "\n", "。", "；", " "],  # 优先按段落切
            add_start_index=True
        )

    def get_index_stats(self):
        """测试索引是否创建成功"""
        print(self.index.get_index_stats())
        return True

    """
    添加文本到数据库中需要:
    加载文本,
    文本分块,
    向量化
    插入数据到namespace
    
    具体实施:
    (正则表达式)
    将清洗好的文件加载
    以每一个案例为单位
    先提取元数据
    (分块)
    从[基本案情]到最后的内容提取
    以句子为单位,合成一大段
    
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

        # 指定
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

                # 获取捕获组内容（不含【基本案情】）
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


@:param query
@:return

"""
def search_documents(self, query: str):
    pass


load_dotenv()

# 实例化测试
service = RAG_service(
    index_name="my-rag-index",
    api_key=os.getenv("PINECONE_API_KEY"),
    metric="cosine",
    cloud="aws",
    region="us-east-1",
    dimension=1536,  # OpenAI text-embedding-3-small 默认维度
    embeding_model="text-embedding-3-small"
)

service.add_document(
    "lawApp_LangGraph/MarkDownFiles/中国法院2019年度案例：婚姻家庭与继承纠纷.md",
    encoding="utf-8"
)
