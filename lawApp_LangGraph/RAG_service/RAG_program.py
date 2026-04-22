import os

from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

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
    
    """

    def add_document(
            self,
            file_path: str,
    ):


        pass


"""

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
# 测试连接
service.get_index_stats()
