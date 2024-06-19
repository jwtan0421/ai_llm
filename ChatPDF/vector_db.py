import chromadb
from chromadb.config import Settings

class MyVectorDBConnector:
    def __init__(self, collection_name, embedding_fn):
        """
        初始化Chroma类的实例。

        这个初始化过程主要包括：
        1. 连接到Chroma数据库客户端。
        2. 重置数据库以确保干净的启动状态。
        3. 创建或获取一个指定名称的集合(collection)。
        4. 设置嵌入函数(embedding function)。

        参数:
        collection_name: 字符串，表示集合的名称，用于存储和检索嵌入向量。
        embedding_fn: 函数，用于计算给定输入的嵌入向量。
        """
        # 初始化Chroma数据库客户端，允许在必要时重置数据库状态。
        chroma_client = chromadb.Client(Settings(allow_reset=True))

        # 重置数据库以清除之前的设置和数据，确保每次初始化都是干净的环境。
        # 为了演示，实际不需要每次 reset()
        chroma_client.reset()

        # 获取或创建一个指定名称的集合，用于后续的嵌入向量存储和检索。
        # 创建一个 collection
        self.collection = chroma_client.get_or_create_collection(name=collection_name)
        # 设置用于计算嵌入向量的函数。
        self.embedding_fn = embedding_fn

    def add_documents(self, documents):
        '''向 collection 中添加文档与向量'''
        self.collection.add(
            embeddings=self.embedding_fn(documents),  # 每个文档的向量
            documents=documents,  # 文档的原文
            ids=[f"id{i}" for i in range(len(documents))]  # 每个文档的 id
        )

    def search(self, query, top_n):
        '''检索向量数据库'''
        results = self.collection.query(
            query_embeddings=self.embedding_fn([query]),
            n_results=top_n
        )
        return results
    

    def collection_size(self):
        '''返回 collection 中的文档数量'''
        return self.collection.count()
