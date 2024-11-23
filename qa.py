# 1.Load 导入Document Loaders
import os
from langchain_community.document_loaders import TextLoader
import logging
from embedding import DoubaoEmbeddings

# 设置日志
logging.basicConfig(format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# 加载Documents
base_dir = "./VChart-Doc/faq/en"  # 文档的存放目录
documents = []
for file in os.listdir(base_dir):
    # 构建完整的文件路径
    file_path = os.path.join(base_dir, file)
    if file.endswith(".md"):
        loader = TextLoader(file_path)
        documents.extend(loader.load())

# 2.Split 将Documents切分成块以便后续进行嵌入和向量存储
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=10)
chunked_documents = text_splitter.split_documents(documents)


# 3.Store 将分割嵌入并存储在矢量数据库Qdrant中
from langchain_community.vectorstores import Qdrant

vectorstore = Qdrant.from_documents(
    documents=chunked_documents,  # 以分块的文档
    embedding=DoubaoEmbeddings(
        model=os.environ["EMBEDDING_MODELEND"],
    ),  # 用OpenAI的Embedding Model做嵌入
    location=":memory:",  # in-memory 存储
    # location="storage",  # in-memory 存储
    collection_name="my_documents",
)  # 指定collection_name

# 4. Retrieval 准备模型和Retrieval链
import logging  # 导入Logging工具
from langchain_openai import ChatOpenAI  # ChatOpenAI模型
from langchain.retrievers.multi_query import (
    MultiQueryRetriever,
)  # MultiQueryRetriever工具
from langchain.chains import RetrievalQA  # RetrievalQA链

# 设置Logging
logging.basicConfig()
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)

# 实例化一个大模型工具 - OpenAI的GPT-3.5
llm = ChatOpenAI(model=os.environ["LLM_MODELEND"], temperature=0)

# 实例化一个MultiQueryRetriever
retriever_from_llm = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(), llm=llm
)

# 实例化一个RetrievalQA链
qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever_from_llm)