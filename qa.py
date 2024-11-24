import logging
import os

from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from embedding import DoubaoEmbeddings
from langchain_core.documents.base import Document
from langchain_openai import ChatOpenAI  # ChatOpenAI模型
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.chains import RetrievalQA  # RetrievalQA链

logger = logging.getLogger(__name__)

def NewVectorStore(documents: list[Document]) -> Chroma:
    vectorStore = Chroma(
        embedding_function=DoubaoEmbeddings(model=os.environ["EMBEDDING_MODELEND"]),
        persist_directory="storage"
    )
    logger.info("chroma initialized")

    # 获取已有文档的文件路径
    existing_file_paths = set()
    for doc in vectorStore.get()["metadatas"]:  # 提取存储中的元数据
        if "file_path" in doc:
            existing_file_paths.add(doc["file_path"])

    unique_documents = []
    for doc in documents:
        file_path = doc.metadata.get("file_path")  # 假设 metadata 中包含 file_path
        if file_path and file_path not in existing_file_paths:
            unique_documents.append(doc)
            existing_file_paths.add(file_path)  # 记录已处理的路径

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=10)
    chunked_documents = text_splitter.split_documents(unique_documents)
    logger.info("documents split")

    if len(unique_documents) > 0:
        vectorStore.add_documents(chunked_documents)
    logger.info("documents added")
    return vectorStore

def GetQaChain(vectorStore: Chroma):
    # 实例化一个大模型工具 - OpenAI的GPT-3.5
    llm = ChatOpenAI(model=os.environ["LLM_MODELEND"], temperature=0)

    # 实例化一个MultiQueryRetriever
    retriever_from_llm = MultiQueryRetriever.from_llm(
        retriever=vectorStore.as_retriever(), llm=llm
    )

    # 实例化一个RetrievalQA链
    qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever_from_llm)

    return qa_chain

def TestQaChain(vectorStore: Chroma) -> list[Document]:
    retriever = vectorStore.as_retriever()
    docs = retriever.invoke("Does the tooltip support changing the background color?")
    return docs