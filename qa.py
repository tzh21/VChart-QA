# 1.Load 导入Document Loaders
import os
from embedding import DoubaoEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain_core.documents.base import Document
from langchain_openai import ChatOpenAI  # ChatOpenAI模型
from langchain.retrievers.multi_query import (
    MultiQueryRetriever,
)  # MultiQueryRetriever工具
from langchain.chains import RetrievalQA  # RetrievalQA链

def GetVectorStore(documents: list[Document], url: str) -> Qdrant:
    vectorstore = Qdrant.from_documents(
        documents=documents,
        embedding=DoubaoEmbeddings(
            model=os.environ["EMBEDDING_MODELEND"],
        ),
        url=url,
        # location=":memory:",  # in-memory 存储
        collection_name="qa_doc",
    )
    return vectorstore

# 4. Retrieval 准备模型和Retrieval链

def GetQaChain(vectorStore: Qdrant):
    # 实例化一个大模型工具 - OpenAI的GPT-3.5
    llm = ChatOpenAI(model=os.environ["LLM_MODELEND"], temperature=0)

    # 实例化一个MultiQueryRetriever
    retriever_from_llm = MultiQueryRetriever.from_llm(
        retriever=vectorStore.as_retriever(), llm=llm
    )

    # 实例化一个RetrievalQA链
    qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever_from_llm)

    return qa_chain
