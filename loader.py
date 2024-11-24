import os
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents.base import Document

def LoadDocs(docsPath) -> list[Document]:
    documents = []
    for file in os.listdir(docsPath):
        file_path = os.path.join(docsPath, file)
        if file.endswith(".md"):
            loader = TextLoader(file_path)
            documents.extend(loader.load())
    return documents
