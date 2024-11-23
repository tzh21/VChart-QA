"""
本文件是【用LangChain快速构建基于“易速鲜花”本地知识库的智能问答系统】章节的配套代码，课程链接：https://juejin.cn/book/7387702347436130304/section/7388069933021986856
您可以点击最上方的“运行“按钮，直接运行该文件；更多操作指引请参考Readme.md文件。
"""
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

# 5. Output 问答系统的UI实现
from flask import Flask, request, render_template

app = Flask(__name__)  # Flask APP


@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        # 接收用户输入作为问题
        question = request.form.get("question")

        # RetrievalQA链 - 读入问题，生成答案
        result = qa_chain({"query": question})

        # 把大模型的回答结果返回网页进行渲染
        return render_template("index.html", result=result)

    return render_template("index.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True, port=5020)
