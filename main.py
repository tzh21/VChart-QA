import os
from langchain_community.document_loaders import TextLoader
from flask import Flask, request, render_template
from qa import GetQaChain, NewVectorStore, TestQaChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI  # ChatOpenAI模型
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.chains import RetrievalQA  # RetrievalQA链
import logging
from langchain import hub
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Initialize logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

logger.info("Logger initialized")

base_dir = "./VChart-Doc/faq/en"  # 文档的存放目录
documents = []
for file in os.listdir(base_dir):
    # 构建完整的文件路径
    file_path = os.path.join(base_dir, file)
    if file.endswith(".md"):
        loader = TextLoader(file_path)
        documents.extend(loader.load())

logger.info("loaded documents")

vectorStore = NewVectorStore(documents)

retriever = vectorStore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

llm = ChatOpenAI(model=os.environ["LLM_MODELEND"], temperature=0)

# 文档
query = "Does the tooltip support changing the background color?"
retrieved_docs = retriever.invoke(query)
print("Retrieved Documents:")
for i, doc in enumerate(retrieved_docs):
    print(f"Document {i + 1}:")
    print("Content:")
    print(doc.page_content)  # 文档内容
    print("-" * 50)  # 分隔线

prompt = hub.pull("rlm/rag-prompt")
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
for chunk in rag_chain.stream(query):
    print(chunk, end="", flush=True)

# app = Flask(__name__)  # Flask APP

# @app.route("/", methods=["GET", "POST"])
# def home():
#     if request.method == "POST":
#         # 接收用户输入作为问题
#         question = request.form.get("question")

#         # RetrievalQA链 - 读入问题，生成答案
#         result = qaChain({"query": question})

#         # 把大模型的回答结果返回网页进行渲染
#         return render_template("index.html", result=result)
#     return render_template("index.html")

# app.run(host="0.0.0.0", debug=True, port=5020)
