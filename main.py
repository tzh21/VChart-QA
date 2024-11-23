import os
from langchain_community.document_loaders import TextLoader
from flask import Flask, request, render_template
from qa import GetQaChain, GetVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter

base_dir = "./VChart-Doc/faq/en"  # 文档的存放目录
documents = []
for file in os.listdir(base_dir):
    # 构建完整的文件路径
    file_path = os.path.join(base_dir, file)
    if file.endswith(".md"):
        loader = TextLoader(file_path)
        documents.extend(loader.load())

text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=10)
chunked_documents = text_splitter.split_documents(documents)

vectorStore = GetVectorStore(chunked_documents, "http://183.172.21.164:6333")
qaChain = GetQaChain(vectorStore)

app = Flask(__name__)  # Flask APP

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        # 接收用户输入作为问题
        question = request.form.get("question")

        # RetrievalQA链 - 读入问题，生成答案
        result = qaChain({"query": question})

        # 把大模型的回答结果返回网页进行渲染
        return render_template("index.html", result=result)
    return render_template("index.html")

app.run(host="0.0.0.0", debug=True, port=5020)
