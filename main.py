import os
from flask import Flask, request, render_template
from vector_store import NewVectorStore
from langchain_openai import ChatOpenAI
import logging
from loader import LoadDocs
from rag import NewRagChain

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

docsPath = "./VChart-Doc/faq/en"
documents = LoadDocs(docsPath)

vectorStore = NewVectorStore(documents)

retriever = vectorStore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

llm = ChatOpenAI(model=os.environ["LLM_MODELEND"], temperature=0)

query = "How can I customize the display of fields in data in Tooltip in VChart?"

ragChain = NewRagChain(retriever, llm)

result = ragChain.invoke(query)
print(result)

# app = Flask(__name__)  # Flask APP

# @app.route("/", methods=["GET", "POST"])
# def home():
#     if request.method == "POST":
#         # 接收用户输入作为问题
#         question = request.form.get("question")

#         # RetrievalQA链 - 读入问题，生成答案
#         result = ragChain.invoke(question)

#         # 把大模型的回答结果返回网页进行渲染
#         return render_template("index.html", result=result)
#     return render_template("index.html")

# app.run(host="0.0.0.0", debug=True, port=5020)
