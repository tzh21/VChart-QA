from flask import Flask, request, render_template
from qa import getQaChain

qaChain = getQaChain()

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

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True, port=5020)
