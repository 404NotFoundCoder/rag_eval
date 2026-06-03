import pandas as pd
import requests
from tabulate import tabulate
from tqdm import tqdm
import os
from dotenv import load_dotenv

load_dotenv()


# API_URL = "http://localhost:5000/api/chat"
API_URL = "https://senpai-40.vercel.app/"
ACCESS_TOKEN = os.getenv("ACCESS_TOKEN")

user_input_list = [
    "從系統分析與設計課程中遇到甚麼問題?如何解決?",
    "從系統開發中遇到甚麼問題?如何解決?",
    "使用 git 嗎?遇到甚麼問題?",
    "使用哪個語言、資料庫開發?為何?優缺點是?",
    "如何定期追蹤組員進度?遇到甚麼問題?",
    "(到目前為止,)專題遇到的最大問題是甚麼?如何解決?",
    "(到目前為止,)系統開發中遇到甚麼問題?如何解決? (專題)",
    "使用 git 嗎?遇到甚麼問題? (專題)",
    "使用哪個語言、資料庫開發?跟 SA 所使用的一樣嗎?為何?優缺點是? (專題)",
    "如何定期追蹤組員進度? 跟 SA 一樣嗎?遇到甚麼問題? (專題)",
    "對各位的建議是?",
]

retrieved_contexts_list = []
response_list = []
reference_list = []
retrieved_metadata_list = []

status_list = []

for user_input in tqdm(user_input_list, desc="Processing", unit="q"):
    try:
        res = requests.post(
            API_URL,
            headers={"Content-Type": "application/json"},
            json={"message": user_input, "accessToken": ACCESS_TOKEN, "history": []},
            timeout=30,
        )

        res.raise_for_status()
        data = res.json()

        answer = data.get("answer", "")
        references = data.get("references", [])

        retrieved_contexts = [ref.get("content", "") for ref in references]

        retrieved_metadata = [
            {"source": ref.get("source"), "id": ref.get("id")} for ref in references
        ]

        status = "success"

    except Exception as e:
        print(f"\n❌ 失敗：{user_input}")
        print(e)

        answer = ""
        retrieved_contexts = []
        retrieved_metadata = []
        status = "fail"

    retrieved_contexts_list.append(retrieved_contexts)
    response_list.append(answer)
    reference_list.append(answer)
    retrieved_metadata_list.append(retrieved_metadata)
    status_list.append(status)

df = pd.DataFrame(
    {
        "user_input": user_input_list,
        "retrieved_contexts": retrieved_contexts_list,
        "response": response_list,
        "reference": reference_list,
        "retrieved_metadata": retrieved_metadata_list,
        "status": status_list,
    }
)

print(tabulate(df, headers="keys", tablefmt="fancy_grid", showindex=False))
df.to_csv("chat_eval_result_hybrid_v1.csv", index=False, encoding="utf-8-sig")
