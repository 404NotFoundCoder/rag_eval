import json
import os

import pandas as pd
import requests
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

# ===== 設定這裡 =====
# RETRY_FAILED_ONLY = True  # True = 只跑 status == fail；False = 跑全部
RETRY_FAILED_ONLY = False  # True = 只跑 status == fail；False = 跑全部
# ====================

API_URL = "https://senpai-40.vercel.app/api/chat"

ACCESS_TOKENS = [
    os.getenv("ACCESS_TOKEN"),
    os.getenv("ACCESS_TOKEN_1"),
    os.getenv("ACCESS_TOKEN_2"),
]

INPUT_CSV = "ragas_testset_sa_student_2026-05-22.csv"
# INPUT_CSV = "ragas_eval_with_response_2026-05-22.csv"
OUTPUT_CSV = "ragas_eval_with_response_2026-05-22.csv"


df = pd.read_csv(INPUT_CSV, encoding="utf-8-sig")


# 檢查必要欄位
required_columns = ["user_input", "reference"]
for col in required_columns:
    if col not in df.columns:
        raise ValueError(f"CSV 缺少必要欄位：{col}")


# 決定要跑哪些資料
if RETRY_FAILED_ONLY:
    if "status" not in df.columns:
        raise ValueError("CSV 沒有 status 欄位，無法使用 RETRY_FAILED_ONLY=True")

    df_to_run = df[df["status"] == "fail"]
    print(f"🔄 只重跑 status == fail 的資料，共 {len(df_to_run)} 筆")

else:
    df_to_run = df
    print(f"🚀 跑全部資料，共 {len(df_to_run)} 筆")


def is_quota_error(error_text: str, status_code=None):
    """
    判斷是否為額度、速率限制、quota 類錯誤
    """
    error_text = str(error_text).lower()

    quota_keywords = [
        "quota",
        "rate limit",
        "rate_limit",
        "too many requests",
        "insufficient_quota",
        "exceeded",
        "429",
    ]

    if status_code == 429:
        return True

    return any(keyword in error_text for keyword in quota_keywords)


current_key_index = 0


for idx, row in tqdm(
    df_to_run.iterrows(), total=len(df_to_run), desc="Processing", unit="q"
):
    user_input = row["user_input"]

    response = ""
    retrieved_contexts = []
    retrieved_metadata = []
    status = "fail"
    success = False

    for retry_count in range(len(ACCESS_TOKENS)):
        access_token = ACCESS_TOKENS[current_key_index]

        try:
            res = requests.post(
                API_URL,
                headers={"Content-Type": "application/json"},
                json={
                    "message": user_input,
                    "accessToken": access_token,
                    "history": [],
                },
                timeout=60,
            )

            # 如果 HTTP 狀態碼不是 2xx，先判斷是不是 quota / rate limit
            if not res.ok:
                error_text = res.text

                if is_quota_error(error_text, res.status_code):
                    print(
                        f"\n⚠️ Key {current_key_index + 1} 額度或限制錯誤，切換下一隻 key"
                    )
                    current_key_index = (current_key_index + 1) % len(ACCESS_TOKENS)
                    continue

                # 不是 quota 類錯誤，就直接拋出
                res.raise_for_status()

            data = res.json()

            response = data.get("answer", "")
            references = data.get("references", [])

            retrieved_contexts = [ref.get("content", "") for ref in references]

            retrieved_metadata = [
                {
                    "source": ref.get("source"),
                    "id": ref.get("id"),
                }
                for ref in references
            ]

            status = "success"
            success = True
            break

        except Exception as e:
            error_text = str(e)

            if is_quota_error(error_text):
                print(f"\n⚠️ Key {current_key_index + 1} 可能額度不足，切換下一隻 key")
                current_key_index = (current_key_index + 1) % len(ACCESS_TOKENS)
                continue

            print(f"\n❌ 失敗：{user_input}")
            print(e)
            status = "fail"
            break

    if not success:
        print(f"\n❌ 此問題所有 key 都嘗試失敗：{user_input}")

    # 直接寫回原本 df
    # RETRY_FAILED_ONLY=True 時，只會更新 fail 的 row
    # RETRY_FAILED_ONLY=False 時，會更新全部 row
    df.at[idx, "response"] = response
    df.at[idx, "retrieved_contexts"] = json.dumps(
        retrieved_contexts, ensure_ascii=False
    )
    df.at[idx, "retrieved_metadata"] = json.dumps(
        retrieved_metadata, ensure_ascii=False
    )
    df.at[idx, "status"] = status


df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

print(f"\n✅ 已輸出：{OUTPUT_CSV}")
