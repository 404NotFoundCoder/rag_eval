"""
fetch_conversations.py
----------------------
從 Firebase users collection 讀取 conversation-0610 子集合
產生兩個 JSON 檔：
  1. all_conversations.json  — 每位 user 完整對話（按 createdAt asc）
  2. user_questions.json     — 只保留使用者問題（按對話 createdAt asc）

messagePairs 欄位結構（依照 Firebase 實際資料）：
  - user       : 使用者問題 (string)
  - ai         : AI 回覆 (string)
  - metadata   : RAG 來源文件摘要 (string)
  - references : 引用文件清單 (array of map)
"""

import os
import json
from datetime import datetime
import firebase_admin
from firebase_admin import credentials, firestore
from dotenv import load_dotenv

load_dotenv()

COLLECTION_NAME = "conversation-0610"


# ─── Firebase 初始化 ─────────────────────────────


def init_firebase():
    if not firebase_admin._apps:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        key_path = os.path.join(base_dir, "serviceAccountKey.json")
        if not os.path.exists(key_path):
            raise FileNotFoundError(f"找不到 Firebase key：{key_path}")
        cred = credentials.Certificate(key_path)
        firebase_admin.initialize_app(cred)
    return firestore.client()


# ─── 時間戳工具 ─────────────────────────────


def to_sortable_ts(val) -> float:
    if val is None:
        return 0.0
    if hasattr(val, "timestamp"):
        return val.timestamp()
    if isinstance(val, (int, float)):
        return float(val)
    return 0.0


def ts_to_iso(val) -> str | None:
    if val is None:
        return None
    try:
        if hasattr(val, "timestamp"):
            return datetime.utcfromtimestamp(val.timestamp()).isoformat() + "Z"
        if isinstance(val, (int, float)):
            return datetime.utcfromtimestamp(float(val)).isoformat() + "Z"
    except Exception:
        pass
    return str(val)


# ─── messagePairs 序列化 ─────────────────────────────


def serialize_pair(pair: dict) -> dict:
    """
    序列化一個 messagePair map。
    - Firestore Timestamp → ISO string
    - references array 裡的每個 map 也一併序列化
    """
    result = {}
    for k, v in pair.items():
        if hasattr(v, "timestamp"):
            result[k] = ts_to_iso(v)
        elif isinstance(v, list):
            result[k] = [
                (
                    {
                        ik: (ts_to_iso(iv) if hasattr(iv, "timestamp") else iv)
                        for ik, iv in item.items()
                    }
                    if isinstance(item, dict)
                    else item
                )
                for item in v
            ]
        else:
            result[k] = v
    return result


# ─── 讀取所有對話 ─────────────────────────────


def fetch_all_conversations(db) -> dict:
    """
    回傳結構：
    {
      "<uid>": [
        {
          "doc_id": "...",
          "createdAt": "2026-03-29T14:14:24Z",
          "updatedAt": "...",
          "messagePairs": [
            {
              "user": "使用者問題",
              "ai": "AI 回覆",
              "metadata": "RAG 來源文字",
              "references": [
                {"id": "...", "source": "...", "content": "..."},
                ...
              ]
            },
            ...
          ]
        },
        ...  ← 按 createdAt asc
      ]
    }
    """
    print(f"📥 讀取 users，子集合：{COLLECTION_NAME} ...")
    result = {}
    user_docs = list(db.collection("users").stream())
    print(f"👥 共 {len(user_docs)} 位 user")

    for user_doc in user_docs:
        uid = user_doc.id
        conv_docs = list(
            db.collection("users").document(uid).collection(COLLECTION_NAME).stream()
        )
        if not conv_docs:
            continue

        conversations = []
        for doc in conv_docs:
            data = doc.to_dict() or {}
            created_raw = data.get("createdAt") or data.get("created_at")
            updated_raw = data.get("updatedAt") or data.get("updated_at")

            pairs = [
                serialize_pair(p)
                for p in data.get("messagePairs", [])
                if isinstance(p, dict)
            ]

            conversations.append(
                {
                    "doc_id": doc.id,
                    "_sort_ts": to_sortable_ts(created_raw),
                    "createdAt": ts_to_iso(created_raw),
                    "updatedAt": ts_to_iso(updated_raw),
                    "messagePairs": pairs,
                }
            )

        # 按 createdAt asc 排序
        conversations.sort(key=lambda c: c["_sort_ts"])
        for c in conversations:
            c.pop("_sort_ts", None)

        result[uid] = conversations
        total_pairs = sum(len(c["messagePairs"]) for c in conversations)
        print(f"  ✅ {uid}：{len(conversations)} 筆對話，{total_pairs} 個問答對")

    return result


# ─── 萃取使用者問題 ─────────────────────────────


def extract_user_questions(all_conversations: dict) -> dict:
    """
    回傳結構：
    {
      "<uid>": [
        {
          "doc_id": "conversation doc id",
          "conversation_createdAt": "ISO string",
          "pair_index": 0,
          "question": "使用者問題文字"
        },
        ...  ← 依對話 createdAt asc，同對話內依 pair_index asc
      ]
    }
    """
    result = {}
    for uid, conversations in all_conversations.items():
        questions = []
        for conv in conversations:
            for idx, pair in enumerate(conv.get("messagePairs", [])):
                q = (pair.get("user") or "").strip()
                if q:
                    questions.append(
                        {
                            "doc_id": conv["doc_id"],
                            "conversation_createdAt": conv.get("createdAt"),
                            "pair_index": idx,
                            "question": q,
                        }
                    )
        if questions:
            result[uid] = questions
            print(f"  💬 {uid}：{len(questions)} 個問題")
    return result


# ─── 存 JSON ─────────────────────────────


def save_json(data: dict, filename: str):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"✅ 已儲存：{filename}")


# ─── 主程式 ─────────────────────────────

if __name__ == "__main__":
    db = init_firebase()

    all_conversations = fetch_all_conversations(db)
    save_json(all_conversations, "all_conversations.json")

    print("\n📝 萃取使用者問題...")
    user_questions = extract_user_questions(all_conversations)
    save_json(user_questions, "user_questions.json")

    total_users = len(user_questions)
    total_q = sum(len(v) for v in user_questions.values())
    print(f"\n📊 {total_users} 位 user，共 {total_q} 個問題")
    print("🎉 完成！all_conversations.json / user_questions.json")
