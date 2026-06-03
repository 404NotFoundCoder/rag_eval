# import os
# import firebase_admin
# from firebase_admin import credentials, firestore
# from dotenv import load_dotenv

# from langchain_core.documents import Document
# from langchain_openai import ChatOpenAI
# from ragas.llms import LangchainLLMWrapper
# from ragas.embeddings import OpenAIEmbeddings
# from ragas.testset import TestsetGenerator
# from ragas.testset.synthesizers.single_hop.specific import (
#     SingleHopSpecificQuerySynthesizer,
# )
# from ragas.testset.synthesizers.multi_hop.specific import (
#     MultiHopSpecificQuerySynthesizer,
# )
# import openai

# # 讀取 .env
# load_dotenv()


# # ─── Firebase 初始化 ─────────────────────────────


# def init_firebase():
#     if not firebase_admin._apps:
#         base_dir = os.path.dirname(os.path.abspath(__file__))
#         key_path = os.path.join(base_dir, "serviceAccountKey.json")

#         if not os.path.exists(key_path):
#             raise FileNotFoundError(f"找不到 Firebase key：{key_path}")

#         cred = credentials.Certificate(key_path)
#         firebase_admin.initialize_app(cred)

#     return firestore.client()


# # ─── 留言格式化 ─────────────────────────────


# def format_comments_for_rag(comments: list[dict]) -> str | None:
#     if not comments:
#         return None

#     roots = [c for c in comments if c.get("parent_id") is None]
#     segments = []

#     for root in roots:
#         name = root.get("author", {}).get("name", "匿名")
#         label = root.get("floor_label", "")
#         content = root.get("content", "")

#         if content:
#             segments.append(f"{name}({label})說 {content}")

#         replies = sorted(
#             [r for r in comments if r.get("parent_id") == root["id"]],
#             key=lambda r: r.get("created_at") or 0,
#         )

#         for reply in replies:
#             r_name = reply.get("author", {}).get("name", "匿名")
#             r_label = reply.get("floor_label", "")
#             r_content = reply.get("content", "")

#             if r_content:
#                 segments.append(f"{r_name}({r_label})說 {r_content}")

#     if not segments:
#         return None

#     return "留言說：" + "\n".join(segments)


# def fetch_comments(db, post_id: str) -> list[dict]:
#     comments = []

#     for doc in db.collection("posts").document(post_id).collection("comments").stream():
#         data = doc.to_dict() or {}
#         author = data.get("author", {})

#         comments.append(
#             {
#                 "id": doc.id,
#                 "parent_id": data.get("parent_id"),
#                 "floor_label": data.get("floor_label", ""),
#                 "content": data.get("content", ""),
#                 "author": {"name": author.get("name", "匿名")},
#                 "created_at": data.get("created_at"),
#             }
#         )

#     return comments


# # ─── 從 Firebase 讀 posts ─────────────────────────────


# def fetch_posts_from_firebase() -> list[dict]:
#     db = init_firebase()
#     print("📥 讀取 Firebase /posts ...")

#     posts = []

#     for doc in db.collection("posts").stream():
#         data = doc.to_dict() or {}

#         title = data.get("title", "")
#         content = data.get("content", "")

#         if not content:
#             print(f"⚠️ 跳過 {doc.id}（content 空）")
#             continue

#         raw_comments = fetch_comments(db, doc.id)
#         comment_str = format_comments_for_rag(raw_comments)

#         posts.append(
#             {
#                 "id": doc.id,
#                 "title": title,
#                 "content": content,
#                 "comment": comment_str,
#             }
#         )

#     print(f"✅ 共 {len(posts)} 筆文件")
#     return posts


# # ─── posts → LangChain Documents ─────────────────────────────


# def posts_to_langchain_docs(posts: list[dict]) -> list[Document]:
#     docs = []

#     for post in posts:
#         title = post.get("title", "")
#         content = post.get("content", "")
#         comment = post.get("comment")

#         page_content = f"標題：{title}\n\n內文：{content}"

#         if comment:
#             page_content += f"\n\n{comment}"

#         docs.append(
#             Document(
#                 page_content=page_content,
#                 metadata={
#                     "post_id": post["id"],
#                     "title": title,
#                     "source": f"firebase/posts/{post['id']}",
#                 },
#             )
#         )

#     return docs


# # ─── Ragas 產生測試集 ─────────────────────────────


# def generate_ragas_testset(testset_size: int = 10):
#     if not os.getenv("OPENAI_API_KEY"):
#         raise RuntimeError("缺少 OPENAI_API_KEY，請確認 .env 是否已設定")

#     posts = fetch_posts_from_firebase()
#     docs = posts_to_langchain_docs(posts)

#     print(f"📄 轉換完成，共 {len(docs)} 份 LangChain Documents")

#     generator_llm = LangchainLLMWrapper(
#         ChatOpenAI(
#             model="gpt-4o-mini",
#             temperature=0,
#         )
#     )

#     openai_client = openai.OpenAI()

#     generator_embeddings = OpenAIEmbeddings(
#         client=openai_client,
#         model="text-embedding-3-small",
#     )

#     generator = TestsetGenerator(
#         llm=generator_llm,
#         embedding_model=generator_embeddings,
#     )

#     print(f"⚙️ 開始產生 Ragas testset，共 {testset_size} 筆...")

#     query_distribution = [
#         (SingleHopSpecificQuerySynthesizer(llm=generator_llm), 0.8),
#         (MultiHopSpecificQuerySynthesizer(llm=generator_llm), 0.2),
#     ]

#     dataset = generator.generate_with_langchain_docs(
#         docs,
#         testset_size=testset_size,
#         query_distribution=query_distribution,
#     )

#     print("✅ Testset 產生完成")
#     return dataset


# if __name__ == "__main__":
#     dataset = generate_ragas_testset(testset_size=10)

#     # 轉成 pandas dataframe
#     df = dataset.to_pandas()

#     # 輸出 CSV
#     output_path = OUTPUT_PATH
#     df.to_csv(output_path, index=False, encoding="utf-8-sig")

#     print(f"✅ 已輸出：{output_path}")
#     print(df.head())
import asyncio
import sys
from datetime import datetime, timedelta, timezone

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import json
import os
from pathlib import Path
from zoneinfo import ZoneInfo

import firebase_admin
import openai
from dotenv import load_dotenv
from firebase_admin import credentials, firestore
from langchain_core.documents import Document
from ragas.embeddings import OpenAIEmbeddings
from ragas.llms import llm_factory
from ragas.testset import TestsetGenerator
from ragas.testset.persona import Persona
from ragas.testset.synthesizers.multi_hop.specific import (
    MultiHopSpecificQuerySynthesizer,
)
from ragas.testset.synthesizers.single_hop.specific import (
    SingleHopSpecificQuerySynthesizer,
)

load_dotenv()

TAIPEI_TZ = ZoneInfo("Asia/Taipei")
TODAY_STR = datetime.now(TAIPEI_TZ).strftime("%Y-%m-%d")

DOCS_CACHE_PATH = Path(f"docs_cache_{TODAY_STR}.json")
OUTPUT_PATH = f"ragas_testset_sa_student_{TODAY_STR}.csv"

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


# ─── 留言格式化 ─────────────────────────────


def format_comments_for_rag(comments: list[dict]) -> str | None:
    if not comments:
        return None

    roots = [c for c in comments if c.get("parent_id") is None]
    segments = []

    for root in roots:
        name = root.get("author", {}).get("name", "匿名")
        label = root.get("floor_label", "")
        content = root.get("content", "")

        if content:
            segments.append(f"{name}({label})說 {content}")

        replies = sorted(
            [r for r in comments if r.get("parent_id") == root["id"]],
            key=lambda r: r.get("created_at") or 0,
        )

        for reply in replies:
            r_name = reply.get("author", {}).get("name", "匿名")
            r_label = reply.get("floor_label", "")
            r_content = reply.get("content", "")

            if r_content:
                segments.append(f"{r_name}({r_label})說 {r_content}")

    if not segments:
        return None

    return "留言說：" + "\n".join(segments)


def fetch_comments(db, post_id: str) -> list[dict]:
    comments = []

    for doc in db.collection("posts").document(post_id).collection("comments").stream():
        data = doc.to_dict() or {}
        author = data.get("author", {})

        comments.append(
            {
                "id": doc.id,
                "parent_id": data.get("parent_id"),
                "floor_label": data.get("floor_label", ""),
                "content": data.get("content", ""),
                "author": {"name": author.get("name", "匿名")},
                "created_at": data.get("created_at"),
            }
        )

    return comments


# ─── 從 Firebase 讀 posts ─────────────────────────────


def fetch_posts_from_firebase() -> list[dict]:
    db = init_firebase()
    print("📥 讀取 Firebase /posts ...")

    posts = []

    for doc in db.collection("posts").stream():
        data = doc.to_dict() or {}

        title = data.get("title", "")
        content = data.get("content", "")

        if not content:
            print(f"⚠️ 跳過 {doc.id}（content 空）")
            continue

        raw_comments = fetch_comments(db, doc.id)
        comment_str = format_comments_for_rag(raw_comments)

        posts.append(
            {
                "id": doc.id,
                "title": title,
                "content": content,
                "comment": comment_str,
            }
        )

    print(f"✅ 共 {len(posts)} 筆文件")
    return posts


# ─── posts → LangChain Documents ─────────────────────────────


def posts_to_langchain_docs(posts: list[dict]) -> list[Document]:
    docs = []

    for post in posts:
        title = post.get("title", "")
        content = post.get("content", "")
        comment = post.get("comment")

        page_content = f"標題：{title}\n\n內文：{content}"

        if comment:
            page_content += f"\n\n{comment}"

        docs.append(
            Document(
                page_content=page_content,
                metadata={
                    "post_id": post["id"],
                    "title": title,
                    "source": f"firebase/posts/{post['id']}",
                },
            )
        )

    return docs


def save_docs_cache(posts: list[dict]):
    with open(DOCS_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(posts, f, ensure_ascii=False, indent=2, default=str)
    print(f"💾 快取已儲存：{DOCS_CACHE_PATH}")


def load_docs_cache() -> list[dict] | None:
    if not DOCS_CACHE_PATH.exists():
        return None
    with open(DOCS_CACHE_PATH, "r", encoding="utf-8") as f:
        posts = json.load(f)
    print(f"✅ 從快取載入，共 {len(posts)} 筆")
    return posts


def get_posts() -> list[dict]:
    cached = load_docs_cache()
    if cached is not None:
        return cached
    print("🔄 快取不存在，從 Firebase 抓取...")
    posts = fetch_posts_from_firebase()
    save_docs_cache(posts)
    return posts


# ─── Ragas 產生測試集 ─────────────────────────────


def generate_ragas_testset(testset_size: int = 50):
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("缺少 OPENAI_API_KEY，請確認 .env 是否已設定")

    posts = get_posts()
    docs = posts_to_langchain_docs(posts)

    print(f"📄 轉換完成，共 {len(docs)} 份 LangChain Documents")

    openai_client = openai.OpenAI()

    generator_llm = llm_factory("gpt-4o-mini", client=openai_client)

    generator_embeddings = OpenAIEmbeddings(
        client=openai_client,
        model="text-embedding-3-small",
    )

    sa_student_persona = Persona(
        name="SA學生",
        role_description=(
            "正在修習輔仁大學資管系「系統分析與設計」課程的學生。"
            "這位學生會以真實修課學生的角度詢問"
            # "所有問題必須使用繁體中文(專有名詞除外)。"
            "問題要自然、具體、口語化，像學生真的會問的問題。"
            # "避免過度抽象、避免英文問題、避免中英夾雜。"
            # "問題必須能從提供的文件內容中找到根據，不要憑空延伸。"
        ),
    )

    generator = TestsetGenerator(
        llm=generator_llm,
        embedding_model=generator_embeddings,
        persona_list=[sa_student_persona],
    )

    query_distribution = [
        (SingleHopSpecificQuerySynthesizer(llm=generator_llm), 0.4),
        (MultiHopSpecificQuerySynthesizer(llm=generator_llm), 0.6),
    ]

    print(f"⚙️ 開始產生 Ragas testset，共 {testset_size} 筆...")

    dataset = generator.generate_with_langchain_docs(
        docs,
        testset_size=testset_size,
        query_distribution=query_distribution,
    )

    print("✅ Testset 產生完成")
    return dataset


if __name__ == "__main__":
    dataset = generate_ragas_testset(testset_size=50)

    df = dataset.to_pandas()

    output_path = OUTPUT_PATH
    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"✅ 已輸出：{output_path}")
    print(df.head())
