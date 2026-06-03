import asyncio
import sys

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import os
import json
from pathlib import Path
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import firebase_admin
from firebase_admin import credentials, firestore
from google.cloud.firestore_v1.base_query import FieldFilter

from dotenv import load_dotenv
from langchain_core.documents import Document

from ragas.embeddings import OpenAIEmbeddings
from ragas.testset import TestsetGenerator
from ragas.testset.persona import Persona
from ragas.testset.synthesizers.single_hop.specific import (
    SingleHopSpecificQuerySynthesizer,
)
from ragas.testset.synthesizers.multi_hop.specific import (
    MultiHopSpecificQuerySynthesizer,
)
from ragas.llms import llm_factory

import openai

# ─── 基本設定 ─────────────────────────────

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


# ─── 取得今天時間範圍 ─────────────────────────────


def get_today_range_utc():
    """
    以 Asia/Taipei 為今天範圍，轉成 UTC 給 Firestore 查詢。
    例如：
    台灣時間 2026-05-15 00:00:00 ~ 2026-05-16 00:00:00
    會轉成 UTC 後查詢。
    """
    now_taipei = datetime.now(TAIPEI_TZ)

    start_taipei = now_taipei.replace(
        hour=0,
        minute=0,
        second=0,
        microsecond=0,
    )
    end_taipei = start_taipei + timedelta(days=1)

    start_utc = start_taipei.astimezone(timezone.utc)
    end_utc = end_taipei.astimezone(timezone.utc)

    return start_utc, end_utc


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
                "created_at": data.get("created_at") or data.get("createdAt"),
            }
        )

    return comments


# ─── 從 Firebase 讀今天的 posts ─────────────────────────────


def fetch_posts_from_firebase() -> list[dict]:
    db = init_firebase()

    start_utc, end_utc = get_today_range_utc()

    print("📥 讀取 Firebase /posts ...")
    print(f"🗓️ 只讀取今天上傳或建立的文件：{TODAY_STR}")
    print(f"UTC 查詢範圍：{start_utc} ~ {end_utc}")

    posts = []
    seen_ids = set()

    """
    這裡會依序嘗試常見時間欄位。
    如果你的 Firestore posts 實際欄位只有 createdAt，
    可以只保留 ["createdAt"]。
    """
    timestamp_fields = [
        "createdAt",
        "created_at",
        "uploadedAt",
        "uploaded_at",
    ]

    for field in timestamp_fields:
        print(f"🔎 嘗試使用 `{field}` 查詢今天的文件...")

        try:
            query = (
                db.collection("posts")
                # .where(filter=FieldFilter(field, ">=", start_utc))
                # .where(filter=FieldFilter(field, "<", end_utc))
            )

            for doc in query.stream():
                if doc.id in seen_ids:
                    continue

                data = doc.to_dict() or {}

                title = data.get("title", "")
                content = data.get("content", "")

                if not content:
                    print(f"⚠️ 跳過 {doc.id}（content 空）")
                    continue

                raw_comments = fetch_comments(db, doc.id)
                comment_str = format_comments_for_rag(raw_comments)

                created_time = (
                    data.get("createdAt")
                    or data.get("created_at")
                    or data.get("uploadedAt")
                    or data.get("uploaded_at")
                    or ""
                )

                posts.append(
                    {
                        "id": doc.id,
                        "title": title,
                        "content": content,
                        "comment": comment_str,
                        "created_at": str(created_time),
                    }
                )

                seen_ids.add(doc.id)

        except Exception as e:
            print(f"⚠️ 使用欄位 `{field}` 查詢失敗：{e}")

    print(f"✅ 今天共讀取 {len(posts)} 筆文件")

    if not posts:
        print("⚠️ 今天沒有讀到文件")
        print("請確認 Firestore /posts 是否有以下其中一個時間欄位：")
        print("createdAt / created_at / uploadedAt / uploaded_at")
        print("也請確認該欄位型別是 Firestore Timestamp，而不是純字串")

    return posts


# ─── posts → LangChain Documents ─────────────────────────────


def posts_to_langchain_docs(posts: list[dict]) -> list[Document]:
    docs = []

    for post in posts:
        title = post.get("title", "")
        content = post.get("content", "")
        comment = post.get("comment")

        page_content = f"標題：{title}\n\n內文：{content}"
        print(f"📄 {page_content}")

        if comment:
            page_content += f"\n\n{comment}"

        docs.append(
            Document(
                page_content=page_content,
                metadata={
                    "post_id": post["id"],
                    "title": title,
                    "source": f"firebase/posts/{post['id']}",
                    "created_at": post.get("created_at", ""),
                },
            )
        )

    return docs


# ─── 快取處理 ─────────────────────────────


def save_docs_cache(posts: list[dict]):
    with open(DOCS_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(posts, f, ensure_ascii=False, indent=2, default=str)

    print(f"💾 今天文件快取已儲存：{DOCS_CACHE_PATH}")


def load_docs_cache() -> list[dict] | None:
    if not DOCS_CACHE_PATH.exists():
        return None

    with open(DOCS_CACHE_PATH, "r", encoding="utf-8") as f:
        posts = json.load(f)

    print(f"✅ 從今天快取載入，共 {len(posts)} 筆：{DOCS_CACHE_PATH}")
    return posts


def get_posts(use_cache: bool = True) -> list[dict]:
    if use_cache:
        cached = load_docs_cache()
        if cached is not None:
            return cached

    print("🔄 快取不存在或不使用快取，從 Firebase 抓取今天文件...")
    posts = fetch_posts_from_firebase()
    save_docs_cache(posts)

    return posts


# ─── Ragas 產生測試集 ─────────────────────────────


def generate_ragas_testset(
    testset_size: int = 10,
    use_cache: bool = True,
):
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("缺少 OPENAI_API_KEY，請確認 .env 是否已設定")

    posts = get_posts(use_cache=use_cache)
    docs = posts_to_langchain_docs(posts)

    print(f"📄 轉換完成，共 {len(docs)} 份 LangChain Documents")

    if not docs:
        raise RuntimeError("今天沒有可用文件，無法產生 Ragas testset")

    openai_client = openai.OpenAI()

    generator_llm = llm_factory(
        "gpt-4o-mini",
        client=openai_client,
    )

    generator_embeddings = OpenAIEmbeddings(
        client=openai_client,
        model="text-embedding-3-small",
    )

    sa_student_persona = Persona(
        name="SA學生",
        role_description=(
            "正在修習輔仁大學資管系「系統分析與設計」課程的學生。"
            "這位學生會以真實修課學生的角度詢問"
            "問題要自然、具體、口語化，像學生真的會問的問題。"
            "所有問題必須使用繁體中文(專有名詞除外)。"
            "問題要自然、具體、口語化，像學生真的會問的問題。"
            "避免過度抽象、避免英文問題、避免中英夾雜。"
            "問題必須能從提供的文件內容中找到根據，不要憑空延伸。"
        ),
    )

    generator = TestsetGenerator(
        llm=generator_llm,
        embedding_model=generator_embeddings,
        # persona_list=[sa_student_persona],
    )

    query_distribution = [
        (SingleHopSpecificQuerySynthesizer(llm=generator_llm), 1.0),
    ]

    # ── 方法一：濾掉 HeadlineSplitter ──────────────────────────
    from ragas.testset.transforms import default_transforms
    from ragas.testset.transforms.splitters import HeadlineSplitter

    transforms = [
        t
        for t in default_transforms(
            documents=docs,
            llm=generator_llm,
            embedding_model=generator_embeddings,
        )
        if not isinstance(t, HeadlineSplitter)
    ]
    print("🔧 已移除 HeadlineSplitter，避免短文件炸掉")
    # ────────────────────────────────────────────────────────────

    print(f"⚙️ 開始產生 Ragas testset，共 {testset_size} 筆...")
    print("👤 Persona：SA學生")
    print("🧪 Query distribution：SingleHop 100%")

    dataset = generator.generate_with_langchain_docs(
        docs,
        testset_size=testset_size,
        query_distribution=query_distribution,
        # transforms=[],  # ← 傳入自訂的
    )

    print("✅ Testset 產生完成")
    return dataset


# ─── 主程式 ─────────────────────────────

if __name__ == "__main__":
    dataset = generate_ragas_testset(
        testset_size=4,
        use_cache=True,
    )

    df = dataset.to_pandas()

    df.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    print(f"✅ 已輸出：{OUTPUT_PATH}")
    print(df.head())
