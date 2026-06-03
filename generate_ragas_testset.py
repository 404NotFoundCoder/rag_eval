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
RUN_DATE_STR = datetime.now(TAIPEI_TZ).strftime("%Y%m%d")

RUN_ROOT = Path("eval_ragas_testset")
DOCS_CACHE_DIR = RUN_ROOT / "docs"
DOCS_CACHE_FILENAME_PREFIX = "docs_cache_"
DOCS_CACHE_NOTE_FILENAME = "docs_cache_used.txt"
DOCS_CACHE_PATH = DOCS_CACHE_DIR / f"{DOCS_CACHE_FILENAME_PREFIX}{TODAY_STR}.json"
OUTPUT_PATH = f"ragas_testset_sa_student_{TODAY_STR}.csv"
TESTSET_CSV_FILENAME = "ragas_testset.csv"


def get_ragas_run_dir(
    run_root: str | Path = RUN_ROOT,
    date_str: str | None = None,
    version: int | str | None = None,
    create: bool = True,
) -> Path:
    date_folder = date_str or datetime.now(TAIPEI_TZ).strftime("%Y%m%d")
    date_dir = Path(run_root) / date_folder

    if version is None:
        existing_versions = []
        if date_dir.exists():
            for child in date_dir.iterdir():
                if child.is_dir() and child.name.startswith("v"):
                    try:
                        existing_versions.append(int(child.name[1:]))
                    except ValueError:
                        pass
        version_num = max(existing_versions, default=0) + 1
        version_folder = f"v{version_num}"
    elif isinstance(version, int):
        version_folder = f"v{version}"
    else:
        version_folder = version if str(version).startswith("v") else f"v{version}"

    run_dir = date_dir / version_folder
    if create:
        print(f"[generate] creating/using run_dir: {run_dir}")
        run_dir.mkdir(parents=True, exist_ok=True)
    else:
        print(f"[generate] resolved run_dir: {run_dir}")
    return run_dir


def normalize_docs_cache_date(date_str: str) -> str:
    date_value = str(date_str).strip()
    if len(date_value) == 8 and date_value.isdigit():
        return f"{date_value[:4]}-{date_value[4:6]}-{date_value[6:]}"
    return date_value


def get_docs_cache_path(
    docs_cache_root: str | Path = DOCS_CACHE_DIR,
    docs_cache_date_str: str | None = None,
) -> Path:
    cache_date = normalize_docs_cache_date(docs_cache_date_str or TODAY_STR)
    return Path(docs_cache_root) / f"{DOCS_CACHE_FILENAME_PREFIX}{cache_date}.json"


def find_latest_docs_cache_path(
    docs_cache_root: str | Path = DOCS_CACHE_DIR,
) -> Path | None:
    cache_root = Path(docs_cache_root)
    if not cache_root.exists():
        return None

    cache_files = sorted(
        cache_root.glob(f"{DOCS_CACHE_FILENAME_PREFIX}*.json"),
        key=lambda path: path.name,
    )
    if not cache_files:
        return None

    return cache_files[-1]


def resolve_docs_cache_path(
    docs_cache_path: str | Path | None = None,
    docs_cache_root: str | Path = DOCS_CACHE_DIR,
    docs_cache_date_str: str | None = None,
    refresh_docs_cache: bool = False,
) -> Path:
    if docs_cache_path is not None:
        resolved_path = Path(docs_cache_path)
        print(f"[generate] using explicit docs_cache_path: {resolved_path}")
        return resolved_path

    if docs_cache_date_str is not None:
        resolved_path = get_docs_cache_path(docs_cache_root, docs_cache_date_str)
        print(f"[generate] using specified docs cache date: {resolved_path}")
        return resolved_path

    if refresh_docs_cache:
        resolved_path = get_docs_cache_path(docs_cache_root, TODAY_STR)
        print(f"[generate] refreshing latest docs cache into: {resolved_path}")
        return resolved_path

    latest_path = find_latest_docs_cache_path(docs_cache_root)
    if latest_path is not None:
        print(f"[generate] using latest docs cache: {latest_path}")
        return latest_path

    resolved_path = get_docs_cache_path(docs_cache_root, TODAY_STR)
    print(f"[generate] no docs cache found; will create: {resolved_path}")
    return resolved_path


def write_docs_cache_note(
    run_dir: str | Path,
    docs_cache_path: str | Path,
    refresh_docs_cache: bool = False,
    docs_cache_date_str: str | None = None,
) -> Path:
    note_path = Path(run_dir) / DOCS_CACHE_NOTE_FILENAME
    cache_path = Path(docs_cache_path)
    note_path.parent.mkdir(parents=True, exist_ok=True)

    note_lines = [
        "Ragas generate docs cache note",
        f"generated_at: {datetime.now(TAIPEI_TZ).isoformat()}",
        f"docs_cache_path: {cache_path}",
        f"docs_cache_exists_at_note_time: {cache_path.exists()}",
        f"docs_cache_date_str: {docs_cache_date_str or 'latest'}",
        f"refresh_docs_cache: {refresh_docs_cache}",
    ]
    note_path.write_text("\n".join(note_lines) + "\n", encoding="utf-8")
    print(f"[generate] writing docs cache note: {note_path}")
    return note_path


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
from datetime import datetime, timezone


def fetch_posts_from_firebase() -> list[dict]:
    db = init_firebase()
    print("📥 讀取 Firebase /posts ...")

    posts = []

    # 只要這個時間之後的文章
    cutoff = datetime(2026, 5, 15, tzinfo=timezone.utc)

    for doc in db.collection("posts").stream():
        data = doc.to_dict() or {}

        created_at = data.get("created_at")

        # 沒有 created_at 就先跳過
        if not created_at:
            print(f"⚠️ 跳過 {doc.id}（created_at 空）")
            continue

        # created_at 是 Firestore Timestamp 時，通常可以直接跟 datetime 比
        if created_at < cutoff:
            print(f"⏭️ 跳過 {doc.id}（created_at 太早：{created_at}）")
            continue

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
                "created_at": created_at,
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


def save_docs_cache(posts: list[dict], docs_cache_path: str | Path = DOCS_CACHE_PATH):
    docs_cache_path = Path(docs_cache_path)
    print(f"[generate] creating/using docs cache directory: {docs_cache_path.parent}")
    docs_cache_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[generate] writing docs cache: {docs_cache_path}")
    with open(docs_cache_path, "w", encoding="utf-8") as f:
        json.dump(posts, f, ensure_ascii=False, indent=2, default=str)
    print(f"💾 快取已儲存：{docs_cache_path}")


def load_docs_cache(docs_cache_path: str | Path = DOCS_CACHE_PATH) -> list[dict] | None:
    docs_cache_path = Path(docs_cache_path)
    print(f"[generate] checking docs cache: {docs_cache_path}")
    if not docs_cache_path.exists():
        print(
            f"[generate] docs cache not found, will fetch from Firebase: {docs_cache_path}"
        )
        return None
    print(f"[generate] reading docs cache: {docs_cache_path}")
    with open(docs_cache_path, "r", encoding="utf-8") as f:
        posts = json.load(f)
    print(f"✅ 從快取載入，共 {len(posts)} 筆")
    return posts


def get_posts(
    docs_cache_path: str | Path = DOCS_CACHE_PATH,
    refresh_docs_cache: bool = False,
) -> list[dict]:
    if refresh_docs_cache:
        print(
            f"[generate] refresh_docs_cache=True; rebuilding cache: {docs_cache_path}"
        )
    else:
        cached = load_docs_cache(docs_cache_path)
        if cached is not None:
            return cached
    print("🔄 快取不存在，從 Firebase 抓取...")
    posts = fetch_posts_from_firebase()
    save_docs_cache(posts, docs_cache_path)
    return posts


# ─── Ragas 產生測試集 ─────────────────────────────


def generate_ragas_testset(
    testset_size: int = 50,
    docs_cache_path: str | Path | None = None,
    docs_cache_root: str | Path = DOCS_CACHE_DIR,
    docs_cache_date_str: str | None = None,
    refresh_docs_cache: bool = False,
):
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("缺少 OPENAI_API_KEY，請確認 .env 是否已設定")

    resolved_docs_cache_path = resolve_docs_cache_path(
        docs_cache_path=docs_cache_path,
        docs_cache_root=docs_cache_root,
        docs_cache_date_str=docs_cache_date_str,
        refresh_docs_cache=refresh_docs_cache,
    )

    print(f"[generate] docs_cache_path: {resolved_docs_cache_path}")
    posts = get_posts(resolved_docs_cache_path, refresh_docs_cache=refresh_docs_cache)
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


def save_ragas_testset(
    dataset,
    output_csv: str | Path,
) -> tuple[object, Path]:
    output_path = Path(output_csv)
    print(f"[generate] creating/using output directory: {output_path.parent}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = dataset.to_pandas()
    print(f"[generate] writing testset CSV: {output_path}")
    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"Saved testset: {output_path}")
    return df, output_path


def generate_and_save_ragas_testset(
    testset_size: int = 50,
    run_root: str | Path = RUN_ROOT,
    date_str: str | None = None,
    version: int | str | None = None,
    output_filename: str = TESTSET_CSV_FILENAME,
    docs_cache_path: str | Path | None = None,
    docs_cache_root: str | Path | None = None,
    docs_cache_date_str: str | None = None,
    refresh_docs_cache: bool = False,
) -> dict:
    resolved_docs_cache_root = (
        Path(docs_cache_root)
        if docs_cache_root is not None
        else Path(run_root) / "docs"
    )
    resolved_docs_cache_path = resolve_docs_cache_path(
        docs_cache_path=docs_cache_path,
        docs_cache_root=resolved_docs_cache_root,
        docs_cache_date_str=docs_cache_date_str,
        refresh_docs_cache=refresh_docs_cache,
    )

    print(f"[generate] run_root: {Path(run_root)}")
    print(
        f"[generate] date_str: {date_str or datetime.now(TAIPEI_TZ).strftime('%Y%m%d')}"
    )
    print(f"[generate] version: {version if version is not None else 'auto next vN'}")
    print(f"[generate] docs_cache_root: {resolved_docs_cache_root}")
    print(f"[generate] docs_cache_path: {resolved_docs_cache_path}")
    run_dir = get_ragas_run_dir(
        run_root=run_root,
        date_str=date_str,
        version=version,
        create=True,
    )
    output_csv = run_dir / output_filename
    print(f"[generate] target testset CSV: {output_csv}")

    dataset = generate_ragas_testset(
        testset_size=testset_size,
        docs_cache_path=resolved_docs_cache_path,
        refresh_docs_cache=refresh_docs_cache,
    )
    docs_cache_note = write_docs_cache_note(
        run_dir=run_dir,
        docs_cache_path=resolved_docs_cache_path,
        refresh_docs_cache=refresh_docs_cache,
        docs_cache_date_str=docs_cache_date_str,
    )
    df, output_path = save_ragas_testset(dataset, output_csv)

    return {
        "dataset": dataset,
        "dataframe": df,
        "run_dir": run_dir,
        "testset_csv": output_path,
        "docs_cache_path": resolved_docs_cache_path,
        "docs_cache_note": docs_cache_note,
    }


if __name__ == "__main__":
    result = generate_and_save_ragas_testset(testset_size=50)
    print(result["dataframe"].head())
