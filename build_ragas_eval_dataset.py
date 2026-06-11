import json
import os
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

API_URL = "http://localhost:5000/api/chat"
DEFAULT_INPUT_CSV = "ragas_testset_sa_student_2026-05-22.csv"
DEFAULT_OUTPUT_CSV = "ragas_eval_with_response.csv"
RETRY_FAILED_ONLY = False

ACCESS_TOKENS = [
    os.getenv("ACCESS_TOKEN"),
    os.getenv("ACCESS_TOKEN_1"),
    os.getenv("ACCESS_TOKEN_2"),
]


def is_quota_error(error_text: str, status_code=None) -> bool:
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


def resolve_build_output_path(
    input_csv: str | Path,
    output_csv: str | Path | None = None,
    default_filename: str = DEFAULT_OUTPUT_CSV,
) -> Path:
    if output_csv is not None:
        resolved_output = Path(output_csv)
        print(f"[build] using explicit output_csv: {resolved_output}")
        return resolved_output

    input_path = Path(input_csv)
    resolved_output = input_path.parent / default_filename
    print(f"[build] output_csv not provided; using default path: {resolved_output}")
    return resolved_output


def build_ragas_eval_dataset(
    input_csv: str | Path,
    output_csv: str | Path | None = None,
    retry_failed_only: bool = RETRY_FAILED_ONLY,
    api_url: str = API_URL,
    access_tokens: list[str | None] | None = None,
    timeout: int = 60,
) -> tuple[pd.DataFrame, Path]:
    tokens = [token for token in (access_tokens or ACCESS_TOKENS) if token]
    if not tokens:
        raise RuntimeError("Missing access token. Please set ACCESS_TOKEN in .env.")

    input_path = Path(input_csv)
    print(f"[build] reading input CSV: {input_path}")
    output_path = resolve_build_output_path(input_path, output_csv)
    print(f"[build] creating/using output directory: {output_path.parent}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(
        input_path,
        encoding="utf-8-sig",
        dtype={
            "response": "string",
            "retrieved_contexts": "string",
            "retrieved_metadata": "string",
            "sourceTypeDecision": "string",
            "status": "string",
        },
    )
    print(f"[build] loaded rows: {len(df)}")

    string_columns = [
        "response",
        "retrieved_contexts",
        "retrieved_metadata",
        "sourceTypeDecision",
        "status",
    ]

    for col in string_columns:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].astype("object")
    print(df[string_columns].dtypes)

    required_columns = ["user_input", "reference"]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"CSV missing required column: {col}")

    if retry_failed_only:
        if "status" not in df.columns:
            raise ValueError("CSV missing status column for retry_failed_only=True")

        df_to_run = df[df["status"] == "fail"]
        print(f"Retrying failed rows only: {len(df_to_run)} rows")
    else:
        df_to_run = df
        print(f"Processing all rows: {len(df_to_run)} rows")

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
        sourceType_decision = ""

        for _ in range(len(tokens)):
            access_token = tokens[current_key_index]

            try:
                res = requests.post(
                    api_url,
                    headers={"Content-Type": "application/json"},
                    json={
                        "message": user_input,
                        "accessToken": access_token,
                        "history": [],
                    },
                    timeout=timeout,
                )

                if not res.ok:
                    error_text = res.text

                    if is_quota_error(error_text, res.status_code):
                        print(
                            f"\nToken {current_key_index + 1} hit quota/rate limit; switching token"
                        )
                        current_key_index = (current_key_index + 1) % len(tokens)
                        continue

                    res.raise_for_status()

                data = res.json()

                response = data.get("answer", "")
                references = data.get("references", [])

                retrieved_contexts = [ref.get("content", "") for ref in references]
                retrieved_metadata = [
                    {
                        "source": ref.get("source"),
                        "id": ref.get("id"),
                        "sourceType": ref.get("sourceType"),
                    }
                    for ref in references
                ]
                sourceType_decision = data.get("sourceTypeDecision", "")

                status = "success"
                success = True
                break

            except Exception as e:
                error_text = str(e)

                if is_quota_error(error_text):
                    print(
                        f"\nToken {current_key_index + 1} may have hit quota/rate limit; switching token"
                    )
                    current_key_index = (current_key_index + 1) % len(tokens)
                    continue

                print(f"\nFailed question: {user_input}")
                print(e)
                status = "fail"
                break

        if not success:
            print(f"\nAll available tokens failed for question: {user_input}")

        df.at[idx, "response"] = response
        df.at[idx, "retrieved_contexts"] = json.dumps(
            retrieved_contexts, ensure_ascii=False
        )
        df.at[idx, "retrieved_metadata"] = json.dumps(
            retrieved_metadata, ensure_ascii=False
        )
        df.at[idx, "sourceTypeDecision"] = sourceType_decision
        df.at[idx, "status"] = status

    print(f"[build] writing eval CSV: {output_path}")
    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(f"\nSaved eval dataset: {output_path}")
    return df, output_path


if __name__ == "__main__":
    build_ragas_eval_dataset(
        input_csv=DEFAULT_INPUT_CSV,
        output_csv=DEFAULT_OUTPUT_CSV,
        retry_failed_only=RETRY_FAILED_ONLY,
    )
