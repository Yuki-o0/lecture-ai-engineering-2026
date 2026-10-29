"""Measure response time and token usage for Gemini models and questions.

Example:
    uv run python -m app.benchmark \
        --model gemini-3.5-flash-lite --model gemini-3.5-flash \
        --question "AIエンジニアリングとは何ですか？" \
        --question "LoRAとは何ですか？" > results.csv
"""

import argparse
import csv
import sys
import time
from dataclasses import dataclass

from google import genai
from google.genai import errors
from google.genai.types import GenerateContentConfig

from app.config import MODEL, api_error_message, get_api_key


@dataclass(frozen=True)
class Measurement:
    model: str
    question: str
    run: int
    elapsed_seconds: float
    input_tokens: int | None
    output_tokens: int | None
    answer: str


def measure_one(
    client: genai.Client,
    model: str,
    question: str,
    temperature: float,
    run: int,
) -> Measurement:
    started_at = time.perf_counter()
    response = client.models.generate_content(
        model=model,
        contents=question,
        config=GenerateContentConfig(temperature=temperature),
    )
    elapsed_seconds = time.perf_counter() - started_at

    usage = response.usage_metadata
    return Measurement(
        model=model,
        question=question,
        run=run,
        elapsed_seconds=elapsed_seconds,
        input_tokens=usage.prompt_token_count if usage is not None else None,
        output_tokens=usage.candidates_token_count if usage is not None else None,
        answer=response.text or "",
    )


def write_measurement(writer: csv.writer, result: Measurement) -> None:
    writer.writerow(
        (
            result.model,
            result.question,
            result.run,
            f"{result.elapsed_seconds:.3f}",
            result.input_tokens if result.input_tokens is not None else "",
            result.output_tokens if result.output_tokens is not None else "",
            result.answer,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="モデルと質問ごとの応答時間・トークン数をCSVで比較する"
    )
    parser.add_argument(
        "--model",
        action="append",
        help=f"比較するモデル（複数回指定可。省略時: {MODEL}）",
    )
    parser.add_argument(
        "--question",
        action="append",
        required=True,
        help="比較する質問（複数回指定可）",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="各モデル・質問の組み合わせの実行回数（既定値: 1）",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="回答のランダムさ（0.0〜2.0、既定値: 1.0）",
    )
    args = parser.parse_args()

    if args.repeat < 1:
        parser.error("--repeat は1以上を指定してください")
    if not 0.0 <= args.temperature <= 2.0:
        parser.error("--temperature は0.0〜2.0の範囲で指定してください")

    models = args.model or [MODEL]
    request_count = len(models) * len(args.question) * args.repeat
    print(f"[実行予定] APIリクエスト数: {request_count}", file=sys.stderr)

    writer = csv.writer(sys.stdout, lineterminator="\n")
    writer.writerow(
        ("model", "question", "run", "elapsed_seconds", "input_tokens", "output_tokens", "answer")
    )
    sys.stdout.flush()

    completed = 0
    try:
        client = genai.Client(api_key=get_api_key())
        for question in args.question:
            for model in models:
                for run in range(1, args.repeat + 1):
                    result = measure_one(client, model, question, args.temperature, run)
                    write_measurement(writer, result)
                    sys.stdout.flush()
                    completed += 1
    except errors.APIError as exc:
        print(api_error_message(exc.code, exc.message), file=sys.stderr)
        print(
            f"[中断] {completed} / {request_count} 件完了。出力済みのCSV行は利用できます。",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
