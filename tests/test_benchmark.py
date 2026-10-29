import csv
import io
import sys
from types import SimpleNamespace
from unittest.mock import Mock

from app import benchmark
from app.benchmark import Measurement, measure_one, write_measurement


def test_measure_one_records_elapsed_time_tokens_and_answer():
    response = SimpleNamespace(
        usage_metadata=SimpleNamespace(prompt_token_count=12, candidates_token_count=34),
        text="測定結果",
    )
    client = Mock()
    client.models.generate_content.return_value = response

    result = measure_one(client, "test-model", "質問", 0.2, 1)

    client.models.generate_content.assert_called_once()
    assert result.model == "test-model"
    assert result.question == "質問"
    assert result.run == 1
    assert result.elapsed_seconds >= 0
    assert result.input_tokens == 12
    assert result.output_tokens == 34
    assert result.answer == "測定結果"


def test_write_measurement_outputs_csv_with_missing_tokens_as_empty_fields():
    output = io.StringIO()
    writer = csv.writer(output, lineterminator="\n")

    write_measurement(
        writer,
        Measurement(
            model="test-model",
            question="質問, です",
            run=1,
            elapsed_seconds=1.23456,
            input_tokens=None,
            output_tokens=None,
            answer="回答\nです",
        ),
    )

    assert next(csv.reader(io.StringIO(output.getvalue()))) == [
        "test-model",
        "質問, です",
        "1",
        "1.235",
        "",
        "",
        "回答\nです",
    ]


def test_main_runs_each_model_question_pair_and_outputs_csv(monkeypatch, capsys):
    client = Mock()
    client.models.generate_content.return_value = SimpleNamespace(
        usage_metadata=SimpleNamespace(prompt_token_count=1, candidates_token_count=2),
        text="回答",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark",
            "--model",
            "model-a",
            "--model",
            "model-b",
            "--question",
            "質問1",
            "--question",
            "質問2",
        ],
    )
    monkeypatch.setattr(benchmark.genai, "Client", Mock(return_value=client))
    monkeypatch.setattr(benchmark, "get_api_key", lambda: "test-key")

    benchmark.main()

    captured = capsys.readouterr()
    rows = list(csv.reader(io.StringIO(captured.out)))
    assert rows[0] == [
        "model",
        "question",
        "run",
        "elapsed_seconds",
        "input_tokens",
        "output_tokens",
        "answer",
    ]
    assert [(row[0], row[1]) for row in rows[1:]] == [
        ("model-a", "質問1"),
        ("model-b", "質問1"),
        ("model-a", "質問2"),
        ("model-b", "質問2"),
    ]
    assert client.models.generate_content.call_count == 4
    assert "[実行予定] APIリクエスト数: 4" in captured.err
