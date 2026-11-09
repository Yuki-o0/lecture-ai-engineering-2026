"""エージェントの配線テスト。

TestModel は実際のLLMを呼ばない偽モデルで、登録されたツールを全部呼んでから
ダミーの回答を返す。APIキーも利用枠も使わずに、
「ツールが登録され、呼ばれ、結果がモデルに戻るか」を確かめられる。
"""

import app.agent as agent_module
from pydantic_ai import DeferredToolRequests
from pydantic_ai.messages import ToolCallPart, ToolReturnPart
from pydantic_ai.models.test import TestModel

from app.agent import agent
from app import tools

BASE_TOOLS = {"search_glossary", "get_lecture_summary", "make_quiz"}
ALL_TOOLS = BASE_TOOLS | {"save_learning_log"}


def _parts(result, kind):
    return [p for m in result.all_messages() for p in m.parts if isinstance(p, kind)]


def test_base_tools_are_registered_and_called():
    model = TestModel(call_tools=sorted(BASE_TOOLS))
    result = agent.run_sync("LoRAとは？", model=model)

    offered = {t.name for t in model.last_model_request_parameters.function_tools}
    assert ALL_TOOLS <= offered

    called = {p.tool_name for p in _parts(result, ToolCallPart)}
    assert BASE_TOOLS <= called
    assert len(_parts(result, ToolReturnPart)) == len(_parts(result, ToolCallPart))
    assert result.output


def test_glossary_result_reaches_the_model():
    model = TestModel(call_tools=["search_glossary"])
    with agent.override(model=model):
        result = agent.run_sync("RAGとは？")
    returns = _parts(result, ToolReturnPart)
    assert returns and returns[0].tool_name == "search_glossary"


def test_save_learning_log_runs_only_after_cli_approval(monkeypatch, tmp_path):
    log_dir = tmp_path / "learning_log"
    monkeypatch.setattr(tools, "LEARNING_LOG_DIR", log_dir)
    models = iter([TestModel(call_tools=["save_learning_log"]), TestModel(call_tools=[])])
    monkeypatch.setattr(agent_module, "build_model", lambda: next(models))
    first_result = agent.run_sync(
        "今日学んだことを学習記録として保存して",
        model=TestModel(call_tools=["save_learning_log"]),
    )
    assert isinstance(first_result.output, DeferredToolRequests)

    def approve(_):
        assert not log_dir.exists()
        return "y"

    monkeypatch.setattr("builtins.input", approve)

    result = agent_module._run_with_approval(agent, "今日の学習記録を保存して")

    assert isinstance(result.output, str)
    saved_files = list(log_dir.glob("*.md"))
    assert len(saved_files) == 1


def test_save_learning_log_can_be_denied_without_resuming_model(monkeypatch, tmp_path, capsys):
    log_dir = tmp_path / "learning_log"
    monkeypatch.setattr(tools, "LEARNING_LOG_DIR", log_dir)
    models = iter([TestModel(call_tools=["save_learning_log"])])
    monkeypatch.setattr(agent_module, "build_model", lambda: next(models))
    monkeypatch.setattr("builtins.input", lambda _: "n")

    result = agent_module._run_with_approval(agent, "今日の学習記録を保存して")

    assert result is None
    assert not log_dir.exists()
    assert "学習記録の保存をキャンセルしました。" in capsys.readouterr().out
