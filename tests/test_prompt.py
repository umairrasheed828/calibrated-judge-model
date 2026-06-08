import json

from src.judge.prompt import (
    SYSTEM_PROMPT,
    build_messages,
    target_completion,
    to_chat_example,
)


def test_system_prompt_contains_both_axes() -> None:
    assert "faithfulness" in SYSTEM_PROMPT
    assert "relevance" in SYSTEM_PROMPT


def test_user_prompt_includes_all_three_inputs() -> None:
    msgs = build_messages("Q?", "CTX", "ANS")
    assert msgs[0]["role"] == "system"
    user = msgs[1]["content"]
    assert "Q?" in user and "CTX" in user and "ANS" in user


def test_target_completion_is_valid_json() -> None:
    assert json.loads(target_completion(4, 5)) == {"faithfulness": 4, "relevance": 5}


def test_chat_example_has_three_roles_ending_in_assistant() -> None:
    ex = to_chat_example("Q?", "CTX", "ANS", 3, 2)
    assert [m["role"] for m in ex["messages"]] == ["system", "user", "assistant"]
    assert json.loads(ex["messages"][-1]["content"]) == {
        "faithfulness": 3,
        "relevance": 2,
    }
