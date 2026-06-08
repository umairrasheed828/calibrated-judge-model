def test_teacher_module_imports_without_key() -> None:
    # Importing must not touch the network or require a key (lazy openai import).
    from src.teacher import get_completer

    assert callable(get_completer)
