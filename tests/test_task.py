from src.judge.task import AXES, FAITHFULNESS, RELEVANCE


def test_axes_have_valid_ranges() -> None:
    for axis in AXES:
        assert axis.min_score < axis.max_score
        assert axis.description.strip()


def test_axes_are_the_two_we_calibrated() -> None:
    assert {a.name for a in AXES} == {"faithfulness", "relevance"}
    assert FAITHFULNESS.max_score == 5
    assert RELEVANCE.max_score == 5
