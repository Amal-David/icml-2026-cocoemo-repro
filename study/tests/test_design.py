import pytest

from study.design import build_retained_assignments, condition_for, validate_balance


def test_retained_design_is_exactly_balanced() -> None:
    assignments = build_retained_assignments()

    assert len(assignments) == 30 * 24
    validate_balance(assignments)


def test_group_rotation_assigns_each_condition_once_per_item() -> None:
    for item_index in range(24):
        observed = {condition_for(group=group, item_index=item_index) for group in range(3)}
        assert observed == {"alpha0", "instruction", "cocoemo"}


def test_invalid_group_is_rejected() -> None:
    with pytest.raises(ValueError, match="group"):
        condition_for(group=3, item_index=0)
