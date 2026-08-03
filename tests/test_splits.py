import pandas as pd

from sessionformer.data.splits import assign_split, identify_cold_start_items, identify_cold_start_users


def test_assign_split_boundaries():
    session_start_ms = pd.Series(
        {
            "s_train": pd.Timestamp("2015-09-04 23:59:59").value // 10**6,
            "s_val_start": pd.Timestamp("2015-09-05 00:00:00").value // 10**6,
            "s_val_end": pd.Timestamp("2015-09-11 23:59:59").value // 10**6,
            "s_test_start": pd.Timestamp("2015-09-12 00:00:00").value // 10**6,
            "s_test_end": pd.Timestamp("2015-09-18 23:59:59").value // 10**6,
            "s_after_test": pd.Timestamp("2015-09-19 00:00:00").value // 10**6,
        }
    )
    result = assign_split(
        session_start_ms,
        val_start_date="2015-09-05",
        test_start_date="2015-09-12",
        test_end_date="2015-09-18",
    )
    assert result["s_train"] == "train"
    assert result["s_val_start"] == "val"
    assert result["s_val_end"] == "val"
    assert result["s_test_start"] == "test"
    assert result["s_test_end"] == "test"
    assert result["s_after_test"] == "train"  # outside the configured window falls back to train


def test_identify_cold_start_items_threshold():
    events = pd.DataFrame({"itemid": [1, 1, 1, 1, 1, 2, 2], "visitorid": [1, 2, 3, 4, 5, 1, 2]})
    cold = identify_cold_start_items(events, min_interactions=5)
    assert cold == {2}


def test_identify_cold_start_users_threshold():
    events = pd.DataFrame({"itemid": [1, 2, 3, 4, 5, 1, 2], "visitorid": [1, 1, 1, 1, 1, 2, 2]})
    cold = identify_cold_start_users(events, min_interactions=5)
    assert cold == {2}
