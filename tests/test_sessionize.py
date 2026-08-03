import pandas as pd

from sessionformer.data.sessionize import sessionize

COLUMNS = ["timestamp", "visitorid", "event", "itemid", "transactionid"]


def _events(rows):
    return pd.DataFrame(rows, columns=COLUMNS)


def test_exact_30_min_gap_stays_in_same_session():
    t0 = 1_000_000_000_000
    gap = 30 * 60 * 1000
    events = _events(
        [
            (t0, 1, "view", 10, None),
            (t0 + gap, 1, "view", 11, None),
        ]
    )
    result = sessionize(events, min_session_length=1)
    assert result["session_id"].nunique() == 1


def test_gap_over_30_min_splits_session():
    t0 = 1_000_000_000_000
    gap = 30 * 60 * 1000 + 1
    events = _events(
        [
            (t0, 1, "view", 10, None),
            (t0 + gap, 1, "view", 11, None),
        ]
    )
    result = sessionize(events, min_session_length=1)
    assert result["session_id"].nunique() == 2


def test_sessions_shorter_than_min_length_are_dropped():
    t0 = 1_000_000_000_000
    events = _events(
        [
            (t0, 1, "view", 10, None),  # lone event -> session of length 1, dropped
            (t0, 2, "view", 20, None),
            (t0 + 1000, 2, "view", 21, None),  # session of length 2, kept
        ]
    )
    result = sessionize(events, min_session_length=2)
    assert set(result["visitorid"]) == {2}
    assert result["session_id"].nunique() == 1


def test_different_visitors_get_different_sessions():
    t0 = 1_000_000_000_000
    events = _events(
        [
            (t0, 1, "view", 10, None),
            (t0 + 1000, 1, "view", 11, None),
            (t0, 2, "view", 20, None),
            (t0 + 1000, 2, "view", 21, None),
        ]
    )
    result = sessionize(events, min_session_length=2)
    assert result["session_id"].nunique() == 2


def test_events_within_session_stay_time_ordered():
    t0 = 1_000_000_000_000
    events = _events(
        [
            (t0 + 2000, 1, "view", 12, None),
            (t0, 1, "view", 10, None),
            (t0 + 1000, 1, "view", 11, None),
        ]
    )
    result = sessionize(events, min_session_length=2)
    assert list(result["itemid"]) == [10, 11, 12]
