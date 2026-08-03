import pandas as pd


def _date_to_ms(date: str) -> int:
    # Timestamp.value is ns since epoch treating a naive date as UTC, avoiding
    # any local-timezone conversion that Timestamp.timestamp() would apply.
    return pd.Timestamp(date).value // 10**6


def assign_split(
    session_start_ms: pd.Series,
    val_start_date: str,
    test_start_date: str,
    test_end_date: str,
) -> pd.Series:
    """Label each session (indexed by session_id) as train/val/test by its start time.

    train: before val_start_date
    val:   [val_start_date, test_start_date)
    test:  [test_start_date, test_end_date] (inclusive of the full test_end_date day)
    """
    val_start_ms = _date_to_ms(val_start_date)
    test_start_ms = _date_to_ms(test_start_date)
    test_end_ms = _date_to_ms(test_end_date) + 24 * 3600 * 1000

    split = pd.Series("train", index=session_start_ms.index)
    split[(session_start_ms >= val_start_ms) & (session_start_ms < test_start_ms)] = "val"
    split[(session_start_ms >= test_start_ms) & (session_start_ms < test_end_ms)] = "test"
    return split


def identify_cold_start_items(train_events: pd.DataFrame, min_interactions: int = 5) -> set[int]:
    counts = train_events["itemid"].value_counts()
    return set(counts[counts < min_interactions].index)


def identify_cold_start_users(train_events: pd.DataFrame, min_interactions: int = 5) -> set[int]:
    counts = train_events["visitorid"].value_counts()
    return set(counts[counts < min_interactions].index)
