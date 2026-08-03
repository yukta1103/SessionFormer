import pandas as pd

SESSION_TIMEOUT_SECONDS = 30 * 60


def sessionize(
    events: pd.DataFrame,
    timeout_seconds: int = SESSION_TIMEOUT_SECONDS,
    min_session_length: int = 2,
) -> pd.DataFrame:
    """Assign a session_id to each event and drop sessions shorter than min_session_length.

    A new session starts when the gap since a visitor's previous event exceeds
    timeout_seconds (a gap of exactly timeout_seconds stays in the same session).
    """
    df = events.sort_values(["visitorid", "timestamp"]).reset_index(drop=True)
    timeout_ms = timeout_seconds * 1000

    gap = df.groupby("visitorid")["timestamp"].diff()
    new_session = gap.isna() | (gap > timeout_ms)
    session_seq = new_session.groupby(df["visitorid"]).cumsum()
    df["session_id"] = df.groupby([df["visitorid"], session_seq]).ngroup()

    session_sizes = df.groupby("session_id").size()
    valid_sessions = session_sizes[session_sizes >= min_session_length].index
    df = df[df["session_id"].isin(valid_sessions)].reset_index(drop=True)

    return df
