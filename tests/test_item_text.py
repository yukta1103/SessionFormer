import pandas as pd

from sessionformer.content.item_text import build_item_text


def test_categoryid_becomes_a_single_token():
    df = pd.DataFrame(
        [
            {"timestamp": 100, "itemid": 1, "property": "categoryid", "value": "42"},
        ]
    )
    text = build_item_text(df)
    assert text[1] == "categoryid_42"


def test_multi_value_property_splits_into_multiple_tokens():
    df = pd.DataFrame(
        [
            {"timestamp": 100, "itemid": 1, "property": "888", "value": "111 222 n5.000"},
        ]
    )
    text = build_item_text(df)
    tokens = set(text[1].split())
    assert tokens == {"prop_888_111", "prop_888_222", "prop_888_n5.000"}


def test_only_most_recent_value_is_kept_per_property():
    df = pd.DataFrame(
        [
            {"timestamp": 100, "itemid": 1, "property": "888", "value": "old_value"},
            {"timestamp": 200, "itemid": 1, "property": "888", "value": "new_value"},
        ]
    )
    text = build_item_text(df)
    assert text[1] == "prop_888_new_value"


def test_items_are_independent():
    df = pd.DataFrame(
        [
            {"timestamp": 100, "itemid": 1, "property": "categoryid", "value": "1"},
            {"timestamp": 100, "itemid": 2, "property": "categoryid", "value": "2"},
        ]
    )
    text = build_item_text(df)
    assert text[1] == "categoryid_1"
    assert text[2] == "categoryid_2"
