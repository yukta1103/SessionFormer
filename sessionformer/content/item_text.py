import pandas as pd


def build_item_text(item_properties: pd.DataFrame) -> dict[int, str]:
    """Builds a pseudo-document per item from its (anonymized) property/value
    pairs, keeping only the most recent value per (itemid, property) since
    values change over time in this dataset. RetailRocket has no human-
    readable text (property names and values are hashed/numeric), so this
    isn't real language — it's a bag of category/property tokens that still
    lets items sharing a category or property values land close together
    once embedded."""
    latest = item_properties.sort_values("timestamp").drop_duplicates(subset=["itemid", "property"], keep="last")

    is_category = latest["property"] == "categoryid"
    category_tokens = latest.loc[is_category, ["itemid"]].copy()
    category_tokens["token"] = "categoryid_" + latest.loc[is_category, "value"].astype(str)

    other = latest.loc[~is_category, ["itemid", "property", "value"]].copy()
    other["value"] = other["value"].astype(str).str.split()
    other = other.explode("value")
    other["token"] = "prop_" + other["property"].astype(str) + "_" + other["value"].astype(str)

    all_tokens = pd.concat([category_tokens[["itemid", "token"]], other[["itemid", "token"]]])
    grouped = all_tokens.groupby("itemid")["token"].apply(lambda tokens: " ".join(tokens))
    return grouped.to_dict()


def build_item_categories(item_properties: pd.DataFrame) -> dict[int, str]:
    """Most recent categoryid per item only -- a much lighter artifact than
    the full pseudo-document text, for display purposes (e.g. a demo UI)
    where loading hundreds of MB of property text per item would be
    impractical."""
    category_rows = item_properties[item_properties["property"] == "categoryid"]
    latest = category_rows.sort_values("timestamp").drop_duplicates(subset=["itemid"], keep="last")
    return dict(zip(latest["itemid"], latest["value"].astype(str)))
