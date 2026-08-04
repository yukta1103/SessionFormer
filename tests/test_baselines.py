from sessionformer.baselines.item_knn import ItemKNNBaseline
from sessionformer.baselines.popularity import PopularityBaseline


def test_popularity_ranks_by_frequency():
    items = [5, 5, 5, 3, 3, 7]
    model = PopularityBaseline().fit(items)
    assert model.recommend([], k=3) == [5, 3, 7]


def test_popularity_excludes_session_items():
    items = [5, 5, 5, 3, 3, 7]
    model = PopularityBaseline().fit(items)
    assert model.recommend([5], k=2) == [3, 7]


def test_popularity_recommend_before_fit_raises():
    model = PopularityBaseline()
    try:
        model.recommend([], k=3)
        assert False, "expected RuntimeError"
    except RuntimeError:
        pass


def test_item_knn_recommends_co_occurring_items_ranked_by_strength():
    # items 2 and 3 co-occur in two sessions; 2 and 4 co-occur in only one
    sessions = [[2, 3], [2, 3], [2, 4]]
    model = ItemKNNBaseline().fit(sessions, vocab_size=6)
    recs = model.recommend([2], k=2)
    assert recs[0] == 3
    assert 4 in recs


def test_item_knn_excludes_session_items():
    sessions = [[2, 3], [2, 3]]
    model = ItemKNNBaseline().fit(sessions, vocab_size=6)
    recs = model.recommend([2, 3], k=5)
    assert recs == []


def test_item_knn_empty_session_returns_no_recommendations():
    sessions = [[2, 3], [2, 3]]
    model = ItemKNNBaseline().fit(sessions, vocab_size=6)
    assert model.recommend([], k=5) == []
