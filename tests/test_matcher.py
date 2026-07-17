import numpy as np

from app.matcher import Matcher


def _unit(seed: int) -> np.ndarray:
    v = np.random.default_rng(seed).standard_normal(512).astype(np.float32)
    return v / np.linalg.norm(v)


def test_empty_matcher_returns_none():
    m = Matcher(threshold=0.4)
    assert m.search(_unit(1)) is None
    assert len(m) == 0


def test_exact_match_returns_user_and_score():
    m = Matcher(threshold=0.4)
    emb = _unit(1)
    m.add(face_id=10, user_id=7, embedding=emb)
    result = m.search(emb)
    assert result is not None
    user_id, score = result
    assert user_id == 7
    assert score > 0.99


def test_below_threshold_returns_none():
    m = Matcher(threshold=0.4)
    m.add(10, 7, _unit(1))
    # a different random vector is nearly orthogonal -> score ~ 0
    assert m.search(_unit(2)) is None


def test_picks_best_of_multiple_users():
    m = Matcher(threshold=0.4)
    a, b = _unit(1), _unit(2)
    m.add(10, 7, a)
    m.add(11, 8, b)
    user_id, _ = m.search(b)
    assert user_id == 8


def test_remove_user_removes_all_their_faces():
    m = Matcher(threshold=0.4)
    emb = _unit(1)
    m.add(10, 7, emb)
    m.add(11, 7, _unit(3))
    m.remove_user(7)
    assert len(m) == 0
    assert m.search(emb) is None


def test_unnormalized_input_is_normalized():
    m = Matcher(threshold=0.4)
    emb = _unit(1)
    m.add(10, 7, emb * 5.0)
    _, score = m.search(emb * 0.1)
    assert score > 0.99
