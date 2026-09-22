import numpy as np
import pytest

from app import db


@pytest.fixture
def conn(tmp_path):
    return db.connect(tmp_path / "test.db")


def _emb(seed: int) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal(512).astype(np.float32)


def test_create_and_get_user(conn):
    uid = db.create_user(conn, "An", "an@x.com", "0901")
    assert db.get_user(conn, uid) == {"id": uid, "name": "An", "email": "an@x.com", "phone": "0901"}


def test_duplicate_trio_raises(conn):
    db.create_user(conn, "An", "an@x.com", "0901")
    with pytest.raises(db.DuplicateUserError):
        db.create_user(conn, "An", "an@x.com", "0901")
    # differing in any single field means no duplicate
    db.create_user(conn, "An", "an@x.com", "0902")


def test_faces_roundtrip(conn):
    uid = db.create_user(conn, "An", "an@x.com", "0901")
    emb = _emb(1)
    db.add_face(conn, uid, emb, "data/images/a.jpg")
    faces = db.load_all_faces(conn)
    assert len(faces) == 1
    face_id, user_id, loaded = faces[0]
    assert user_id == uid
    np.testing.assert_allclose(loaded, emb, rtol=1e-6)
    assert db.count_faces(conn, uid) == 1


def test_list_users_with_face_count(conn):
    uid = db.create_user(conn, "An", "an@x.com", "0901")
    db.add_face(conn, uid, _emb(1), "a.jpg")
    db.add_face(conn, uid, _emb(2), "b.jpg")
    users, total = db.list_users(conn)
    assert total == 1
    assert len(users) == 1
    assert users[0]["face_count"] == 2


def test_delete_user_cascades_and_returns_paths(conn):
    uid = db.create_user(conn, "An", "an@x.com", "0901")
    db.add_face(conn, uid, _emb(1), "a.jpg")
    paths = db.delete_user(conn, uid)
    assert paths == ["a.jpg"]
    assert db.get_user(conn, uid) is None
    assert db.load_all_faces(conn) == []
    assert db.delete_user(conn, 999) is None


def test_delete_faces_returns_old_paths(conn):
    uid = db.create_user(conn, "An", "an@x.com", "0901")
    db.add_face(conn, uid, _emb(1), "a.jpg")
    assert db.delete_faces(conn, uid) == ["a.jpg"]
    assert db.count_faces(conn, uid) == 0
