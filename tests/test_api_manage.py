from tests.conftest import upload


def _enroll(client, seed="an", name="An", email="a@x.com", phone="0901"):
    r = client.post("/users", data={"name": name, "email": email, "phone": phone},
                    files=upload(f"face:{seed}".encode()))
    assert r.status_code == 201
    return r.json()["user_id"]


def test_list_users(client):
    _enroll(client)
    _enroll(client, seed="binh", name="Binh", email="b@x.com", phone="0902")
    r = client.get("/users")
    assert r.status_code == 200
    body = r.json()
    assert body["total"] == 2
    assert body["limit"] == 50 and body["offset"] == 0
    users = body["users"]
    assert len(users) == 2
    assert users[0]["face_count"] == 1
    assert set(users[0]) == {"id", "name", "email", "phone", "created_at", "face_count"}


def test_list_users_search_and_paginate(client):
    _enroll(client, seed="an", name="An", email="a@x.com", phone="0901")
    _enroll(client, seed="binh", name="Binh", email="b@x.com", phone="0902")

    # search matches name/email/phone
    r = client.get("/users", params={"q": "Binh"})
    assert r.json()["total"] == 1
    assert r.json()["users"][0]["phone"] == "0902"

    # pagination returns the page slice but the full total
    r = client.get("/users", params={"limit": 1, "offset": 0})
    body = r.json()
    assert body["total"] == 2 and len(body["users"]) == 1


def test_update_faces_append(client):
    uid = _enroll(client)
    r = client.post(f"/users/{uid}/faces?mode=append", files=upload(b"face:an2"))
    assert r.status_code == 200
    assert r.json() == {"user_id": uid, "face_count": 2}
    # both the old and the new embedding recognize this user
    for content in (b"face:an", b"face:an2"):
        rec = client.post("/recognize", files=upload(content))
        assert rec.json()["matched"] is True
        assert rec.json()["user"]["id"] == uid


def test_update_faces_replace_is_default(client):
    uid = _enroll(client)
    r = client.post(f"/users/{uid}/faces", files=upload(b"face:an_new"))
    assert r.status_code == 200
    assert r.json() == {"user_id": uid, "face_count": 1}
    # the old image is no longer recognized, the new one is
    assert client.post("/recognize", files=upload(b"face:an")).json()["matched"] is False
    assert client.post("/recognize", files=upload(b"face:an_new")).json()["matched"] is True


def test_update_faces_unknown_user_404(client):
    r = client.post("/users/999/faces", files=upload(b"face:x"))
    assert r.status_code == 404
    assert r.json()["reason"] == "user_not_found"


def test_update_faces_validates_image(client):
    uid = _enroll(client)
    r = client.post(f"/users/{uid}/faces", files=upload(b"noface"))
    assert r.status_code == 422
    assert r.json()["reason"] == "no_face"


def test_delete_user(client):
    uid = _enroll(client)
    r = client.delete(f"/users/{uid}")
    assert r.status_code == 204
    assert client.get("/users").json()["total"] == 0
    assert client.post("/recognize", files=upload(b"face:an")).json()["matched"] is False


def test_delete_unknown_user_404(client):
    r = client.delete("/users/999")
    assert r.status_code == 404
