from tests.conftest import upload


def _upsert(client, phone, seed, name="An", email="a@x.com"):
    return client.put(
        f"/users/by-phone/{phone}",
        data={"name": name, "email": email},
        files=upload(f"face:{seed}".encode()),
    )


def test_upsert_creates_then_updates(client):
    # first call creates the user (201, created=True)
    r = _upsert(client, "0901", "an")
    assert r.status_code == 201
    body = r.json()
    assert body["created"] is True and body["face_count"] == 1
    uid = body["user_id"]
    assert client.post("/recognize", files=upload(b"face:an")).json()["matched"] is True

    # second call to the same phone updates info + replaces the face (200)
    r = _upsert(client, "0901", "an_new", name="An Updated", email="an2@x.com")
    assert r.status_code == 200
    assert r.json()["created"] is False and r.json()["user_id"] == uid

    # old face no longer matches, new one does
    assert client.post("/recognize", files=upload(b"face:an")).json()["matched"] is False
    rec = client.post("/recognize", files=upload(b"face:an_new")).json()
    assert rec["matched"] is True and rec["user"]["name"] == "An Updated"

    # still a single user with a single face
    body = client.get("/users").json()
    assert body["total"] == 1 and body["users"][0]["face_count"] == 1


def test_upsert_rejects_bad_image(client):
    r = client.put(
        "/users/by-phone/0901",
        data={"name": "An", "email": "a@x.com"},
        files=upload(b"noface"),
    )
    assert r.status_code == 422
    assert r.json()["reason"] == "no_face"


def test_patch_updates_info_only(client):
    _upsert(client, "0901", "an")
    r = client.patch("/users/by-phone/0901", data={"name": "New Name", "new_phone": "0999"})
    assert r.status_code == 200
    assert r.json()["name"] == "New Name" and r.json()["phone"] == "0999"

    # the face still matches and now reports the new identity
    rec = client.post("/recognize", files=upload(b"face:an")).json()
    assert rec["matched"] is True
    assert rec["user"]["phone"] == "0999" and rec["user"]["name"] == "New Name"


def test_patch_unknown_phone_404(client):
    r = client.patch("/users/by-phone/0000", data={"name": "X"})
    assert r.status_code == 404
    assert r.json()["reason"] == "user_not_found"


def test_delete_by_phone(client):
    _upsert(client, "0901", "an")
    r = client.delete("/users/by-phone/0901")
    assert r.status_code == 204
    assert client.get("/users").json()["total"] == 0
    assert client.post("/recognize", files=upload(b"face:an")).json()["matched"] is False


def test_delete_by_phone_unknown_404(client):
    r = client.delete("/users/by-phone/0000")
    assert r.status_code == 404
    assert r.json()["reason"] == "user_not_found"


def test_delete_by_id_still_works(client):
    """The int-keyed delete must not be shadowed by the by-phone route."""
    uid = _upsert(client, "0901", "an").json()["user_id"]
    assert client.delete(f"/users/{uid}").status_code == 204
    assert client.get("/users").json()["total"] == 0
