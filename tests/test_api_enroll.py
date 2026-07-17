from tests.conftest import upload


def test_missing_api_key_returns_401(client):
    client.headers.pop("X-API-Key")
    r = client.post("/users", data={"name": "An", "email": "a@x.com", "phone": "1"},
                    files=upload(b"face:an"))
    assert r.status_code == 401


def test_wrong_api_key_returns_401(client):
    client.headers["X-API-Key"] = "wrong"
    r = client.post("/users", data={"name": "x", "email": "y", "phone": "z"},
                    files=upload(b"face:x"))
    assert r.status_code == 401


def test_health_needs_no_auth(client):
    client.headers.pop("X-API-Key")
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_enroll_success(client):
    r = client.post("/users", data={"name": "An", "email": "a@x.com", "phone": "0901"},
                    files=upload(b"face:an"))
    assert r.status_code == 201
    body = r.json()
    assert body["user_id"] == 1
    assert body["name"] == "An"


def test_enroll_duplicate_trio_409(client):
    data = {"name": "An", "email": "a@x.com", "phone": "0901"}
    client.post("/users", data=data, files=upload(b"face:an"))
    r = client.post("/users", data=data, files=upload(b"face:an"))
    assert r.status_code == 409
    assert r.json()["reason"] == "duplicate_user"


def test_enroll_rejects_bad_images(client):
    data = {"name": "An", "email": "a@x.com", "phone": "0901"}
    cases = [
        (b"garbage", "invalid_image"),
        (b"noface", "no_face"),
        (b"faces:an,binh", "multiple_faces"),
        (b"face:an:small", "face_too_small"),
    ]
    for content, reason in cases:
        r = client.post("/users", data=data, files=upload(content))
        assert r.status_code == 422, content
        assert r.json()["reason"] == reason
