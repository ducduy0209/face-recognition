from tests.conftest import upload


def _enroll(client, seed="an", name="An", email="a@x.com", phone="0901"):
    r = client.post("/users", data={"name": name, "email": email, "phone": phone},
                    files=upload(f"face:{seed}".encode()))
    assert r.status_code == 201
    return r.json()["user_id"]


def test_recognize_enrolled_user(client):
    uid = _enroll(client)
    r = client.post("/recognize", files=upload(b"face:an"))
    assert r.status_code == 200
    body = r.json()
    assert body["matched"] is True
    assert body["confidence"] > 0.99
    assert body["user"] == {"id": uid, "name": "An", "email": "a@x.com", "phone": "0901"}


def test_recognize_stranger_below_threshold(client):
    _enroll(client)
    r = client.post("/recognize", files=upload(b"face:stranger"))
    assert r.status_code == 200
    assert r.json() == {"matched": False, "reason": "below_threshold"}


def test_recognize_empty_database(client):
    r = client.post("/recognize", files=upload(b"face:an"))
    assert r.status_code == 200
    assert r.json() == {"matched": False, "reason": "below_threshold"}


def test_recognize_no_face_returns_200(client):
    r = client.post("/recognize", files=upload(b"noface"))
    assert r.status_code == 200
    assert r.json() == {"matched": False, "reason": "no_face"}


def test_recognize_uses_largest_face(client):
    uid = _enroll(client, seed="an")
    _enroll(client, seed="binh", name="Binh", email="b@x.com", phone="0902")
    # frame has 2 faces, "an" is the largest (first in the list)
    r = client.post("/recognize", files=upload(b"faces:an,binh"))
    assert r.json()["matched"] is True
    assert r.json()["user"]["id"] == uid


def test_recognize_garbage_returns_422(client):
    r = client.post("/recognize", files=upload(b"garbage"))
    assert r.status_code == 422
    assert r.json()["reason"] == "invalid_image"


def test_recognize_requires_auth(client):
    client.headers.pop("X-API-Key")
    r = client.post("/recognize", files=upload(b"face:an"))
    assert r.status_code == 401
