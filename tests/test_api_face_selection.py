"""Which uploaded photos are accepted for enrollment, beyond the basic checks."""
from tests.conftest import upload


def _upsert(client, content: bytes, phone="0901"):
    return client.put(
        f"/users/by-phone/{phone}",
        data={"name": "An", "email": "a@x.com"},
        files=upload(content),
    )


def _matched_name(client, content: bytes):
    body = client.post("/recognize", files=upload(content)).json()
    return body["user"]["name"] if body["matched"] else None


# --- A selfie taken too close (forehead or chin cut off) ---

def test_upsert_accepts_a_selfie_taken_too_close(client):
    r = _upsert(client, b"tight:an")
    assert r.status_code == 201, r.text


def test_post_users_accepts_a_selfie_taken_too_close(client):
    r = client.post("/users", data={"name": "An", "email": "a@x.com", "phone": "0901"},
                    files=upload(b"tight:an"))
    assert r.status_code == 201, r.text


def test_replacing_faces_accepts_a_selfie_taken_too_close(client):
    uid = _upsert(client, b"face:an").json()["user_id"]
    r = client.post(f"/users/{uid}/faces", files=upload(b"tight:an_new"))
    assert r.status_code == 200, r.text


def test_recognize_does_not_retry_padded(client):
    """Check-in sends a frame every second, empty camera included; a second
    detection pass on every faceless frame would double the CPU per frame."""
    _upsert(client, b"face:an")
    body = client.post("/recognize", files=upload(b"tight:an")).json()
    assert body == {"matched": False, "reason": "no_face"}


# --- Strangers in the background ---

def test_upsert_accepts_one_face_clearly_larger_than_the_rest(client):
    r = _upsert(client, b"faces:an@361,stranger@75,other@62")
    assert r.status_code == 201, r.text


def test_the_enrolled_face_is_the_largest_one(client):
    _upsert(client, b"faces:an@361,stranger@75")
    assert _matched_name(client, b"face:an") == "An"
    assert _matched_name(client, b"face:stranger") is None


def test_largest_face_exactly_twice_the_next_is_accepted(client):
    assert _upsert(client, b"faces:an@200,binh@100").status_code == 201


def test_two_faces_of_similar_size_are_still_rejected(client):
    """Two friends side by side: no way to tell which one is the member."""
    r = _upsert(client, b"faces:an@200,binh@120")
    assert r.status_code == 422
    assert r.json()["reason"] == "multiple_faces"


def test_dominant_face_must_still_be_large_enough(client):
    r = _upsert(client, b"faces:an@60,stranger@20")
    assert r.status_code == 422
    assert r.json()["reason"] == "face_too_small"
