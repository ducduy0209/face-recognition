# SDD Progress Ledger — Face Recognition API

Plan: docs/superpowers/plans/2026-07-17-face-recognition-api.md
Mode: no git (user chose) — commit steps skipped; reviews read files directly.

## Completed tasks

- Task 1: complete (scaffold + config; 1 fix round for brief deviations + controller verbatim restore; review clean, 2/2 tests pass)

- Task 2: complete (app/db.py + tests verbatim, review clean, 8/8 tests pass)
- Task 3: complete (matcher verbatim, review approved)
- Task 4-7: complete (inline by controller per user request — engine, auth, app factory, 6 endpoints; 34/34 fast tests pass)
- Task 8: complete (integration test real model PASS 1/1, 85s; fixed ml_dtypes 0.4.1→0.5.4 incompat with onnx 1.19, pinned ml_dtypes>=0.5 in requirements.txt)
- Task 9: complete (Docker build OK after 2 fixes: libgl1+libglib2.0-0 for the full opencv that insightface pulls in; container smoke test clean: health 200, empty image 422, no-auth 401)
- Extra fix during smoke testing: face_engine.extract guards against empty bytes (cv2.imdecode asserts on an empty buffer → 500), added an assert to the integration test
- Final whole-codebase review: READY WITH NOTES, no blockers. Applied 3 non-blocking fixes: (1) recognize returns matched:false if the user is deleted between search and get_user; (2) update_faces checks the user inside write_lock (avoids TOCTOU with delete); (3) db.add_face guards that the embedding has exactly 512 elements. 34/34 fast tests pass after the fixes. Deferred: zero-vector guard in Matcher (the real engine always emits unit-norm), enroll atomicity (user row with 0 faces if it fails midway — recoverable), upload size limit (trusted caller, YAGNI).

- Post-project: A temporary test UI was created (static/index.html + GET /ui + COPY static + tests/test_ui.py); the user finished testing and requested removal — fully removed, codebase back to the pure-API state. 34/34 fast tests after removal.

## Minor findings (for final review triage)

- [Task 2, plan-mandated] add_face does not validate that the embedding has exactly 512 elements before writing the BLOB — a wrong size would silently corrupt load_all_faces.
- [Task 2, plan-mandated] connect() enables PRAGMA foreign_keys twice around executescript — necessary but fragile.
- [Task 3, plan-mandated] Matcher._normalize does not guard against a zero vector (division by 0 → nan); no 512-dimension check on add/search.
- [Test output] 1 StarletteDeprecationWarning (httpx/testclient) + FutureWarning from insightface face_align — noise from external libraries, not project code.
