"""KoeAxis CI: queue, run and review subtitle pipeline runs through a Web UI and an open HTTP API.

Layout (light modules first; only `phase` imports the heavy ASR stack):

- `pipeline`  stage catalogue, presets and run-argument building (shared by server and worker)
- `store`     SQLAlchemy-backed run/user/token store (any SQLAlchemy URL; SQLite by default)
- `blobs`     artifact storage interface + local-directory implementation
- `auth`      password hashing, sessions, API/worker tokens
- `api`       FastAPI application (`create_app`) serving `/api/v1/*` and the Web UI
- `worker`    worker loop: claims runs over HTTP, executes phases, reports progress and artifacts
- `phase`     subprocess entry that runs one pipeline phase in a workspace (GPU or API)
"""
