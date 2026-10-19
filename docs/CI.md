# KoeAxis CI：字幕流水线 WebUI / 服务端 / Worker

> 状态：可用的参考实现。**所有部署细节（反代、HTTPS、数据库、对象存储、SSO、进程托管）都设计为可替换**，
> 本文只描述接口契约与默认行为；`deploy/` 下的文件是示例，部署者按实际环境调整即可。

现有本地 WebUI（`webapp.py` / `qwen_asr/web/`）未受影响，CI 服务是独立的包 `qwen_asr/ci/`。

## 1. 架构

```
浏览器 / 脚本 ──HTTP(S)──▶ 控制端 server（无需 GPU）◀──HTTP 轮询── Worker（有 GPU，可在 NAT 后）
                          ├ SQL 数据库（默认 SQLite，任意 SQLAlchemy URL）
                          ├ BlobStore（默认本地 sha256 内容寻址目录）
                          └ 运行日志（默认本地 append-only 文件）
```

- Worker **主动拉取**（`claim`），服务端不需要访问 Worker，Worker 可在家用网络/内网。
- 一个 run = 一次 `main.py run`，分两个 phase 子进程：`gpu`（prepare→transcribe→correct→align→align-recovery）
  和 `api`（split→translate→mimo-proofread→proofread-realign→text-proofread→quality-gate→normalize→export；
  text-proofread 需 Worker 环境有 `GEMINI_API_KEY` 或 `GEMINI_API_KEY_1..9`）。
  Worker 同时最多 `--max-runs` 个 run，但**仅当自己没有 run 处于 GPU 阶段时才会 claim 新任务**，
  因此 GPU 串行、API 阶段可与下一个 run 的 GPU 阶段重叠。
- 重跑（retry）默认复用同一 `workspace_key` 的工作目录，并在一段时间内优先派给持有该工作区的 Worker
  （`worker_affinity_seconds`），配合 `reset_from=<stage>` 只重算该阶段及之后。

## 2. 启动

依赖：`pip install -r requirements-ci.txt`（服务端只需要这些；Worker 还需要项目的 `requirements.txt`）。

```bash
# 服务端
python -m qwen_asr.ci create-user --username admin --role admin      # 交互输入密码，或设 KOEAXIS_PASSWORD
python -m qwen_asr.ci create-token --kind worker --name gpu-1         # 令牌只打印一次
python -m qwen_asr.ci server --host 0.0.0.0 --port 8710

# Worker（GPU 机器，项目根目录）
export KOEAXIS_WORKER_TOKEN=kxw_...
export GEMINI_API_KEY=...          # 文本 LLM（默认 gemini-3.8-flash）；密钥只放在 Worker 环境里，服务端不保存
export GEMINI_BASE_URL=https://relay.example/   # 可选：中转站；不设则用 Google 官方
export MIMO_API_KEY=...            # MiMo 听音校对
python -m qwen_asr.ci worker --server https://ci.example.com --id gpu-1 \
    --data-dir ci-worker --model-cache-dir .model-cache
```

其他命令：`python -m qwen_asr.ci openapi --output openapi.json` 导出 API 规范。

## 3. 配置（服务端，环境变量 `KOEAXIS_CI_<FIELD>`）

| 变量 | 默认 | 说明 |
|---|---|---|
| `DATA_DIR` | `ci-data` | 数据根目录 |
| `DB_URL` | `sqlite:///<data_dir>/ci.sqlite3` | 任意 SQLAlchemy URL（PostgreSQL/MySQL 需自行安装驱动） |
| `BLOB_DIR` / `LOG_DIR` | `<data_dir>/blobs` / `logs` | 本地存储位置 |
| `PRESETS_DIRS` | `ci-presets` | 预设目录，逗号分隔可多个 |
| `MAX_UPLOAD_MB` | 4096 | 单文件上传上限 |
| `SESSION_TTL_HOURS` | 336 | 登录会话有效期 |
| `COOKIE_NAME` / `COOKIE_SECURE` | `koeaxis_session` / false | HTTPS 部署时设 `COOKIE_SECURE=1` |
| `TRUSTED_USER_HEADER` | 空（关闭） | 反代 SSO：如 `X-Forwarded-User`、`Remote-User` |
| `TRUSTED_PROXY_IPS` | `127.0.0.1,::1` | 仅信任来自这些 IP/CIDR（如 `172.16.0.0/12`）的身份头；`*` = 全部（慎用） |
| `TRUSTED_HEADER_AUTO_CREATE` | false | SSO 用户首次访问自动建 member 账号 |
| `MEMBER_EXTRA_ARGS` | false | 允许 member 传原始 CLI 参数（admin 总是允许） |
| `INPUT_FLAGS` | 内置白名单 | 允许用户上传替换的文件参数（`--glossary-xlsx` 等） |
| `WORKER_STALE_SECONDS` | 600 | running 且超过该时间无心跳 → 判失败 |
| `WORKER_AFFINITY_SECONDS` | 300 | 重跑优先派给原 Worker 的时长 |
| `WEBHOOK_URLS` / `WEBHOOK_SECRET` | 空 | 见 §7 |
| `ROOT_PATH` | 空 | 挂在子路径下时（如 `/koeaxis`） |
| `CORS_ORIGINS` | 空 | 跨域前端（单独部署 UI 时） |

列表类变量用逗号（或分号）分隔，布尔用 `1/true/yes/on`。

## 4. 鉴权

| 方式 | 用法 | 适用 |
|---|---|---|
| 会话 cookie | `POST /api/v1/auth/login`；之后写操作须带 `X-Requested-With: koeaxis`（CSRF 防护） | 浏览器 |
| 用户令牌 | `Authorization: Bearer kxu_...`（`/api/v1/me/tokens` 自助创建） | 脚本、CI 集成 |
| Worker 令牌 | `Authorization: Bearer kxw_...` + `X-KoeAxis-Worker: <worker id>` | Worker |
| 反代身份头 | 反代完成登录后注入 `TRUSTED_USER_HEADER`，来源 IP 须在 `TRUSTED_PROXY_IPS` | OAuth2-Proxy / Authelia / Cloudflare Access 等 |

角色：`admin`（全部）/ `member`（只能看/操作自己的 run，不能传原始 CLI 参数）。
令牌只存 sha256；密码用 scrypt。登录有简单的按 IP+用户名限流。

## 5. HTTP API（`/api/v1`，完整定义见 `/api/v1/openapi.json`，交互文档 `/api/docs`）

| 方法 路径 | 说明 |
|---|---|
| `GET health` / `GET meta` | 健康检查 / 阶段列表、预设、允许的输入参数等 |
| `POST auth/login`, `POST auth/logout`, `GET me`, `POST me/password` | 账号 |
| `GET/POST me/tokens`, `DELETE me/tokens/{id}` | 个人 API 令牌 |
| `GET presets` | 预设列表 |
| `POST blobs`（multipart `file`） | 上传素材/输入文件 → `{id, sha256, size, name}`，同内容去重 |
| `GET runs?status=&preset_id=&mine=&q=&limit=&offset=` / `POST runs` | 列表（`{items,total}`）/ 入队 |
| `GET/PATCH/DELETE runs/{id}` | 详情（阶段、指标、checks、队列位置）/ 改标题标签优先级 / 删除 |
| `POST runs/{id}/cancel` | 取消（排队中立即取消；运行中由 Worker 杀进程树） |
| `POST runs/{id}/retry` | 重跑：`reset_from`、`options`、`extra_args`、`fresh_workspace`、`refresh_preset`、`title` |
| `GET runs/{id}/log?offset=N` / `log.txt` | 增量日志（返回 `next_offset`）/ 全文 |
| `GET runs/{id}/artifacts`, `GET runs/{id}/artifacts/{name}` | 成品列表 / 下载 |
| `GET runs/{id}/media` | 下载原始素材 |
| `GET compare?a=&b=` | A/B 对比：阶段耗时、指标、checks、SRT 差异 |
| `admin/users`, `admin/tokens`, `admin/audit`, `POST admin/reap`, `GET workers` | 管理 |

入队示例（`POST /api/v1/runs`）：

```json
{
  "preset_id": "togeradi",
  "media_blob_id": "<blobs 返回的 id>",
  "title": "第30期",
  "options": {"llm_model": "gemini-3.8-flash"},
  "inputs": [{"flag": "--glossary-xlsx", "blob_id": "<id>"}],
  "labels": {"batch": "2026-10"},
  "priority": 0
}
```

run 状态：`queued → running → succeeded | failed | cancelled`。

curl 示例：

```bash
H="Authorization: Bearer $KOEAXIS_TOKEN"
blob=$(curl -s -H "$H" -F file=@ep30.mp3 $S/api/v1/blobs | jq -r .id)
curl -s -H "$H" -H 'Content-Type: application/json' --data-binary @run.json $S/api/v1/runs
```

> Windows Git Bash 下命令行参数中的中文会被转成 GBK：JSON 请求体请写进 UTF-8 文件后用
> `--data-binary @file`，含中文的文件名先改名再上传（浏览器上传不受影响）。

## 6. Worker 协议（任何语言都可实现自己的 Worker）

所有请求带 Worker 令牌与 `X-KoeAxis-Worker`。

1. `POST worker/claim`，体 `{"name"?, "info"?, "preset_ids"?}` → `204`（无任务）或 `{"run": RunSpec}`。
   `RunSpec` 含 `id, preset(快照), options, extra_args, workspace_key, reset_from, env, media{url,sha256,...},
   inputs[{index,flag,name,sha256,url}]`。
2. 下载 `media.url`、各 `inputs[].url`（可按 sha256 缓存）。
3. 执行期间：
   - `POST worker/runs/{id}/log`，原始 UTF-8 文本追加（单块有上限）；
   - `POST worker/runs/{id}/heartbeat`，体 `{stages?, metrics?, checks?, summary?}`，返回 `{cancel_requested, status}`；
     `cancel_requested=true` 时应终止并以 `cancelled` 完成；
   - `GET worker/runs/{id}` 也可查询取消标志。
4. `POST worker/runs/{id}/artifacts`（multipart：`file`, `name`, `kind`），可多次。
5. `POST worker/runs/{id}/complete`，体 `{status: succeeded|failed|cancelled, error?, stages?, metrics?, checks?, summary?}`。

数据形状（UI 依赖的约定，可扩展字段）：

```json
stages:  [{"name":"translate","phase":"api","status":"completed","elapsed_s":82.1,"detail":{...}}]
metrics: {"subtitle_count":619,"gpu_phase_s":0.1,"api_phase_s":116.8,"llm_requests":209,"llm_request_failures":5,"input_tokens":...,"cached_tokens":...,"cache_hit_rate":0.903}
checks:  [{"name":"final_quality","status":"<质量报告原样状态>","message":"..."}]
```

## 7. Webhook

配置 `WEBHOOK_URLS` 后，事件 `run.queued`（入队/重跑）、`run.started`（被 claim）、`run.finished`（成功/失败/取消/失联回收）会 POST
`{"event": ..., "run": {...}}`，头 `X-KoeAxis-Event`；设置 `WEBHOOK_SECRET` 时附
`X-KoeAxis-Signature: sha256=<HMAC-SHA256(secret, body)>`。失败不重试（fire and forget）。

## 8. 预设（`ci-presets/*.json`）

```json
{
  "id": "togeradi", "title": "...", "description": "...",
  "language": "Japanese", "target_language": "简体中文",
  "run_args": ["--with-correct", "--with-align", "..."],
  "files": [{"flag": "--glossary-xlsx", "path": "../samples/翻译对照.xlsx"}],
  "env": {"QWEN_ASR_MFA_BACKGROUND_PREFETCH": "1"},
  "options": {"llm_model": "gemini-3.8-flash"}
}
```

- `files` 路径相对预设文件；入队时服务端把文件内容**快照**进 BlobStore，之后修改预设不影响已有 run
  （重跑时 `refresh_preset=true` 才取新版本）。
- 参数合并顺序（argparse 后者覆盖前者）：预设 `run_args` < Worker `--default-run-args` < run `options` < run `extra_args`。
- `options` 键：`asr_model`、`align_model`、`llm_model`、`target_language`、`language`。
  Worker 用 `--model-cache-dir` 把 `org/name` 形式的模型 id 解析为本地快照路径。

## 9. 扩展点

| 要替换的东西 | 怎么做 |
|---|---|
| 数据库 | `KOEAXIS_CI_DB_URL`（SQLAlchemy Core，表在首次启动时创建） |
| 文件存储（S3/OSS 等） | 实现 `qwen_asr/ci/blobs.py::BlobStore`（put/open/local_path/exists/delete），传给 `create_app(blob_store=...)` |
| 日志存储 | 替换 `LogStore`（append / read(offset)） |
| 鉴权 | 反代身份头，或在 `create_app` 外层加中间件；令牌接口照常可用 |
| 前端 | `qwen_asr/ci/static/` 是无构建步骤的原生 JS，可整体替换；或用 `CORS_ORIGINS` 单独部署 |
| Worker 实现 | 按 §6 协议自写；或 `--phase-command` 替换阶段执行命令（如套一层容器） |
| 阶段事件 | phase 子进程 stdout 中以 `@@KOEAXIS ` 开头的行是 JSON 事件，见下 |
| 嵌入其他 ASGI 应用 | `from qwen_asr.ci.api import create_app; app = create_app(settings)` |

phase 事件（每行一个，其余行视为普通日志）：

```
@@KOEAXIS {"event":"phase","phase":"gpu","status":"running","at":"..."}
@@KOEAXIS {"event":"stage","stage":"align","status":"completed","elapsed_s":12.3,"summary":{...}}
@@KOEAXIS {"event":"reset","stage":"translate","removed":["..."]}
@@KOEAXIS {"event":"gpu_lock","status":"acquired"}
```

## 10. Worker 选项

`--server`、`--token`（或 `KOEAXIS_WORKER_TOKEN`）、`--id`（稳定 id，默认主机名）、`--data-dir`、`--max-runs`(2)、
`--poll-interval`(10s)、`--default-run-args`、`--env KEY=VALUE`（可重复）、`--preset`（只领指定预设）、
`--model-cache-dir`、`--phase-command`、`--no-manifests`、`--once`。

Worker 目录：`runs/<workspace_key>/`（工作区，重跑复用）、`media/`、`inputs/<run>/`、`ci/<run>/spec.json`。
上传成品：`*.srt`、`reports.zip`（报告/诊断/日志/质量报告）、`manifests.zip`（中间清单）。

## 11. 已知限制

- 已在 Windows 本机做过真实端到端（服务端+Worker 同机）；Linux Worker、Docker、多 Worker 并发、HTTPS 反代尚未实测。
- 29 期实测 `reset_from=translate` 重跑时 mimo-proofread 只发了约 6 次请求，疑似复用了已有 checkpoint（是否按内容校验尚未确认）；需要完全重算请用 `fresh_workspace=true`。
- 服务端不提供在线编辑字幕/对齐恢复审阅（仍用本地 WebUI）。
- Webhook 无重试队列；Worker 失联只靠心跳超时回收。
