# API Credentials Template

Do not commit real API keys to this repository. Store them in environment variables
or a local untracked `.env` file.

## Gemini (default text LLM: split / correct / translate)

- Model: `gemini-3.8-flash` (default since 2026-10)
- Protocol: native `generateContent`; the base URL must end with `/v1beta`
- Default Base URL: `$GEMINI_BASE_URL` + `/v1beta` when set (e.g. a relay), else
  `https://generativelanguage.googleapis.com/v1beta`
- Environment variables: `GEMINI_API_KEY`, optional `GEMINI_BASE_URL`
- Default concurrency: 4 (`--thread-num`); many relays allow at most ~5 in-flight requests per key
- Key value: `<set this only in your environment>`

```powershell
$env:GEMINI_API_KEY = "<your-gemini-api-key>"
$env:GEMINI_BASE_URL = "https://your-relay.example/"   # optional
```

## MiMo official (audio proofread; optional text LLM)

- Default Base URL: `https://api.xiaomimimo.com/v1` (Token Plan `tp-` keys are routed to the Token Plan host)
- Model: `mimo-v2.6-pro` (proofread default, concurrency 10)
- Environment variable: `MIMO_API_KEY`
- Extra body: `{"thinking":{"type":"disabled"}}`
- Key value: `<set this only in your environment>`
- Use MiMo for text stages again: `--llm-model mimo-v2.6-pro --llm-base-url https://api.xiaomimimo.com/v1 --thread-num 10`

```powershell
$env:MIMO_API_KEY = "<your-mimo-api-key>"
```

## Gemini-compatible provider (OpenAI protocol)

- Base URL: `https://api.qhaigc.net`
- Model: `gemini-3.5-flash`
- Environment variable: `GEMINI_API_KEY`
- Key value: `<set this only in your environment>`

## DeepSeek official

- Base URL: `https://api.deepseek.com`
- Environment variable: `DEEPSEEK_API_KEY`
- Key value: `<set this only in your environment>`

```powershell
$env:DEEPSEEK_API_KEY = "<your-deepseek-api-key>"
```
