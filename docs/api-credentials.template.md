# API Credentials Template

Do not commit real API keys to this repository. Store them in environment variables
or a local untracked `.env` file.

## MiMo official

- Default Base URL: `https://api.xiaomimimo.com/v1`
- Model: `mimo-v2.5`
- Environment variable: `MIMO_API_KEY`
- Extra body: `{"thinking":{"type":"disabled"}}`
- Key value: `<set this only in your environment>`
Example:

```powershell
$env:MIMO_API_KEY = "<your-mimo-api-key>"
```

## Gemini-compatible provider

- Base URL: `https://api.qhaigc.net`
- Model: `gemini-3.5-flash`
- Environment variable: `GEMINI_API_KEY`
- Key value: `<set this only in your environment>`
## DeepSeek official

- Base URL: `https://api.deepseek.com`
- Environment variable: `DEEPSEEK_API_KEY`
- Key value: `<set this only in your environment>`
Example:

```powershell
$env:DEEPSEEK_API_KEY = "<your-deepseek-api-key>"
```
