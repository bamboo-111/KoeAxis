"""Ask MiMo to transcribe a short audio span turn by turn, with speakers. Usage: speaker_listen.py MEDIA START END"""
import base64
import subprocess
import sys
import tempfile
from pathlib import Path

from openai import OpenAI

sys.path.insert(0, "E:/project/qwen3-asr")
from qwen_asr.credentials import resolve_mimo_api_key  # noqa: E402
from qwen_asr.defaults import DEFAULT_LLM_BASE_URL  # noqa: E402

media, start, end = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
wav = Path(tempfile.gettempdir()) / "speaker_listen.wav"
subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", str(start), "-to", str(end), "-i", media, "-ac", "1", "-ar", "16000", str(wav)],
               check=True)
prompt = (
    "これは広島のテレビ番組の一部です。インタビュアー（男性）とトゲナシトゲアリのボーカル・理名（リナ、女性）がお好み焼き店で話しています。"
    "音声を一発話ずつ正確に書き起こしてください。話者が変わるたびに必ず改行し、各行を「[開始秒-終了秒] 話者: 発話」の形式にしてください。"
    "秒はこの音声の先頭からの秒数（小数1桁）。話者は「インタビュアー」「リナ」「ナレーション」「その他」から選んでください。"
    "相づち（あ、そうなんですね等）も別の行にしてください。聞き取れない部分は推測せず（不明）と書いてください。書き起こし以外の説明は不要です。"
)
client = OpenAI(base_url=DEFAULT_LLM_BASE_URL, api_key=resolve_mimo_api_key(), timeout=300)
response = client.chat.completions.create(
    model=sys.argv[4] if len(sys.argv) > 4 else "mimo-v2.6-pro",
    temperature=0.1,
    max_tokens=4000,
    extra_body={"thinking": {"type": "disabled"}},
    messages=[{"role": "user", "content": [
        {"type": "text", "text": prompt},
        {"type": "input_audio", "input_audio": {"data": base64.b64encode(wav.read_bytes()).decode("ascii"), "format": "wav"}},
    ]}],
)
print(f"(offset {start}s)")
print(response.choices[0].message.content)
