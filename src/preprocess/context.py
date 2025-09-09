
# --- src/preprocess/context.py ---
"""
Context/transcript utilities:
- transcribe_with_whisper: use OpenAI whisper if installed
- fallback_transcribe: placeholder using other libs
"""

from pathlib import Path


def transcribe_with_whisper(video_path: str, out_txt: str, model_name: str = 'small'):
    """Transcribe audio using whisper (if available). Saves transcript to out_txt."""
    try:
        import whisper
    except Exception:
        raise RuntimeError("Whisper not installed. Install via `pip install -U openai-whisper` or use another transcription method.")

    model = whisper.load_model(model_name)
    result = model.transcribe(video_path)
    text = result.get('text', '').strip()
    Path(out_txt).parent.mkdir(parents=True, exist_ok=True)
    with open(out_txt, 'w', encoding='utf-8') as f:
        f.write(text)
    return out_txt


def simple_transcribe_placeholder(video_path: str, out_txt: str):
    """A placeholder transcription function. Use Whisper or other ASR for real runs."""
    Path(out_txt).parent.mkdir(parents=True, exist_ok=True)
    with open(out_txt, 'w', encoding='utf-8') as f:
        f.write("[TRANSCRIPT_PLACEHOLDER]")
    return out_txt