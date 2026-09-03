"""
Speech-to-Text.

Transcribes audio into text BEFORE it enters the same NLP pipeline used for
text requests (normalizer -> intent -> entities -> resolver -> LLM tool
selection). Because ISpeechToText is a narrow interface, the transcription
backend is swappable independently of everything else.

WhisperSpeechToText wraps OpenAI's Whisper (or any whisper-API-compatible
model). The import is done lazily inside __init__ so that environments
which don't need speech input (text-only deployments) don't pay the cost of
installing/loading a speech model. A MockSpeechToText is provided for tests
and for local development without the dependency installed.
"""
from __future__ import annotations

from app.domain.interfaces import ISpeechToText


class MockSpeechToText(ISpeechToText):
    """
    Deterministic stand-in used in tests/dev: returns a fixed or
    injected transcript instead of running a real model. This keeps the
    pipeline testable end-to-end without bundling a speech model.
    """

    def __init__(self, canned_transcript: str = ""):
        self._canned_transcript = canned_transcript

    def transcribe(self, audio_bytes: bytes, content_type: str) -> str:
        if not audio_bytes:
            raise ValueError("No audio data provided")
        return self._canned_transcript


class WhisperSpeechToText(ISpeechToText):
    """
    Production implementation backed by openai-whisper (or faster-whisper).
    Not installed by default in this codebase to keep the base install
    lightweight; install `openai-whisper` (or `faster-whisper`) and this
    class will load a model lazily on first use.
    """

    def __init__(self, model_size: str = "base"):
        self._model_size = model_size
        self._model = None  # lazy-loaded

    def _ensure_model_loaded(self):
        if self._model is None:
            try:
                import whisper  # type: ignore
            except ImportError as e:
                raise RuntimeError(
                    "openai-whisper is not installed. Run `pip install openai-whisper` "
                    "to enable speech-to-text, or inject a different ISpeechToText "
                    "implementation."
                ) from e
            self._model = whisper.load_model(self._model_size)

    def transcribe(self, audio_bytes: bytes, content_type: str) -> str:
        import tempfile

        self._ensure_model_loaded()
        suffix = _suffix_for_content_type(content_type)
        with tempfile.NamedTemporaryFile(suffix=suffix) as tmp:
            tmp.write(audio_bytes)
            tmp.flush()
            result = self._model.transcribe(tmp.name)
        return result.get("text", "").strip()


def _suffix_for_content_type(content_type: str) -> str:
    mapping = {
        "audio/wav": ".wav", "audio/x-wav": ".wav",
        "audio/mpeg": ".mp3", "audio/mp3": ".mp3",
        "audio/m4a": ".m4a", "audio/webm": ".webm",
        "audio/ogg": ".ogg",
    }
    return mapping.get(content_type, ".wav")
