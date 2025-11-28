from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import json

from utils.config import GOOGLE_API_KEY

# We will *lazily* import google.generativeai inside Gemini
# so that Python 3.9 / env issues don’t crash the module at import time.


class Gemini:
    """
    Minimal wrapper around Gemini. If the real SDK cannot be imported
    (e.g., Python 3.9 issue), it falls back to a dummy model that just
    echoes a short response. That way your pipeline still runs.
    """

    def __init__(self, model: str = "gemini-2.5-flash-lite", **kwargs: Any):
        self.model_name = model
        self.kwargs = kwargs
        self._model = None
        self._real_gemini_available = False

        if not GOOGLE_API_KEY:
            print("⚠️ GOOGLE_API_KEY not set – using dummy Gemini model.")
            return

        try:
            import google.generativeai as genai
            genai.configure(api_key=GOOGLE_API_KEY)
            self._model = genai.GenerativeModel(model)
            self._real_gemini_available = True
        except Exception as e:
            # This is where your Python 3.9 / importlib.metadata issue shows up.
            print("⚠️ Gemini SDK not available, using dummy model instead:", e)
            self._model = None
            self._real_gemini_available = False

    def generate(self, prompt: str, **kwargs: Any) -> str:
        if self._real_gemini_available and self._model is not None:
            resp = self._model.generate_content(prompt, **kwargs)
            if hasattr(resp, "text"):
                return resp.text
            return str(resp)

        # Dummy fallback so the rest of the system can still work
        return (
            "[Dummy Gemini response]\n"
            "Gemini SDK is not available in this environment.\n"
            "Here is a truncated view of the prompt I received:\n\n"
            f"{prompt[:400]}..."
        )


# ---------- Agent (like Kaggle's Agent) ----------

@dataclass
class Agent:
    model: Gemini
    name: str
    description: str
    instruction: str = ""
    tools: List[Any] = field(default_factory=list)

    def build_prompt(
        self,
        user_input: str,
        context: Optional[Dict[str, Any]] = None,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        parts: List[str] = []

        if self.instruction:
            parts.append(f"System instruction:\n{self.instruction}\n")

        if history:
            parts.append("Conversation history:")
            for msg in history:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                parts.append(f"[{role}] {content}")
            parts.append("")

        if context:
            ctx_str = json.dumps(context, indent=2, ensure_ascii=False)
            parts.append(f"Context (JSON):\n{ctx_str}\n")

        parts.append(f"User input:\n{user_input}\n")
        parts.append("Respond clearly and concisely as the described agent.")
        return "\n".join(parts)

    def run(
        self,
        user_input: str,
        context: Optional[Dict[str, Any]] = None,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        prompt = self.build_prompt(user_input=user_input, context=context, history=history)
        return self.model.generate(prompt)


# ---------- In-memory session service ----------

class InMemorySessionService:
    """
    Simple in-memory session store:
    sessions[(app_name, user_id, session_id)] = [ {role, content}, ... ]
    """

    def __init__(self):
        self.sessions: Dict[tuple, List[Dict[str, str]]] = {}

    def get_history(self, app_name: str, user_id: str, session_id: str) -> List[Dict[str, str]]:
        return self.sessions.get((app_name, user_id, session_id), [])

    def append_message(self, app_name: str, user_id: str, session_id: str, role: str, content: str):
        key = (app_name, user_id, session_id)
        if key not in self.sessions:
            self.sessions[key] = []
        self.sessions[key].append({"role": role, "content": content})


# ---------- Runner (like Kaggle's Runner) ----------

class Runner:
    def __init__(self, agent: Agent, app_name: str, session_service: InMemorySessionService):
        self.agent = agent
        self.app_name = app_name
        self.session_service = session_service

    def run(
        self,
        user_id: str,
        session_id: str,
        user_input: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        history = self.session_service.get_history(self.app_name, user_id, session_id)

        self.session_service.append_message(self.app_name, user_id, session_id, "user", user_input)

        response = self.agent.run(
            user_input=user_input,
            context=context,
            history=history,
        )

        self.session_service.append_message(self.app_name, user_id, session_id, "assistant", response)

        return response