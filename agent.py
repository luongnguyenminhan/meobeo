import logging
import os
import time
from typing import AsyncGenerator, List, Tuple

from dotenv import load_dotenv

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import TaskState, UnsupportedOperationError, Part, TextPart
from a2a.types import Message as A2AMessage
from a2a.utils import new_agent_text_message, new_task
from a2a.utils.errors import ServerError

from agno.agent import Agent, Message
from agno.models.google import Gemini

# RAG deps (minimal)
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

try:
    # Preferred embeddings via Agno/Chonkie Gemini embeddings
    from chonkie.embeddings import GeminiEmbeddings  # type: ignore
except Exception:  # pragma: no cover - optional fallback
    GeminiEmbeddings = None  # type: ignore


load_dotenv()


LOGGER = logging.getLogger("meobeo")
LOGGER.setLevel(logging.DEBUG)


class SecretRedactionFilter(logging.Filter):
    def __init__(self, secrets: list[str]):
        super().__init__()
        self._secrets = [s for s in secrets if s]

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            msg = str(record.getMessage())
            redacted = msg
            for secret in self._secrets:
                if secret and secret in redacted:
                    redacted = redacted.replace(secret, "[REDACTED]")
            if redacted != msg:
                record.msg = redacted
        except Exception:
            pass
        return True


handler = logging.StreamHandler()
handler.addFilter(SecretRedactionFilter([os.getenv("GOOGLE_API_KEY", "")]))
LOGGER.addHandler(handler)


# ------------------------------
# RAG configuration (minimal)
# ------------------------------
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "knowledge_base")
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "5"))
RAG_CONTEXT_CHAR_LIMIT = int(os.getenv("RAG_CONTEXT_CHAR_LIMIT", "8000"))  # ~2000 tokens
RAG_NO_SOURCE_NOTE = " (no sources used)"
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")

class MeoBeoAgent:
    """MeoBeo Agent using Google Generative AI via Agno wrapper."""

    def __init__(self) -> None:
        self.persona_instructions = (
            "Bạn là MeoBeo — một trợ lý trò chuyện tiếng Việt vui vẻ, thân thiện, và hài hước.\n"
            "- Luôn giữ ngôn ngữ tích cực, dễ gần, khích lệ người dùng.\n"
            "- Trả lời ngắn gọn, mạch lạc; ưu tiên tính tự nhiên như đang tán gẫu.\n"
            "- Có thể pha trò nhẹ nhàng, dùng emoji tiết chế nếu phù hợp (không lạm dụng).\n"
            "- Không ràng buộc kiểm duyệt gắt gao; tuy nhiên tránh nội dung bất hợp pháp/độc hại.\n"
            "- Tôn trọng người dùng; không tiết lộ bí mật hay khóa API.\n"
        )

        # Initialize Agno Agent with GoogleChat (Gemini 2.0 Flash)
        model_id = "gemini-2.0-flash"
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY is required for GoogleChat")

        self.agent = Agent(
            name="MeoBeoAgent",
            instructions=self.persona_instructions,
            description="Chit-chat vui vẻ bằng tiếng Việt với phong cách thân thiện và dí dỏm.",
            model=Gemini(id=model_id, api_key=api_key, temperature=0.9, max_output_tokens=10000),
            markdown=True,
            debug_mode=False,
        )

    async def invoke(self, message: Message) -> AsyncGenerator[str, None]:
        response = await self.agent.arun(message, stream=True)
        async for chunk in response:
            text = getattr(chunk, "content", None)
            if text:
                yield text

    # ---------------
    # RAG helpers
    # ---------------
    def _get_qdrant_client(self) -> QdrantClient:
        LOGGER.debug(
            "RAG client: creating QdrantClient url=%s collection=%s",
            QDRANT_URL,
            QDRANT_COLLECTION,
        )
        if QDRANT_API_KEY:
            return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        return QdrantClient(url=QDRANT_URL)

    def _embed_query_sync(self, text: str) -> List[float]:
        if not text:
            return []
        if GeminiEmbeddings is None:
            raise RuntimeError("GeminiEmbeddings not available. Please install chonkie.")
        embeddings = GeminiEmbeddings(api_key=os.getenv("GOOGLE_API_KEY"))
        vec = embeddings.embed(text)
        LOGGER.debug(
            "RAG embedding: query_len=%s dim=%s",
            len(text),
            len(vec) if hasattr(vec, "__len__") else 0,
        )
        return list(vec)

    def _ensure_collection(self, client: QdrantClient, vector_dim: int) -> None:
        try:
            exists = False
            try:
                # qdrant-client >=1.6
                exists = client.collection_exists(QDRANT_COLLECTION)
            except Exception:
                # Fallback: try fetch
                client.get_collection(QDRANT_COLLECTION)
                exists = True
            if not exists:
                LOGGER.debug(
                    "RAG create_collection: name=%s dim=%s distance=COSINE",
                    QDRANT_COLLECTION,
                    vector_dim,
                )
                client.create_collection(
                    collection_name=QDRANT_COLLECTION,
                    vectors_config=qmodels.VectorParams(
                        size=vector_dim, distance=qmodels.Distance.COSINE
                    ),
                )
        except Exception as e:
            LOGGER.debug("RAG ensure_collection error: %s", str(e))

    def _truncate_context(self, chunks: List[str], limit_chars: int) -> str:
        if not chunks:
            return ""
        acc: List[str] = []
        total = 0
        for ch in chunks:
            if not ch:
                continue
            n = len(ch)
            if total + n + (2 if acc else 0) > limit_chars:
                break
            acc.append(ch)
            total += n + (2 if acc else 0)
        return "\n\n".join(acc)

    def _build_system_message(self, context_text: str) -> str:
        if not context_text:
            return self.persona_instructions
        extra = (
            "\n\n[Context]\n" + context_text +
            "\n\n[Instruction]\nHãy trả lời dựa trên phần Context ở trên."
        )
        return self.persona_instructions + extra

    async def rag_answer(self, user_text: str) -> Tuple[str, bool, List[float]]:
        """Return (answer_text, used_context, scores)."""
        try:
            t0 = time.time()
            LOGGER.debug("RAG start: top_k=%s", RAG_TOP_K)
            qvec = self._embed_query_sync(user_text)
            client = self._get_qdrant_client()
            LOGGER.debug(
                "RAG search: collection=%s top_k=%s url=%s",
                QDRANT_COLLECTION,
                RAG_TOP_K,
                QDRANT_URL,
            )
            # Ensure collection exists (create if missing) using query vector dimension
            self._ensure_collection(client, len(qvec) if hasattr(qvec, "__len__") else 0)
            results = client.search(
                collection_name=QDRANT_COLLECTION,
                query_vector=qvec,
                limit=RAG_TOP_K,
                with_payload=True,
            )
            elapsed_ms = int((time.time() - t0) * 1000)

            contexts: List[str] = []
            scores: List[float] = []
            for idx, r in enumerate(results or []):
                payload = getattr(r, "payload", {}) or {}
                text = payload.get("summary_text") or payload.get("text") or ""
                if text:
                    contexts.append(str(text))
                    try:
                        s = float(getattr(r, "score", 0.0))
                        scores.append(s)
                    except Exception:
                        scores.append(0.0)
                LOGGER.debug(
                    "RAG result[%s]: score=%s has_summary=%s summary_len=%s",
                    idx,
                    round(float(getattr(r, "score", 0.0)), 4) if hasattr(r, "score") else 0.0,
                    bool(payload.get("summary_text")),
                    len(text),
                )
                # content truncate to 200 characters
                LOGGER.debug("RAG result[%s]: text=%s", idx, text[:200])

            LOGGER.debug(
                "RAG search done: ms=%s top_k=%s results=%s scores=%s",
                elapsed_ms,
                RAG_TOP_K,
                len(contexts),
                [round(s, 4) for s in scores],
            )

            context_text = self._truncate_context(contexts, RAG_CONTEXT_CHAR_LIMIT)
            used_context = bool(context_text)
            LOGGER.debug(
                "RAG context: used=%s chars=%s pieces=%s",
                used_context,
                len(context_text),
                len(contexts),
            )
            system_msg = self._build_system_message(context_text)

            # One-shot answer with context baked into system instructions
            temp_agent = Agent(
                name="MeoBeoAgent",
                instructions=system_msg,
                description=self.agent.description,
                model=self.agent.model,
                markdown=True,
                debug_mode=False,
            )
            resp = await temp_agent.arun(Message(role="user", content=user_text))
            answer_text = getattr(resp, "content", "") or ""
            return answer_text, used_context, scores
        except Exception as e:  # fallback to non-RAG
            LOGGER.debug("RAG error: %s", str(e))
            temp_agent = Agent(
                name="MeoBeoAgent",
                instructions=self.persona_instructions,
                description=self.agent.description,
                model=self.agent.model,
                markdown=True,
                debug_mode=False,
            )
            resp = await temp_agent.arun(Message(role="user", content=user_text))
            answer_text = getattr(resp, "content", "") or ""
            LOGGER.debug("RAG fallback: using non-RAG answer with note")
            return answer_text + RAG_NO_SOURCE_NOTE, False, []


class MeoBeoAgentExecutor(AgentExecutor):
    """AgentExecutor that streams raw chunks without buffering."""

    def __init__(self) -> None:
        self.agent = MeoBeoAgent()

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        task = context.current_task

        # Extract user message text
        user_message_text: str = ""
        for part in context.message.parts:
            if isinstance(part, Part) and isinstance(part.root, TextPart):
                user_message_text = part.root.text
                break

        if not task:
            # If incoming message has empty text, synthesize a minimal A2A message to satisfy protocol
            if not user_message_text.strip():
                safe_msg = A2AMessage(parts=[Part(root=TextPart(text="(empty input)"))])  # type: ignore
                task = new_task(safe_msg)  # type: ignore
            else:
                task = new_task(context.message)  # type: ignore
            await event_queue.enqueue_event(task)

        updater = TaskUpdater(event_queue, task.id, task.context_id)

        # Greet cheerfully at session start if input is empty
        if not user_message_text.strip():
            greeting = (
                "Xin chào! MeoBeo đây 😺 Mình sẵn sàng tám chuyện cùng bạn. "
                "Bạn đang nghĩ gì nè?"
            )
            await updater.update_status(
                TaskState.working,
                new_agent_text_message(greeting, task.context_id, task.id),
            )
            # Mark task completed to properly close the stream
            return

        try:
            # Retrieve first, then single final answer
            LOGGER.debug("Executor: invoking RAG answer")
            answer_text, used_context, scores = await self.agent.rag_answer(
                user_message_text
            )
            LOGGER.debug(
                "Executor: answer ready (used_context=%s, answer_len=%s)",
                used_context,
                len(answer_text),
            )
            await updater.update_status(
                TaskState.working,
                new_agent_text_message(answer_text, task.context_id, task.id),
            )
        except Exception as e:
            friendly_error = (
                f"Ôi không, MeoBeo đang hơi bối rối một chút. Bạn thử nói lại {e} "
                "hoặc đợi mình vài giây rồi nhắn tiếp nhé!"
            )
            await updater.update_status(
                TaskState.working,
                new_agent_text_message(friendly_error, task.context_id, task.id),
            )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise ServerError(error=UnsupportedOperationError())


