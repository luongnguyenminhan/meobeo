import httpx
from typing import List

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import (
    InMemoryTaskStore,
    InMemoryPushNotificationConfigStore,
    BasePushNotificationSender,
)
from a2a.types import AgentCapabilities, AgentCard, AgentSkill

from agent import QDRANT_URL, QDRANT_COLLECTION, QDRANT_API_KEY, MeoBeoAgentExecutor  # reuse config
import logging
import os


if __name__ == "__main__":
    # Enable detailed RAG logs
    logging.getLogger("meobeo").setLevel(logging.DEBUG)
    # Define skills (public card)
    public_skill = AgentSkill(
        id="meobeo_chat",
        name="MeoBeo Chat",
        description="Chit-chat vui vẻ bằng tiếng Việt, thân thiện và hài hước.",
        tags=["chat", "fun"],
        examples=[
            "Tâm trạng hôm nay của mình hơi chán, nói chuyện với mình với?",
            "Kể mình nghe một câu chuyện vui ngắn đi!",
            "Gợi ý giúp mình vài ý tưởng đi chơi cuối tuần nha.",
        ],
    )

    # Extended card details
    long_description = (
        "MeoBeo là bạn đồng hành trò chuyện bằng tiếng Việt với phong cách vui vẻ, "
        "thân thiện và dí dỏm. Mục tiêu của MeoBeo là giúp bạn cảm thấy thoải mái "
        "khi trò chuyện, đưa ra gợi ý hữu ích, và luôn giữ cuộc hội thoại nhẹ nhàng, "
        "tự nhiên như đang tán gẫu. MeoBeo tôn trọng quyền riêng tư và sẽ không bao giờ "
        "tiết lộ thông tin nhạy cảm như khóa API hay bí mật hệ thống."
    )

    advanced_examples: List[str] = [
        "Hãy giả làm một người bạn cũ chào hỏi mình thật tự nhiên được không?",
        "Mình hơi stress vì công việc, có thể trò chuyện và đưa vài mẹo thư giãn ngắn không?",
        "Hãy đóng vai hướng dẫn viên địa phương, gợi ý một buổi tối thú vị tại Hà Nội.",
    ]

    public_agent_card = AgentCard(
        name="MeoBeoAgent",
        description="Trợ lý trò chuyện tiếng Việt vui vẻ, thân thiện.",
        url="http://localhost:9995/",
        version="1.0.0",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=AgentCapabilities(streaming=True, push_notifications=True),
        skills=[public_skill],
        supports_authenticated_extended_card=True,
    )

    extended_agent_card = AgentCard(
        name="MeoBeoAgent (Extended)",
        description=long_description,
        url="http://localhost:9995/",
        version="1.0.0",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=AgentCapabilities(streaming=True, push_notifications=True),
        skills=[
            AgentSkill(
                id="meobeo_chat_advanced",
                name="MeoBeo Chat (Advanced)",
                description="Ví dụ nâng cao và mô tả chi tiết hơn.",
                tags=["chat", "fun"],
                examples=advanced_examples,
            )
        ],
        supports_authenticated_extended_card=True,
    )

    push_config_store = InMemoryPushNotificationConfigStore()
    push_sender = BasePushNotificationSender(
        httpx_client=httpx.AsyncClient(), config_store=push_config_store
    )

    request_handler = DefaultRequestHandler(
        agent_executor=MeoBeoAgentExecutor(),
        task_store=InMemoryTaskStore(),
        push_config_store=push_config_store,
        push_sender=push_sender,
    )

    # Build server
    server = A2AStarletteApplication(
        agent_card=public_agent_card,
        http_handler=request_handler,
        extended_agent_card=extended_agent_card,
    )

    import uvicorn

    app = server.build()

    # Minimal endpoint: index uploaded file into Qdrant (single function)
    from starlette.responses import JSONResponse, HTMLResponse
    from starlette.requests import Request
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as qmodels
    import uuid
    import re
    import json

    async def index_file(request: Request):
        try:
            form = await request.form()
            file = form.get("file")
            if file is None:
                return JSONResponse({"detail": "file is required"}, status_code=400)

            filename = getattr(file, "filename", "upload.txt")
            content_type = getattr(file, "content_type", None)
            content_bytes = await file.read()

            logger = logging.getLogger("meobeo")
            logger.debug(
                "Upload received: file=%s content_type=%s size_bytes=%s",
                filename,
                content_type,
                len(content_bytes),
            )

            # Detect PDF and extract text via PyMuPDF; else handle as text
            is_pdf = (content_type and "pdf" in content_type.lower()) or filename.lower().endswith(".pdf")
            if is_pdf:
                try:
                    import fitz  # PyMuPDF
                except Exception as e:
                    return JSONResponse({"detail": f"PyMuPDF not installed: {e}"}, status_code=500)

                try:
                    doc = fitz.open(stream=content_bytes, filetype="pdf")
                    page_count = len(doc)
                    pages = []
                    for page in doc:
                        pages.append(page.get_text("text", sort=True))
                    doc.close()
                    text = "\n".join(pages)
                    logger.debug(
                        "PDF extracted: pages=%s chars=%s utf8_valid=%s preview='%s'",
                        page_count,
                        len(text),
                        True if isinstance(text, str) else False,
                        text[:120].replace("\n", "\\n"),
                    )
                except Exception as e:
                    return JSONResponse({"detail": f"failed to read pdf: {e}"}, status_code=400)
            else:
                # Basic binary detection and content-type guard for non-PDF
                if b"\x00" in content_bytes:
                    return JSONResponse({"detail": "binary files are not supported; please upload text-based files"}, status_code=400)
                if content_type and not (
                    content_type.startswith("text/") or content_type in ("application/json", "application/xml")
                ):
                    return JSONResponse({"detail": f"unsupported content-type: {content_type}"}, status_code=400)
                try:
                    text = content_bytes.decode("utf-8")
                    logger.debug(
                        "Text decode utf-8 OK: chars=%s preview='%s'",
                        len(text),
                        text[:120].replace("\n", "\\n"),
                    )
                except Exception:
                    text = content_bytes.decode("utf-8", errors="replace")
                    logger.debug(
                        "Text decode utf-8 with replacement: chars=%s preview='%s'",
                        len(text),
                        text[:120].replace("\n", "\\n"),
                    )

            # Simple chunking (~1000 chars, trim whitespace)
            orig_len = len(text)
            text = re.sub(r"\r\n|\r", "\n", text).strip()
            # Remove non-printable control chars (keep tabs/newlines)
            text = re.sub(r"[^\x09\x0A\x0D\x20-\uFFFF]", "", text)
            removed = orig_len - len(text)
            # Validate utf-8 roundtrip
            utf8_ok = True
            try:
                _ = text.encode("utf-8")
            except Exception:
                utf8_ok = False
            logger.debug(
                "Sanitized text: orig_chars=%s removed_ctrl=%s final_chars=%s utf8_ok=%s preview='%s'",
                orig_len,
                removed,
                len(text),
                utf8_ok,
                text[:120].replace("\n", "\\n"),
            )
            chunks = []
            step = 1000
            for i in range(0, len(text), step):
                chunk = text[i : i + step].strip()
                if chunk:
                    chunks.append(chunk)

            if not chunks:
                return JSONResponse({"detail": "no content to index"}, status_code=400)
            logger.debug("Chunking done: chunks=%s avg_len=~%s", len(chunks), (len(text) // len(chunks)) if chunks else 0)

            # Embeddings (Gemini via chonkie)
            try:
                from chonkie.embeddings import GeminiEmbeddings  # type: ignore
            except Exception as e:
                return JSONResponse({"detail": f"embeddings unavailable: {e}"}, status_code=500)

            embeddings = GeminiEmbeddings(api_key=os.getenv("GOOGLE_API_KEY"))
            vectors = [list(embeddings.embed(c)) for c in chunks]
            vec_dim = len(vectors[0]) if vectors and hasattr(vectors[0], "__len__") else 0
            logger.debug("Embeddings: vectors=%s dim=%s", len(vectors), vec_dim)

            # Qdrant client
            qdrant_url = os.getenv("QDRANT_URL", QDRANT_URL)
            qdrant_key = os.getenv("QDRANT_API_KEY", QDRANT_API_KEY)
            client = QdrantClient(url=qdrant_url, api_key=qdrant_key or None)

            # Ensure collection exists
            dim = len(vectors[0]) if vectors and hasattr(vectors[0], "__len__") else 0
            try:
                exists = client.collection_exists(QDRANT_COLLECTION)
            except Exception:
                try:
                    client.get_collection(QDRANT_COLLECTION)
                    exists = True
                except Exception:
                    exists = False
            if not exists and dim > 0:
                client.create_collection(
                    collection_name=QDRANT_COLLECTION,
                    vectors_config=qmodels.VectorParams(size=dim, distance=qmodels.Distance.COSINE),
                )
                logger.debug("Qdrant: collection created name=%s dim=%s", QDRANT_COLLECTION, dim)

            # Upsert
            points = []
            for idx, vec in enumerate(vectors):
                points.append(
                    qmodels.PointStruct(
                        id=str(uuid.uuid4()),
                        vector=vec,
                        payload={
                            "summary_text": chunks[idx],
                            "text": chunks[idx],
                            "source_file": filename,
                            "chunk_index": idx,
                        },
                    )
                )
            client.upsert(collection_name=QDRANT_COLLECTION, points=points)
            logger.debug("Qdrant: upserted points=%s collection=%s", len(points), QDRANT_COLLECTION)

            return JSONResponse({
                "indexed": len(points),
                "collection": QDRANT_COLLECTION,
                "file": filename,
            })
        except Exception as e:
            return JSONResponse({"detail": str(e)}, status_code=500)

    # Register route (POST /index-file)
    try:
        app.router.add_route("/index-file", index_file, methods=["POST"])  # type: ignore
    except Exception:
        # Fallback if add_route not available
        from starlette.routing import Route

        app.router.routes.append(Route("/index-file", endpoint=index_file, methods=["POST"]))  # type: ignore

    # ------------------------------
    # Swagger/OpenAPI (very minimal)
    # ------------------------------
    async def get_openapi(request: Request):
        spec = {
            "openapi": "3.0.0",
            "info": {"title": "MeoBeo RAG API", "version": "1.0.0"},
            "paths": {
                "/index-file": {
                    "post": {
                        "summary": "Index a file into Qdrant",
                        "requestBody": {
                            "required": True,
                            "content": {
                                "multipart/form-data": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "file": {"type": "string", "format": "binary"}
                                        },
                                        "required": ["file"],
                                    }
                                }
                            },
                        },
                        "responses": {
                            "200": {
                                "description": "Indexed",
                                "content": {
                                    "application/json": {
                                        "schema": {
                                            "type": "object",
                                            "properties": {
                                                "indexed": {"type": "integer"},
                                                "collection": {"type": "string"},
                                                "file": {"type": "string"},
                                            },
                                        }
                                    }
                                },
                            },
                            "400": {"description": "Bad Request"},
                            "500": {"description": "Server Error"},
                        },
                    }
                }
            },
        }
        return JSONResponse(spec)

    async def swagger_ui(request: Request):
        html = """
<!DOCTYPE html>
<html>
  <head>
    <meta charset=\"utf-8\" />
    <title>MeoBeo RAG API Docs</title>
    <link rel=\"stylesheet\" href=\"https://unpkg.com/swagger-ui-dist@5.17.14/swagger-ui.css\" />
  </head>
  <body>
    <div id=\"swagger-ui\"></div>
    <script src=\"https://unpkg.com/swagger-ui-dist@5.17.14/swagger-ui-bundle.js\"></script>
    <script>
      window.ui = SwaggerUIBundle({
        url: '/openapi.json',
        dom_id: '#swagger-ui',
        presets: [SwaggerUIBundle.presets.apis],
        layout: 'BaseLayout'
      });
    </script>
  </body>
</html>
"""
        return HTMLResponse(html)

    try:
        app.router.add_route("/openapi.json", get_openapi, methods=["GET"])  # type: ignore
        app.router.add_route("/docs", swagger_ui, methods=["GET"])  # type: ignore
    except Exception:
        from starlette.routing import Route

        app.router.routes.append(Route("/openapi.json", endpoint=get_openapi, methods=["GET"]))  # type: ignore
        app.router.routes.append(Route("/docs", endpoint=swagger_ui, methods=["GET"]))  # type: ignore

    uvicorn.run(app, host="0.0.0.0", port=9995, timeout_keep_alive=10)


