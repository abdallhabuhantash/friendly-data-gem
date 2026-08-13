"""FastAPI entry point for the local AI service.

The API exposes only what the console needs: a health probe, an operational
status document and the annotated MJPEG stream. Camera credentials, RTSP URLs
and the Supabase service-role key are never part of any response.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, Header, HTTPException, Response
from fastapi.responses import StreamingResponse

from .config import get_settings
from .logging_config import configure_logging
from .runtime.orchestrator import Orchestrator
from .security import verify_service_key

logger = logging.getLogger(__name__)

BOUNDARY = "frame"


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(
        settings.log_level,
        secrets=[settings.supabase_service_role_key, settings.telegram_bot_token, settings.ai_service_key],
    )
    orchestrator = Orchestrator(settings)
    app.state.orchestrator = orchestrator
    await asyncio.to_thread(orchestrator.start)
    try:
        yield
    finally:
        await asyncio.to_thread(orchestrator.stop)


app = FastAPI(title="Vigilant Eye AI Service", version="1.0.0", lifespan=lifespan)


def _require_key(provided: Optional[str]) -> None:
    settings = get_settings()
    if not verify_service_key(settings.ai_service_key, provided):
        raise HTTPException(status_code=401, detail="Unauthorized")


@app.get("/health")
async def health() -> dict:
    """Unauthenticated liveness probe. Contains no operational detail."""
    return {"status": "ok", "version": get_settings().service_version}


@app.get("/status")
async def status(x_service_key: Optional[str] = Header(default=None)) -> dict:
    _require_key(x_service_key)
    return app.state.orchestrator.status()


@app.post("/exam-sessions/{exam_session_id}/arm")
async def arm_exam_session(exam_session_id: str, x_service_key: Optional[str] = Header(default=None)) -> dict:
    """Arms anonymous subject monitoring for one configured exam session.

    Deliberately explicit: paper distribution at the start of an exam must not
    be monitored, so nothing is armed until the console calls this endpoint.
    """
    _require_key(x_service_key)
    orchestrator: Orchestrator = app.state.orchestrator
    try:
        return await asyncio.to_thread(orchestrator.arm_exam_session, exam_session_id)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/exam-sessions/{exam_session_id}/end")
async def end_exam_session(exam_session_id: str, x_service_key: Optional[str] = Header(default=None)) -> dict:
    """Disarms monitoring and closes every anonymous subject of the session."""
    _require_key(x_service_key)
    orchestrator: Orchestrator = app.state.orchestrator
    try:
        return await asyncio.to_thread(orchestrator.end_exam_session, exam_session_id)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.get("/stream/{camera_id}")
async def stream(camera_id: str, x_service_key: Optional[str] = Header(default=None)):
    """Annotated MJPEG for one camera, shared across all viewers."""
    _require_key(x_service_key)
    orchestrator: Orchestrator = app.state.orchestrator
    if orchestrator.cameras.worker(camera_id) is None:
        raise HTTPException(status_code=404, detail="Camera is not being processed")

    async def frames():
        # No placeholder imagery is ever produced: when there is no annotated
        # frame the stream simply stays quiet until inference catches up.
        while True:
            jpeg = orchestrator.stream_hub.latest(camera_id)
            if jpeg:
                yield (
                    b"--" + BOUNDARY.encode() + b"\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"Content-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n" + jpeg + b"\r\n"
                )
            await asyncio.sleep(0.1)
            if orchestrator.cameras.worker(camera_id) is None:
                return

    return StreamingResponse(
        frames(),
        media_type=f"multipart/x-mixed-replace; boundary={BOUNDARY}",
        headers={"Cache-Control": "no-store"},
    )


@app.get("/snapshot/{camera_id}")
async def snapshot(camera_id: str, x_service_key: Optional[str] = Header(default=None)):
    """Single annotated JPEG, useful for diagnostics and thumbnails."""
    _require_key(x_service_key)
    jpeg = app.state.orchestrator.stream_hub.latest(camera_id)
    if not jpeg:
        raise HTTPException(status_code=404, detail="No annotated frame available")
    return Response(content=jpeg, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


def run() -> None:
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
        reload=False,
    )


if __name__ == "__main__":
    run()