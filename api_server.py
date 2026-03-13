"""
FastAPI backend for the Autonomous Research Crew.

Endpoints
---------
GET /api/research?query=...
    Server-Sent Events stream.  Events:
      data: {"type": "agent_start",   "agent": "..."}
      data: {"type": "task_complete", "agent": "...", "output": "..."}
      data: {"type": "result",        "output": "..."}   <- final markdown
      data: {"type": "error",         "message": "..."}
      data: {"type": "done"}

GET /api/health   <- simple liveness check
"""

import asyncio
import json
import queue
import threading

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

app = FastAPI(title="Autonomous Research Crew API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

AGENTS_ORDER = [
    "Research Planner",
    "Research Paper Searcher",
    "Research Paper Summarizer",
    "Research Reporter",
    "Report Assembler",
]


# ---------------------------------------------------------------------------
# Background worker
# ---------------------------------------------------------------------------

def _crew_worker(query: str, q: queue.Queue) -> None:
    """Run the research crew in a background thread, posting events to *q*."""
    try:
        from research_crew import run_research  # late import – heavy module

        def event_callback(event_type: str, data):
            q.put((event_type, data))

        result = run_research(query, event_callback=event_callback)
        q.put(("result", {"output": result}))

    except Exception as exc:
        q.put(("error", {"message": str(exc)}))
    finally:
        q.put(("done", {}))


# ---------------------------------------------------------------------------
# SSE endpoint
# ---------------------------------------------------------------------------

@app.get("/api/research")
async def research_stream(
    query: str = "Recent advancements in energy-efficient IoT systems",
):
    """Stream research crew progress and final report as SSE."""
    q: queue.Queue = queue.Queue()

    thread = threading.Thread(
        target=_crew_worker, args=(query, q), daemon=True
    )
    thread.start()

    async def event_generator():
        loop = asyncio.get_event_loop()
        while True:
            # Offload blocking queue.get to a thread pool so we don't block
            # the event loop.
            try:
                event_type, data = await loop.run_in_executor(
                    None, lambda: q.get(timeout=2.0)
                )
            except queue.Empty:
                # Send a keepalive comment so the connection stays alive
                yield ": keepalive\n\n"
                continue

            payload = json.dumps({"type": event_type, **data} if isinstance(data, dict) else {"type": event_type, "output": data})
            yield f"data: {payload}\n\n"

            if event_type in ("done", "error"):
                break

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@app.get("/api/health")
async def health():
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api_server:app", host="0.0.0.0", port=8080, reload=False)
