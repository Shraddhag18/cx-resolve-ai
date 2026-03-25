from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager
from pathlib import Path
from app.config import get_settings
from app.routers import chat, dashboard


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Pre-warm the FAISS index on startup
    try:
        from app.routers.chat import get_vectorstore
        get_vectorstore()
        print("FAISS index loaded successfully.")
    except Exception as e:
        print(f"Warning: Could not pre-load index: {e}")
    yield


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=(
            "Tier-1 customer support chatbot powered by OpenAI, LangChain, and FAISS. "
            "Retrieves relevant documents in under 1 second and generates cited answers."
        ),
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(chat.router)
    app.include_router(dashboard.router)

    @app.get("/health", tags=["health"])
    async def health():
        return {"status": "ok", "version": settings.app_version}

    static_dir = Path(__file__).parent.parent / "static"
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.get("/", include_in_schema=False)
    async def ui():
        return FileResponse(static_dir / "index.html")

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
