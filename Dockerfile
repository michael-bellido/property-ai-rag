# Property AI — containerized Streamlit app.
#
# Build:
#   docker build -t property-ai .
# Run (GROQ_API_KEY is required at runtime, never baked into the image):
#   docker run -p 8501:8501 -e GROQ_API_KEY=your_key_here property-ai
# Then open http://localhost:8501

FROM python:3.11-slim

WORKDIR /app

# build-essential: some transitive deps (e.g. chromadb's dependencies) need
# a C compiler on slim images that don't ship prebuilt wheels for every
# platform. curl: used only by the HEALTHCHECK below.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ app/
COPY data/ data/
COPY .streamlit/ .streamlit/

# Build the vector store at image-build time instead of leaving it to
# app.py's runtime fallback (which exists for platforms like Streamlit
# Community Cloud that have no separate build step). Doing it here bakes
# chroma_store/ and the downloaded embedding model into the image, so a
# container starts serving immediately instead of doing that work on its
# first request.
RUN python app/ingest.py

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s CMD \
    curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
