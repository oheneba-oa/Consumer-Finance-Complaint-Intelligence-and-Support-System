FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

COPY requirements.txt .

# Install CPU-only PyTorch first to avoid unnecessary CUDA packages.
RUN python -m pip install --upgrade pip \
    && python -m pip install torch --index-url https://download.pytorch.org/whl/cpu \
    && python -m pip install -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["sh", "-c", "if [ ! -f data/chroma_db/chroma.sqlite3 ]; then echo 'Knowledge base not found. Building CFPB knowledge base...'; python build_knowledge_base.py; fi && exec streamlit run main.py --server.address=0.0.0.0 --server.port=8501"]