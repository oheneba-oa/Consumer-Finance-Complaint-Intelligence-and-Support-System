from pathlib import Path
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
import chromadb
import re


# --------------------------------------------------
# SETTINGS
# --------------------------------------------------
DOCUMENTS_PATH = Path("data/documents")

CHUNK_SIZE = 120
CHUNK_OVERLAP = 20

COLLECTION_NAME = "customer_service_knowledge"


# --------------------------------------------------
# CLEAN TEXT
# --------------------------------------------------
def clean_text(text):
    """
    Remove unnecessary whitespace while preserving
    the actual content of the document.
    """

    text = re.sub(r"\s+", " ", text)

    return text.strip()


# --------------------------------------------------
# CHUNK TEXT
# --------------------------------------------------
def chunk_text(
    text,
    chunk_size=120,
    overlap=20
):

    words = text.split()

    chunks = []

    start = 0

    while start < len(words):

        end = start + chunk_size

        chunk = words[start:end]

        if chunk:
            chunks.append(
                " ".join(chunk)
            )

        start += (
            chunk_size - overlap
        )

    return chunks


# --------------------------------------------------
# READ TXT FILE
# --------------------------------------------------
def read_txt(file_path):

    return file_path.read_text(
        encoding="utf-8"
    )


# --------------------------------------------------
# READ PDF FILE
# --------------------------------------------------
def read_pdf(file_path):

    reader = PdfReader(file_path)

    pages = []

    for page_number, page in enumerate(
        reader.pages,
        start=1
    ):

        text = page.extract_text()

        if text and text.strip():

            pages.append(
                {
                    "page": page_number,
                    "text": text
                }
            )

    return pages


# --------------------------------------------------
# LOAD EMBEDDING MODEL
# --------------------------------------------------
print(
    "Loading embedding model..."
)

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# --------------------------------------------------
# CONNECT TO CHROMA DATABASE
# --------------------------------------------------
client = chromadb.PersistentClient(
    path="data/chroma_db"
)


# --------------------------------------------------
# REMOVE OLD COLLECTION
# --------------------------------------------------
try:

    client.delete_collection(
        name=COLLECTION_NAME
    )

    print(
        "Old knowledge base removed."
    )

except Exception:

    pass


# --------------------------------------------------
# CREATE NEW COLLECTION
# --------------------------------------------------
collection = client.create_collection(

    name=COLLECTION_NAME,

    metadata={
        "hnsw:space": "cosine"
    }
)


# --------------------------------------------------
# COUNTERS
# --------------------------------------------------
total_chunks = 0


# --------------------------------------------------
# PROCESS TXT DOCUMENTS
# --------------------------------------------------
for file_path in DOCUMENTS_PATH.glob(
    "*.txt"
):

    print(
        f"Processing TXT: "
        f"{file_path.name}"
    )

    raw_text = read_txt(
        file_path
    )

    cleaned_text = clean_text(
        raw_text
    )

    chunks = chunk_text(
        cleaned_text,
        chunk_size=CHUNK_SIZE,
        overlap=CHUNK_OVERLAP
    )


    for chunk_number, chunk in enumerate(
        chunks,
        start=1
    ):

        embedding = (
            embedding_model
            .encode(chunk)
            .tolist()
        )

        chunk_id = (
            f"{file_path.stem}_"
            f"txt_"
            f"{chunk_number}"
        )

        collection.add(

            ids=[
                chunk_id
            ],

            documents=[
                chunk
            ],

            embeddings=[
                embedding
            ],

            metadatas=[
                {
                    "source": file_path.name,
                    "file_type": "txt",
                    "page": 0,
                    "chunk": chunk_number
                }
            ]
        )

        total_chunks += 1


# --------------------------------------------------
# PROCESS PDF DOCUMENTS
# --------------------------------------------------
for file_path in DOCUMENTS_PATH.glob(
    "*.pdf"
):

    print(
        f"Processing PDF: "
        f"{file_path.name}"
    )

    pages = read_pdf(
        file_path
    )


    for page_data in pages:

        page_number = (
            page_data["page"]
        )

        cleaned_text = clean_text(
            page_data["text"]
        )

        chunks = chunk_text(
            cleaned_text,
            chunk_size=CHUNK_SIZE,
            overlap=CHUNK_OVERLAP
        )


        for chunk_number, chunk in enumerate(
            chunks,
            start=1
        ):

            embedding = (
                embedding_model
                .encode(chunk)
                .tolist()
            )

            chunk_id = (
                f"{file_path.stem}_"
                f"pdf_"
                f"p{page_number}_"
                f"c{chunk_number}"
            )

            collection.add(

                ids=[
                    chunk_id
                ],

                documents=[
                    chunk
                ],

                embeddings=[
                    embedding
                ],

                metadatas=[
                    {
                        "source": file_path.name,
                        "file_type": "pdf",
                        "page": page_number,
                        "chunk": chunk_number
                    }
                ]
            )

            total_chunks += 1


# --------------------------------------------------
# COMPLETE
# --------------------------------------------------
print()

print(
    "Knowledge base created successfully."
)

print(
    f"Total chunks stored: "
    f"{total_chunks}"
)