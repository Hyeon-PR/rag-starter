"""Context Management RAG starter — indexer.

Walks documents/, chunks each file, embeds chunks, persists the index to disk
so the chat backend can load it without re-indexing.

TODO: implement chunk_text(). The embedding and storage code is provided so
you can focus on the structure.
"""
import pickle
from pathlib import Path

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

# Multilingual (50+ languages), 384-dim — same model as the /embedding project.
# Lets the corpus and the queries be in different languages and still match.
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
INDEX_PATH = Path(__file__).parent / "index.pkl"
DOCS_DIR = Path(__file__).parent / "documents"


# ════════════════════════════════════════════════════════════════
# TODO — implement chunk_text
#
# Split `text` into overlapping chunks. A reasonable default:
#   - ~1000 characters per chunk
#   - ~100 characters of overlap
#   - try to break on paragraph boundaries (\n\n) when possible
#
# Return a list of non-empty strings.
# See the lecture slide on chunking for one working implementation.
# ════════════════════════════════════════════════════════════════

def chunk_text(text: str, target_chars: int = 1000, overlap_chars: int = 100) -> list[str]:
    """Split text into overlapping chunks, preferring paragraph boundaries.

    Greedily packs paragraphs (split on blank lines) into chunks of up to
    ~target_chars. Each chunk carries ~overlap_chars of trailing context from
    the previous chunk so retrieval doesn't lose information across cut points.
    Paragraphs longer than target_chars are split by character window.
    """
    # Split on blank lines, keeping non-empty paragraphs.
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    # Break up any single paragraph that exceeds the target size.
    pieces: list[str] = []
    for para in paragraphs:
        if len(para) <= target_chars:
            pieces.append(para)
        else:
            for start in range(0, len(para), target_chars):
                pieces.append(para[start:start + target_chars])

    chunks: list[str] = []
    current = ""
    for piece in pieces:
        if current and len(current) + 2 + len(piece) > target_chars:
            chunks.append(current)
            # Carry the tail of the finished chunk as overlap context.
            tail = current[-overlap_chars:] if overlap_chars > 0 else ""
            current = (tail + "\n\n" + piece) if tail else piece
        else:
            current = (current + "\n\n" + piece) if current else piece

    if current:
        chunks.append(current)

    return [c for c in chunks if c.strip()]


# ════════════════════════════════════════════════════════════════
# Provided: embedding (sentence-transformers, no API key required)
# ════════════════════════════════════════════════════════════════

_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        print(f"Loading embedding model ({MODEL_NAME})... (one-time download ~470MB)")
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed(texts: list[str]) -> list[list[float]]:
    """Embed a list of strings. Returns unit-normalized 384-dim vectors."""
    model = get_model()
    vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return vectors.tolist()


# ════════════════════════════════════════════════════════════════
# Provided: build / save / load / search
# ════════════════════════════════════════════════════════════════

def read_pdf(path: Path) -> str:
    """Extract plain text from a PDF file, joining all pages."""
    reader = PdfReader(path)
    return "\n\n".join(
        page.extract_text() or "" for page in reader.pages
    )


def build_index() -> list[dict]:
    """Walk DOCS_DIR, chunk each file, embed, return list of records."""
    records: list[dict] = []
    chunk_id = 0
    for path in sorted(DOCS_DIR.glob("*")):
        if path.is_dir() or path.suffix.lower() not in (".md", ".txt", ".pdf"):
            continue
        if path.suffix.lower() == ".pdf":
            text = read_pdf(path)
        else:
            text = path.read_text(encoding='utf-8', errors='ignore')
        chunks = chunk_text(text)
        if not chunks:
            continue
        vectors = embed(chunks)
        for i, (chunk, vec) in enumerate(zip(chunks, vectors)):
            records.append({
                "chunk_id": chunk_id,
                "source": path.name,
                "chunk_index": i,
                "text": chunk,
                "embedding": vec,
            })
            chunk_id += 1
        print(f"  {path.name}: {len(chunks)} chunks")
    return records


def save_index(records: list[dict]) -> None:
    with INDEX_PATH.open("wb") as f:
        pickle.dump(records, f)


def load_index() -> list[dict]:
    if not INDEX_PATH.exists():
        raise FileNotFoundError(
            f"No index found at {INDEX_PATH}. Run `python indexer.py` from the project root first."
        )
    with INDEX_PATH.open("rb") as f:
        return pickle.load(f)


def cosine_distance(a: list[float], b: list[float]) -> float:
    # Both vectors are unit-normalized, so cosine distance == 1 - dot product.
    return 1.0 - sum(x * y for x, y in zip(a, b))


def search(query: str, records: list[dict], k: int = 5) -> list[dict]:
    """Embed the query, return top-k records by cosine distance."""
    [query_vec] = embed([query])
    scored = [(cosine_distance(r["embedding"], query_vec), r) for r in records]
    scored.sort(key=lambda x: x[0])
    return [r for _, r in scored[:k]]


def main() -> None:
    print(f"Indexing documents from {DOCS_DIR}/")
    records = build_index()
    save_index(records)
    print(f"\n✓ Indexed {len(records)} chunks → {INDEX_PATH.name}")


if __name__ == "__main__":
    main()
    
"""Context Management RAG starter — indexer.

Walks documents/, chunks each file, embeds chunks, persists the index to disk
so the chat backend can load it without re-indexing.

TODO: implement chunk_text(). The embedding and storage code is provided so
you can focus on the structure.
"""
import pickle
from pathlib import Path

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

# Multilingual (50+ languages), 384-dim — same model as the /embedding project.
# Lets the corpus and the queries be in different languages and still match.
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
INDEX_PATH = Path(__file__).parent / "index.pkl"
DOCS_DIR = Path(__file__).parent / "documents"


# ════════════════════════════════════════════════════════════════
# TODO — implement chunk_text
#
# Split `text` into overlapping chunks. A reasonable default:
#   - ~1000 characters per chunk
#   - ~100 characters of overlap
#   - try to break on paragraph boundaries (\n\n) when possible
#
# Return a list of non-empty strings.
# See the lecture slide on chunking for one working implementation.
# ════════════════════════════════════════════════════════════════

def chunk_text(text: str, target_chars: int = 1000, overlap_chars: int = 100) -> list[str]:
    """Split text into overlapping chunks, preferring paragraph boundaries.

    Greedily packs paragraphs (split on blank lines) into chunks of up to
    ~target_chars. Each chunk carries ~overlap_chars of trailing context from
    the previous chunk so retrieval doesn't lose information across cut points.
    Paragraphs longer than target_chars are split by character window.
    """
    # Split on blank lines, keeping non-empty paragraphs.
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    # Break up any single paragraph that exceeds the target size.
    pieces: list[str] = []
    for para in paragraphs:
        if len(para) <= target_chars:
            pieces.append(para)
        else:
            for start in range(0, len(para), target_chars):
                pieces.append(para[start:start + target_chars])

    chunks: list[str] = []
    current = ""
    for piece in pieces:
        if current and len(current) + 2 + len(piece) > target_chars:
            chunks.append(current)
            # Carry the tail of the finished chunk as overlap context.
            tail = current[-overlap_chars:] if overlap_chars > 0 else ""
            current = (tail + "\n\n" + piece) if tail else piece
        else:
            current = (current + "\n\n" + piece) if current else piece

    if current:
        chunks.append(current)

    return [c for c in chunks if c.strip()]


# ════════════════════════════════════════════════════════════════
# Provided: embedding (sentence-transformers, no API key required)
# ════════════════════════════════════════════════════════════════

_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        print(f"Loading embedding model ({MODEL_NAME})... (one-time download ~470MB)")
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed(texts: list[str]) -> list[list[float]]:
    """Embed a list of strings. Returns unit-normalized 384-dim vectors."""
    model = get_model()
    vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return vectors.tolist()


# ════════════════════════════════════════════════════════════════
# Provided: build / save / load / search
# ════════════════════════════════════════════════════════════════

def read_pdf(path: Path) -> str:
    """Extract plain text from a PDF file, joining all pages."""
    reader = PdfReader(path)
    return "\n\n".join(
        page.extract_text() or "" for page in reader.pages
    )


def build_index() -> list[dict]:
    """Walk DOCS_DIR, chunk each file, embed, return list of records."""
    records: list[dict] = []
    chunk_id = 0
    for path in sorted(DOCS_DIR.glob("*")):
        if path.is_dir() or path.suffix.lower() not in (".md", ".txt", ".pdf"):
            continue
        if path.suffix.lower() == ".pdf":
            text = read_pdf(path)
        else:
            text = path.read_text(encoding='utf-8', errors='ignore')
        chunks = chunk_text(text)
        if not chunks:
            continue
        vectors = embed(chunks)
        for i, (chunk, vec) in enumerate(zip(chunks, vectors)):
            records.append({
                "chunk_id": chunk_id,
                "source": path.name,
                "chunk_index": i,
                "text": chunk,
                "embedding": vec,
            })
            chunk_id += 1
        print(f"  {path.name}: {len(chunks)} chunks")
    return records


def save_index(records: list[dict]) -> None:
    with INDEX_PATH.open("wb") as f:
        pickle.dump(records, f)


def load_index() -> list[dict]:
    if not INDEX_PATH.exists():
        raise FileNotFoundError(
            f"No index found at {INDEX_PATH}. Run `python indexer.py` from the project root first."
        )
    with INDEX_PATH.open("rb") as f:
        return pickle.load(f)


def cosine_distance(a: list[float], b: list[float]) -> float:
    # Both vectors are unit-normalized, so cosine distance == 1 - dot product.
    return 1.0 - sum(x * y for x, y in zip(a, b))


def search(query: str, records: list[dict], k: int = 5) -> list[dict]:
    """Embed the query, return top-k records by cosine distance."""
    [query_vec] = embed([query])
    scored = [(cosine_distance(r["embedding"], query_vec), r) for r in records]
    scored.sort(key=lambda x: x[0])
    return [r for _, r in scored[:k]]


def main() -> None:
    print(f"Indexing documents from {DOCS_DIR}/")
    records = build_index()
    save_index(records)
    print(f"\n✓ Indexed {len(records)} chunks → {INDEX_PATH.name}")


if __name__ == "__main__":
    main()