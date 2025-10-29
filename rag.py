import os
from typing import List
from pypdf import PdfReader
import chromadb
from sentence_transformers import SentenceTransformer

# ----------------------------
# Chemins & config (minimal)
# ----------------------------
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.getenv("DATA_DIR", os.path.join(BASE_DIR, "data"))
VSTORE_DIR = os.getenv("VSTORE_DIR", os.path.join(BASE_DIR, "vectorstore"))
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(VSTORE_DIR, exist_ok=True)

# Embeddings 100% locaux (gère diff. langues, léger CPU)
EMB_MODEL = os.getenv("EMB_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
HF_HOME = os.getenv("HF_HOME")  # ex: "E:\\IA MSX.1\\hf_cache" ou ta clé USB
OFFLINE = os.getenv("TRANSFORMERS_OFFLINE", "0") == "1"

# ⚠️ clé : utiliser cache_folder (et non cache_dir)
_embedder = SentenceTransformer(
    EMB_MODEL,
    cache_folder=HF_HOME,
    local_files_only=OFFLINE or os.path.isabs(EMB_MODEL),
)

def _chunk_text(text: str, max_len: int = 900) -> list[str]:
    words = text.split()
    chunk, chunks = [], []
    cur_len = 0
    for w in words:
        wl = len(w) + 1
        if cur_len + wl > max_len and chunk:
            chunks.append(" ".join(chunk))
            chunk, cur_len = [w], wl
        else:
            chunk.append(w)
            cur_len += wl
    if chunk:
        chunks.append(" ".join(chunk))
    return chunks

# Chroma persistant sur disque
_client = chromadb.PersistentClient(path=VSTORE_DIR)
_collection = _client.get_or_create_collection(name="knowledge_base_v1")

def _embed(texts: list[str]) -> list[list[float]]:
    vecs = _embedder.encode(
        texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True
    )
    return [v.tolist() for v in vecs]

def ingest_pdfs(folder: str = DATA_DIR) -> int:
    docs, ids = [], []
    for fn in os.listdir(folder):
        if not fn.lower().endswith(".pdf"):
            continue
        path = os.path.join(folder, fn)
        try:
            reader = PdfReader(path)
        except Exception:
            continue
        full = []
        for p in reader.pages:
            try:
                full.append(p.extract_text() or "")
            except Exception:
                pass
        text = "\n".join(full).strip()
        if not text:
            continue
        for i, c in enumerate(_chunk_text(text)):
            docs.append(c)
            ids.append(f"{fn}:{i}")
    if not docs:
        return 0
    vectors = _embed(docs)
    _collection.add(documents=docs, embeddings=vectors, ids=ids)
    try:
        _client.persist()
    except Exception:
        pass
    return len(docs)

def retrieve(query: str, k: int = 4) -> List[str]:
    qvec = _embed([query])[0]
    res = _collection.query(query_embeddings=[qvec], n_results=k)
    return res.get("documents", [[]])[0]
