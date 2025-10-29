import os
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch


# Configuration modèle
MODEL_ID = os.getenv("LOCAL_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
HF_HOME = os.getenv("HF_HOME")  # ex: "E:\\IA MSX.1\\hf_cache" ou clé USB
OFFLINE = os.getenv("TRANSFORMERS_OFFLINE", "0") == "1"

# Réduit la charge CPU
try:
    torch.set_num_threads(min(6, torch.get_num_threads()))
except Exception:
    pass

torch.set_grad_enabled(False)


# Chargement modèle/tokenizer
# important : on force cache_dir + offline/local
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    cache_dir=HF_HOME,
    local_files_only=OFFLINE or os.path.isabs(MODEL_ID),
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    torch_dtype=torch.float32,   # CPU
    cache_dir=HF_HOME,
    local_files_only=OFFLINE or os.path.isabs(MODEL_ID),
)


# Réglages rapides
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "120"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.4"))

_gen = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=-1,                 # force CPU
    max_new_tokens=MAX_NEW_TOKENS,
    temperature=TEMPERATURE,
    do_sample=False,           # greedy = plus rapide/stable
    top_p=0.95,
)


# Prompt système
SYSTEM_BASE = (
    "Tu es un assistant curieux, tu veux apprendre pour conseiller à l'avenir. "
    "Réponds en français en 2 à 4 phrases maximum. "
    "Quand des documents/contexte sont fournis, appuie-toi dessus."
)


# Fonctions principales
def _build_prompt(user_msg: str, context: str, docs: list[str], web: str) -> str:
    parts = [f"<|system|>\n{SYSTEM_BASE}"]
    ctx_parts = []
    if context:
        ctx_parts.append(f"[MEMOIRE]\n{context}")
    if docs:
        ctx_parts.append(f"[DOCUMENTS]\n" + "\n---\n".join(docs))
    if web:
        ctx_parts.append(f"[WEB]\n{web}")
    if ctx_parts:
        parts.append("<|context|>\n" + "\n\n".join(ctx_parts))
    parts.append(f"<|user|>\n{user_msg}\n<|assistant|>")
    return "\n\n".join(parts)


def answer_with_context(user_msg: str, context: str = "", docs: list[str] = None, web: str = "") -> str:
    docs = docs or []
    prompt = _build_prompt(user_msg, context, docs, web)
    out = _gen(prompt)[0]["generated_text"]
    if "<|assistant|>" in out:
        out = out.split("<|assistant|>", 1)[1]
    # raccourcit au cas où
    return out.strip().split("\n\n")[0].strip()
