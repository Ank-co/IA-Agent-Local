import os
import streamlit as st
from dotenv import load_dotenv
from memory_sqlite import (
    save_profile, load_profile, remember_mood, remember_text, retrieve_context
)
from rag import ingest_pdfs, retrieve
from web_search import web_search, format_search_snippets
from llm import answer_with_context

# charge les variables d'environnement (.env)
load_dotenv()

st.set_page_config(page_title="Agent IA Antoine", layout="wide")

# Sidebar
st.sidebar.header("⚙️ Paramètres")
user_id = st.sidebar.text_input("User ID", value="antoine")

if st.sidebar.button("Créer / Mettre à jour le profil"):
    profile = load_profile(user_id)
    if not profile:
        profile = {"langue": "fr", "style": "paragraphes"}
    save_profile(user_id, profile)
    st.sidebar.success("Profil enregistré ✅")

st.sidebar.header("🧮 Ingestion de PDF")
if st.sidebar.button("Ingestion des PDF (./data) → Base Chroma"):
    n = ingest_pdfs()
    st.sidebar.success(f"Ingestion terminée : {n} chunks ajoutés.")

st.sidebar.write("Ou dépose un PDF :")
uploaded = st.sidebar.file_uploader("Ajouter des PDF", type=["pdf"], accept_multiple_files=True)
if uploaded:
    os.makedirs("data", exist_ok=True)
    for f in uploaded:
        with open(os.path.join("data", f.name), "wb") as out:
            out.write(f.getbuffer())
    st.sidebar.success(f"{len(uploaded)} fichier(s) ajouté(s) dans ./data ✅")

st.sidebar.header("💭 Humeur du jour")
c1, c2 = st.sidebar.columns(2)
with c1:
    if st.button("Je me sens bien"):
        remember_mood(user_id, "bien")
        st.success("Humeur enregistrée : bien ✅")
with c2:
    if st.button("Je suis fatigué"):
        remember_mood(user_id, "fatigué")
        st.success("Humeur enregistrée : fatigué ✅")

st.sidebar.header("🌐 Web")
use_web = st.sidebar.checkbox("Autoriser la recherche web", value=False)

#  Principale
st.title("🧬 Agent IA — Mémoire + RAG + Web")

if "history" not in st.session_state:
    st.session_state.history = []

user_msg = st.text_area("Message", placeholder="Pose ta question ici…", height=120)

if st.button("Envoyer") and user_msg.strip():
    # log user
    st.session_state.history.append({"role": "user", "content": user_msg})

    # contexte (mémoire)
    ctx = retrieve_context(user_id)

    # RAG
    docs = retrieve(user_msg, k=4)

    # recherche web
    web_snips = ""
    if use_web:
        try:
            results = web_search(user_msg, max_results=3)
            web_snips = format_search_snippets(results)
        except Exception as e:
            web_snips = f"(Erreur web : {e})"

    # réponse LLM
    answer = answer_with_context(
        user_msg, context=ctx, docs=docs, web=web_snips
    )
    st.session_state.history.append({"role": "assistant", "content": answer})

    # mémoire libre (trace courte)
    remember_text(user_id, f"Q: {user_msg} | A: {answer[:300]}…")

# affichage historique
for msg in st.session_state.history:
    if msg["role"] == "user":
        st.markdown(f"**👤 Toi :** {msg['content']}")
    else:
        st.markdown(f"**🤖 IA :** {msg['content']}")
