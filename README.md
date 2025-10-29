\# 🧠 IA-Agent-Local



\*\*Assistant IA local\*\* combinant \*\*TinyLlama\*\*, \*\*Streamlit\*\*, \*\*mémoire persistante SQLite\*\*, \*\*RAG (retrieval-augmented generation)\*\* et \*\*recherche web\*\*.  

Ce projet démontre comment exécuter un agent conversationnel \*\*100 % local\*\*, sans API externe, capable de :

\- mémoriser les interactions et le profil utilisateur ;

\- interroger des PDF (RAG) via des embeddings MiniLM ;

\- compléter ses réponses avec des recherches DuckDuckGo.



---



\## 🚀 Fonctionnalités principales



\- 🧩 \*\*LLM local TinyLlama (1.1B)\*\* — rapide et CPU-friendly  

\- 💾 \*\*Mémoire persistante SQLite\*\* — profil, humeur, et souvenirs récents  

\- 📚 \*\*RAG (ChromaDB + SentenceTransformers)\*\* — ingestion et recherche dans les PDF  

\- 🌐 \*\*Recherche web DuckDuckGo\*\* — intégration optionnelle pour enrichir les réponses  

\- 💬 \*\*Interface Streamlit\*\* — simple et ergonomique  



---



\## 🛠️ Installation locale



\### 1️⃣ Cloner le dépôt

```bash

git clone https://github.com/Ank-co/IA-Agent-Local.git

cd IA-Agent-Local



