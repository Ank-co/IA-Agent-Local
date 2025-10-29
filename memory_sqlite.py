# memory.py — Gestion de la mémoire persistante (profil, humeurs, faits, souvenirs)

import os
import json
import sqlite3
from datetime import date, datetime
from typing import Dict, Any


# Configuration
DB_PATH = os.path.join(os.path.dirname(__file__), "brain.db")

SCHEMA = [
    """
    CREATE TABLE IF NOT EXISTS users (
        user_id TEXT PRIMARY KEY,
        profile_json TEXT
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS mood_journal (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT,
        d TEXT,
        mood TEXT,
        note TEXT
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS facts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT,
        k TEXT,
        v TEXT,
        ts TEXT
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS memories (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id TEXT,
        ts TEXT,
        text TEXT
    );
    """
]

INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_mood_user ON mood_journal(user_id, id DESC);",
    "CREATE INDEX IF NOT EXISTS idx_facts_user ON facts(user_id, ts DESC);",
    "CREATE INDEX IF NOT EXISTS idx_mem_user ON memories(user_id, id DESC);"
]



# Fonctions utilitaires
def _conn() -> sqlite3.Connection:
    """Crée une connexion SQLite avec mode WAL pour limiter les corruptions."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn

def init_db() -> None:
    """Initialise la base de données si elle n'existe pas (tables + index)."""
    with _conn() as conn:
        cur = conn.cursor()
        for stmt in SCHEMA:
            cur.execute(stmt)
        for stmt in INDEXES:
            cur.execute(stmt)


# Gestion de l'utilisateur
def save_profile(user_id: str, profile: Dict[str, Any]) -> None:
    """Sauvegarde ou met à jour le profil utilisateur."""
    init_db()
    with _conn() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO users(user_id, profile_json) VALUES (?, ?)",
            (user_id, json.dumps(profile, ensure_ascii=False)),
        )

def load_profile(user_id: str) -> Dict[str, Any]:
    """Charge le profil utilisateur depuis la base (ou retourne {})."""
    init_db()
    with _conn() as conn:
        cur = conn.execute("SELECT profile_json FROM users WHERE user_id=?", (user_id,))
        row = cur.fetchone()
    if not row or not row[0]:
        return {}
    try:
        return json.loads(row[0])
    except json.JSONDecodeError:
        return {}


# Gestion des émotions / faits / souvenirs
def remember_mood(user_id: str, mood: str, note: str = "") -> None:
    """Enregistre une humeur quotidienne."""
    init_db()
    with _conn() as conn:
        conn.execute(
            "INSERT INTO mood_journal(user_id, d, mood, note) VALUES (?, ?, ?, ?)",
            (user_id, date.today().isoformat(), mood, note),
        )

def remember_fact(user_id: str, key: str, value: str) -> None:
    """Sauvegarde un fait clé/valeur lié à l’utilisateur."""
    init_db()
    ts = datetime.utcnow().isoformat()
    with _conn() as conn:
        conn.execute(
            "INSERT INTO facts(user_id, k, v, ts) VALUES (?, ?, ?, ?)",
            (user_id, key, value, ts),
        )

def remember_text(user_id: str, text: str) -> None:
    """Enregistre un texte libre (souvenir, conversation, réflexion)."""
    init_db()
    ts = datetime.utcnow().isoformat()
    with _conn() as conn:
        conn.execute(
            "INSERT INTO memories(user_id, ts, text) VALUES (?, ?, ?)",
            (user_id, ts, text),
        )


# Récupération du contexte
def retrieve_context(user_id: str) -> str:
    """
    Reconstitue un contexte textuel combinant :
    - le profil utilisateur
    - la dernière humeur
    - les 5 derniers souvenirs
    """
    init_db()
    with _conn() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT d, mood, note FROM mood_journal WHERE user_id=? ORDER BY id DESC LIMIT 1",
            (user_id,),
        )
        mood = cur.fetchone()

        profile = load_profile(user_id)

        cur.execute(
            "SELECT ts, text FROM memories WHERE user_id=? ORDER BY id DESC LIMIT 5",
            (user_id,),
        )
        memories = cur.fetchall()

    parts = []
    if profile:
        parts.append("PROFILE: " + json.dumps(profile, ensure_ascii=False))
    if mood:
        d, m, note = mood
        parts.append(f"MOOD: {d} — {m} — {note}")
    if memories:
        parts.append(
            "RECENT_MEMORIES:\n" + "\n".join([f"- {ts}: {txt}" for ts, txt in memories])
        )

    return "\n\n".join(parts)
