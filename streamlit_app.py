import html
import re
from typing import List, Dict

import streamlit as st
from openai import OpenAI
from qdrant_client import QdrantClient

# -----------------------------
# Secrets / Clients
# -----------------------------
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
OPENAI_MODEL = st.secrets.get("OPENAI_MODEL", "gpt-4.1-mini")
OPENAI_EMBED_MODEL = st.secrets.get("OPENAI_EMBED_MODEL", "text-embedding-3-small")

QDRANT_URL = st.secrets["QDRANT_URL"]
QDRANT_API_KEY = st.secrets["QDRANT_API_KEY"]
QDRANT_COLLECTION = st.secrets.get("QDRANT_COLLECTION", "nrel_docs")

openai_client = OpenAI(api_key=OPENAI_API_KEY)
qdrant_client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
)

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="AI Knowledge Agent",
    page_icon="⚡",
    layout="wide",
)

# -----------------------------
# Session state
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "question_history" not in st.session_state:
    st.session_state.question_history = []
if "selected_question" not in st.session_state:
    st.session_state.selected_question = ""
if "last_response" not in st.session_state:
    st.session_state.last_response = None

# -----------------------------
# Styling
# -----------------------------
st.markdown(
    """
    <style>
    .hero {
        padding: 1.2rem 1.4rem;
        border-radius: 18px;
        background: linear-gradient(135deg, #0f766e 0%, #1e3a8a 100%);
        color: white;
        margin-bottom: 1rem;
    }
    .hero h1 { margin: 0; font-size: 1.9rem; font-weight: 700; }
    .hero p { margin: 0.4rem 0 0 0; font-size: 0.98rem; opacity: 0.96; }

    .pill {
        display: inline-block;
        padding: 0.28rem 0.68rem;
        border-radius: 999px;
        background: rgba(255,255,255,0.14);
        margin-right: 0.4rem;
        margin-top: 0.75rem;
        font-size: 0.82rem;
    }

    .disclaimer {
        background: #fff7ed;
        border: 1px solid #fed7aa;
        color: #9a3412;
        border-radius: 12px;
        padding: 0.85rem 1rem;
        margin-bottom: 1rem;
        font-size: 0.92rem;
    }

    .citation-card {
        background: #f8fafc;
        border: 1px solid #e5e7eb;
        border-radius: 14px;
        padding: 0.8rem 0.9rem;
        margin-bottom: 0.7rem;
    }

    .recent-chip {
        display: inline-block;
        padding: 0.35rem 0.65rem;
        border-radius: 999px;
        background: #eef2ff;
        color: #3730a3;
        margin: 0.2rem 0.25rem 0.2rem 0;
        font-size: 0.84rem;
    }

    .answer-heading {
        background-color: #dcfce7;
        color: #166534;
        padding: 0.18rem 0.45rem;
        border-radius: 0.4rem;
        font-weight: 700;
    }

    .explanation-heading {
        background-color: #dbeafe;
        color: #1d4ed8;
        padding: 0.18rem 0.45rem;
        border-radius: 0.4rem;
        font-weight: 700;
    }

    .recommendation-heading {
        background-color: #fef3c7;
        color: #92400e;
        padding: 0.18rem 0.45rem;
        border-radius: 0.4rem;
        font-weight: 700;
    }

    .citation-heading {
        background-color: #ede9fe;
        color: #6d28d9;
        padding: 0.18rem 0.45rem;
        border-radius: 0.4rem;
        font-weight: 700;
    }

    mark {
        background-color: #fde68a;
        padding: 0.08rem 0.22rem;
        border-radius: 0.3rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# UI helper rendering
# -----------------------------
def render_answer_with_highlights(answer: str) -> str:
    safe = html.escape(answer)
    safe = re.sub(r"(?m)^Answer:", '<span class="answer-heading">Answer</span>', safe)
    safe = re.sub(r"(?m)^Explanation:", '<span class="explanation-heading">Explanation</span>', safe)
    safe = re.sub(r"(?m)^Recommendations:", '<span class="recommendation-heading">Recommendations</span>', safe)
    safe = re.sub(r"(?m)^Citations:", '<span class="citation-heading">Citations</span>', safe)
    safe = safe.replace("\n", "<br>")
    return f'<div style="line-height:1.85; font-size:1rem; color:#0f172a;">{safe}</div>'


def render_html_block(text: str) -> str:
    return f'<div style="line-height:1.75; font-size:0.98rem; color:#0f172a;">{text}</div>'


# -----------------------------
# Core helper functions
# -----------------------------
@st.cache_data(show_spinner=False, ttl=3600)
def embed_texts(texts: List[str]) -> List[List[float]]:
    resp = openai_client.embeddings.create(
        model=OPENAI_EMBED_MODEL,
        input=texts,
    )
    return [item.embedding for item in resp.data]


def cosine_similarity(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def highlight_relevant_sentences(question: str, text: str) -> str:
    sentences = split_sentences(text)
    if not sentences:
        return render_html_block(html.escape(text).replace("\n", "<br>"))

    q_embedding = embed_texts([question])[0]
    sentence_embeddings = embed_texts(sentences)

    scored = []
    for sent, emb in zip(sentences, sentence_embeddings):
        score = cosine_similarity(q_embedding, emb)
        scored.append((sent, score))

    top_sentences = {
        s for s, _ in sorted(scored, key=lambda x: x[1], reverse=True)[:2]
    }

    rendered = []
    for sent in sentences:
        safe = html.escape(sent)
        if sent in top_sentences:
            rendered.append(f"<mark>{safe}</mark>")
        else:
            rendered.append(safe)

    return render_html_block(" ".join(rendered))


def search_documents(question: str, top_k: int = 4) -> List[Dict]:
    query_vector = embed_texts([question])[0]

    response = qdrant_client.query_points(
        collection_name=QDRANT_COLLECTION,
        query=query_vector,
        limit=top_k,
        with_payload=True,
    )

    matches = []
    for point in response.points:
        payload = point.payload or {}
        text = payload.get("text", "")
        matches.append({
            "score": point.score,
            "document": payload.get("title", payload.get("document", "Unknown Document")),
            "source_path": payload.get("source_path", ""),
            "section": payload.get("section", "Unknown Section"),
            "text": text,
            "highlighted_text": highlight_relevant_sentences(question, text),
        })

    return matches


def generate_answer(question: str, matches: List[Dict], chat_history: List[Dict] | None = None) -> str:
    context_parts = []
    for i, match in enumerate(matches, start=1):
        context_parts.append(
            f"""Source {i}
Document: {match['document']}
Section: {match['section']}

Content:
{match['text']}
"""
        )

    history_text = ""
    if chat_history:
        history_text = "\n".join(
            f"{turn['role'].capitalize()}: {turn['content']}"
            for turn in chat_history[-6:]
        )

    prompt = f"""
You are a Renewable Energy Expert based on public research documents.

Use the conversation history for context when interpreting follow-up questions.
Use ONLY the provided sources to answer.

Return the answer in the following format:

Answer:
<direct answer>

Explanation:
<explain the concept in simple terms>

Recommendations:
<provide 3-5 practical, actionable recommendations based on the documents>

Citations:
- Document | Section

If information is not found in the sources, say you don't know.

Conversation History:
{history_text if history_text else "No prior conversation."}

Current Question:
{question}

Sources:
{chr(10).join(context_parts)}
"""

    response = openai_client.responses.create(
        model=OPENAI_MODEL,
        input=prompt,
    )
    return response.output_text


def build_dynamic_suggestions(question: str) -> List[str]:
    q = question.lower().strip()

    if "grid" in q or "reliability" in q:
        return [
            "What are the biggest grid reliability risks?",
            "What recommendations are given for grid resilience?",
            "Which report discusses reliability most directly?",
        ]
    if "renewable" in q or "integration" in q:
        return [
            "What lessons are highlighted from renewable integration?",
            "What recommendations are given for renewable energy adoption?",
            "Can you summarize this in simpler terms?",
        ]
    if "cost" in q or "pricing" in q:
        return [
            "What cost drivers are discussed?",
            "Which report talks most directly about cost trends?",
            "What recommendations are given to reduce costs?",
        ]

    return [
        "Can you summarize this in simple terms?",
        "What are the main recommendations?",
        "Which report supports this most directly?",
    ]


# -----------------------------
# Static starter questions
# -----------------------------
starter_questions = [
    "What are key challenges in grid reliability?",
    "What lessons are highlighted from renewable integration?",
    "What recommendations are given for renewable energy adoption?",
    "Summarize the main findings across these reports.",
    "Which report discusses reliability most directly?",
]

# -----------------------------
# Header
# -----------------------------
st.markdown(
    """
    <div class="hero">
        <h1>⚡ AI Knowledge Agent</h1>
        <p>Chat with indexed PDF knowledge using grounded answers, recommendations, citations, and highlighted evidence.</p>
        <span class="pill">Streamlit Cloud</span>
        <span class="pill">Qdrant Cloud</span>
        <span class="pill">OpenAI</span>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="disclaimer">
        <strong>Disclaimer:</strong> This AI agent can make mistakes. Please verify important details against the source documents before relying on the answer.
    </div>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    top_k = st.slider("Chunks to retrieve", 1, 10, 4)
    show_evidence = st.toggle("Show source evidence", value=True)
    show_citations = st.toggle("Show citations", value=True)

    st.divider()
    st.subheader("Starter Questions")
    for i, sq in enumerate(starter_questions):
        if st.button(sq, key=f"starter_{i}", use_container_width=True):
            st.session_state.selected_question = sq

# -----------------------------
# Chat input
# -----------------------------
question = st.chat_input("Ask the agent about your reports...")

if st.session_state.selected_question and not question:
    question = st.session_state.selected_question
    st.session_state.selected_question = ""

if question:
    st.session_state.messages.append({"role": "user", "content": question})

    with st.spinner("Thinking..."):
        chat_history = [
            {"role": msg["role"], "content": msg["content"]}
            for msg in st.session_state.messages[:-1]
        ]

        # Better retrieval for follow-up questions
        recent_user_turns = [
            turn["content"] for turn in chat_history[-4:] if turn["role"] == "user"
        ]
        retrieval_query = " ".join(recent_user_turns + [question]).strip()

        matches = search_documents(retrieval_query, top_k=top_k)
        answer = generate_answer(question, matches, chat_history=chat_history)

        st.session_state.last_response = {
            "question": question,
            "answer": answer,
            "matches": matches,
            "citations": [
                {
                    "document": m["document"],
                    "section": m["section"],
                    "source_path": m["source_path"],
                }
                for m in matches
            ],
        }

        st.session_state.question_history.insert(0, question)
        st.session_state.messages.append({"role": "assistant", "content": answer})

# -----------------------------
# Chat rendering
# -----------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            st.markdown(render_answer_with_highlights(msg["content"]), unsafe_allow_html=True)
        else:
            st.write(msg["content"])

# -----------------------------
# Post-answer panels
# -----------------------------
data = st.session_state.last_response

if data:
    st.divider()

    # dynamic suggestions
    suggestions = build_dynamic_suggestions(data["question"])
    st.subheader("Suggested follow-up questions")
    cols = st.columns(3)
    for i, s in enumerate(suggestions):
        with cols[i]:
            if st.button(s, key=f"sugg_{i}", use_container_width=True):
                st.session_state.selected_question = s
                st.rerun()

    # citations
    if show_citations:
        st.subheader("Citations")
        seen = set()
        for c in data["citations"]:
            key = (c["document"], c["section"], c["source_path"])
            if key in seen:
                continue
            seen.add(key)

            st.markdown(
                f"""
                <div class="citation-card">
                    <div><strong>{html.escape(c['document'])}</strong></div>
                    <div style="margin-top:0.25rem;">Section: {html.escape(str(c['section']))}</div>
                    <div style="color:#94a3b8; font-size:0.75rem; margin-top:0.35rem;">{html.escape(c['source_path'])}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # evidence
    if show_evidence:
        st.subheader("Evidence")
        for i, m in enumerate(data["matches"], start=1):
            with st.expander(f"Match {i} · {m['document']} · {m['section']} · Score {m['score']:.4f}"):
                st.caption(m["source_path"])
                st.markdown(m["highlighted_text"], unsafe_allow_html=True)

# -----------------------------
# Recent questions
# -----------------------------
if st.session_state.question_history:
    st.divider()
    st.subheader("Recent Questions")
    chips = "".join(
        [f'<span class="recent-chip">{html.escape(q)}</span>' for q in st.session_state.question_history[:6]]
    )
    st.markdown(chips, unsafe_allow_html=True)
