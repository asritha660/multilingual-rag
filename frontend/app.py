"""
Streamlit frontend for the Multilingual RAG Assistant.

Thin client that talks to the FastAPI backend over HTTP.
Now includes login/register; the token is sent on protected requests.

Run with (backend must be running too):
  streamlit run frontend/app.py
"""

import requests
import streamlit as st

BACKEND_URL = "http://127.0.0.1:8000"

LANG_NAMES = {
    "en": "English", "hi": "Hindi", "es": "Spanish", "fr": "French",
    "de": "German", "te": "Telugu", "ta": "Tamil", "bn": "Bengali",
    "ar": "Arabic", "zh-cn": "Chinese", "ja": "Japanese", "ru": "Russian",
    "pt": "Portuguese", "it": "Italian",
}

st.title("📚 Multilingual RAG Assistant")

# --- Backend reachability check ---
try:
    health = requests.get(f"{BACKEND_URL}/", timeout=5)
    backend_ok = health.status_code == 200
except Exception:
    backend_ok = False

if not backend_ok:
    st.error(
        "⚠️ Cannot reach the backend. Start it in another terminal:\n\n"
        "`uvicorn backend.main:app --reload --port 8000`"
    )
    st.stop()

# --- Session state for the auth token ---
if "token" not in st.session_state:
    st.session_state.token = None
if "username" not in st.session_state:
    st.session_state.username = None


def auth_headers():
    """Return the Authorization header with the stored JWT."""
    return {"Authorization": f"Bearer {st.session_state.token}"}


# ============================================================
# LOGGED OUT VIEW: show login / register
# ============================================================
if st.session_state.token is None:
    st.write("Please log in or create an account to use the assistant.")

    tab_login, tab_register = st.tabs(["Log in", "Register"])

    with tab_login:
        login_user = st.text_input("Username", key="login_user")
        login_pass = st.text_input("Password", type="password", key="login_pass")
        if st.button("Log in"):
            try:
                resp = requests.post(
                    f"{BACKEND_URL}/login",
                    data={"username": login_user, "password": login_pass},
                    timeout=10,
                )
                if resp.status_code == 200:
                    st.session_state.token = resp.json()["access_token"]
                    st.session_state.username = login_user
                    st.rerun()
                else:
                    st.error("Invalid username or password.")
            except Exception as e:
                st.error(f"Login failed: {e}")

    with tab_register:
        reg_user = st.text_input("Choose a username", key="reg_user")
        reg_pass = st.text_input("Choose a password", type="password", key="reg_pass")
        if st.button("Register"):
            try:
                resp = requests.post(
                    f"{BACKEND_URL}/register",
                    json={"username": reg_user, "password": reg_pass},
                    timeout=10,
                )
                if resp.status_code == 200:
                    st.success("Account created! Switch to the Log in tab.")
                else:
                    detail = resp.json().get("detail", "Registration failed")
                    st.error(detail)
            except Exception as e:
                st.error(f"Registration failed: {e}")

    st.stop()  # don't show the rest of the app until logged in


# ============================================================
# LOGGED IN VIEW: the actual app
# ============================================================
col1, col2 = st.columns([3, 1])
with col1:
    st.write(f"Logged in as **{st.session_state.username}**. Upload a PDF and ask away — in any language!")
with col2:
    if st.button("Log out"):
        st.session_state.token = None
        st.session_state.username = None
        st.rerun()

# --- Upload section ---
uploaded = st.file_uploader("Upload a PDF", type="pdf")
if uploaded is not None:
    if st.button("Process this PDF"):
        with st.spinner("Sending document to the backend..."):
            try:
                files = {"file": (uploaded.name, uploaded.getvalue(), "application/pdf")}
                resp = requests.post(
                    f"{BACKEND_URL}/upload",
                    files=files,
                    headers=auth_headers(),
                    timeout=120,
                )
                resp.raise_for_status()
                data = resp.json()
                st.success(f"Done! Stored {data['chunks_stored']} chunks from {data['filename']}.")
            except Exception as e:
                st.error(f"Upload failed: {e}")

st.divider()

# --- Question section ---
question = st.text_input("Your question:", placeholder="e.g. What is this document about?")
if st.button("Ask"):
    if not question:
        st.warning("Please type a question first.")
    else:
        with st.spinner("Searching and thinking..."):
            try:
                resp = requests.post(
                    f"{BACKEND_URL}/ask",
                    json={"question": question},
                    headers=auth_headers(),
                    timeout=120,
                )
                resp.raise_for_status()
                data = resp.json()

                lang_code = data["detected_language"]
                lang_display = LANG_NAMES.get(lang_code, lang_code.upper())
                st.caption(f"🌐 Detected language: **{lang_display}** (`{lang_code}`)")

                st.subheader("Answer")
                st.write(data["answer"])

                with st.expander("📄 See the sources used"):
                    for i, ch in enumerate(data["sources"]):
                        st.markdown(f"**Source {i+1}:** {ch}")
            except Exception as e:
                st.error(f"Could not get an answer: {e}")

# --- Query History section ---
st.divider()
st.subheader("📊 Query History")
if st.button("Load recent queries"):
    try:
        resp = requests.get(f"{BACKEND_URL}/history", timeout=10)
        resp.raise_for_status()
        rows = resp.json()["queries"]
        if not rows:
            st.info("No queries logged yet. Ask a question above!")
        else:
            for r in rows:
                st.markdown(
                    f"**Q:** {r['user_query']}  \n"
                    f"🌐 {r['detected_language']} · "
                    f"⏱️ {r['response_time_ms']} ms · "
                    f"📄 {r['retrieved_chunks']} chunks"
                )
    except Exception as e:
        st.error(f"Could not load history: {e}")
