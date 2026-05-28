"""
Streamlit frontend for the Multilingual RAG Assistant.

This is now a THIN client: it does no RAG work itself. It just sends
HTTP requests to the FastAPI backend and displays the results.

Run with (in a separate terminal from the backend):
  streamlit run frontend/app.py

The backend must be running first:
  uvicorn backend.main:app --reload --port 8000
"""

import requests
import streamlit as st

# Where the backend lives. For local dev this is localhost:8000.
BACKEND_URL = "http://127.0.0.1:8000"

# Friendly names for detected language codes
LANG_NAMES = {
    "en": "English", "hi": "Hindi", "es": "Spanish", "fr": "French",
    "de": "German", "te": "Telugu", "ta": "Tamil", "bn": "Bengali",
    "ar": "Arabic", "zh-cn": "Chinese", "ja": "Japanese", "ru": "Russian",
    "pt": "Portuguese", "it": "Italian",
}

# ================= THE WEB PAGE =================

st.title("📚 Multilingual RAG Assistant")
st.write("Upload a PDF, then ask questions about it — in any language!")

# --- Check the backend is reachable ---
try:
    health = requests.get(f"{BACKEND_URL}/", timeout=5)
    backend_ok = health.status_code == 200
except Exception:
    backend_ok = False

if not backend_ok:
    st.error(
        "⚠️ Cannot reach the backend. Make sure it's running in another terminal:\n\n"
        "`uvicorn backend.main:app --reload --port 8000`"
    )
    st.stop()

# --- Upload section ---
uploaded = st.file_uploader("Upload a PDF", type="pdf")
if uploaded is not None:
    if st.button("Process this PDF"):
        with st.spinner("Sending document to the backend for processing..."):
            try:
                files = {"file": (uploaded.name, uploaded.getvalue(), "application/pdf")}
                resp = requests.post(f"{BACKEND_URL}/upload", files=files, timeout=120)
                resp.raise_for_status()
                data = resp.json()
                st.success(
                    f"Done! Stored {data['chunks_stored']} chunks from "
                    f"{data['filename']}. You can now ask questions below."
                )
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

            except requests.exceptions.HTTPError:
                detail = ""
                try:
                    detail = resp.json().get("detail", "")
                except Exception:
                    pass
                if "RESOURCE_EXHAUSTED" in str(detail) or "429" in str(detail):
                    st.error("⏳ We've hit the API rate limit. Please wait a minute and try again.")
                else:
                    st.error("📄 Could not answer. Have you uploaded and processed a PDF yet?")
            except Exception as e:
                st.error(f"Something went wrong: {e}")

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

