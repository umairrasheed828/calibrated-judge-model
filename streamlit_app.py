"""Streamlit demo UI for the calibrated judge -- a thin client over /judge.

Run the API first:  uv run uvicorn src.api:app --port 8000
Then in another terminal:  uv run streamlit run streamlit_app.py
"""

import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="Calibrated Research-QA Judge", page_icon="⚖️")
st.title("⚖️ Calibrated Research-QA Judge")
st.caption(
    "Scores an answer on faithfulness + relevance, with a calibrated confidence."
)

question = st.text_input(
    "Question", placeholder="What is retrieval-augmented generation?"
)
context = st.text_area("Context (the source of truth)", height=140)
answer = st.text_area("Answer to judge", height=140)

if st.button("Judge", type="primary") and question.strip() and answer.strip():
    with st.spinner("Judging…"):
        try:
            resp = requests.post(
                f"{API_URL}/judge",
                json={"question": question, "context": context, "answer": answer},
                timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()
            c1, c2, c3 = st.columns(3)
            c1.metric("Faithfulness", f"{data['faithfulness']} / 5")
            c2.metric("Relevance", f"{data['relevance']} / 5")
            c3.metric(
                "Faithful? (calibrated)", f"{data['faithfulness_confidence']:.0%}"
            )
            st.caption("Confidence is temperature-calibrated (T = 1.25, from Step 14).")
        except requests.exceptions.RequestException as e:
            st.error(f"Request failed: {e}. Is the API running at {API_URL}?")
