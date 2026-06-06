"""Tab 5 — AI Insights: Claude-powered natural-language analysis of MMM results."""

from __future__ import annotations

import streamlit as st
import anthropic
import json


_SYSTEM_PROMPT = """You are an expert marketing data scientist specialising in
Marketing Mix Modelling (MMM). You receive structured JSON summaries of a fitted
Bayesian MMM and answer questions concisely and precisely. Focus on actionable
budget recommendations, highlight uncertainty where relevant, and avoid jargon
unless the user asks for technical detail."""


def _build_context(mmm, idata) -> str:
    """Serialize the key model results into a compact JSON string for Claude."""
    try:
        contrib = mmm.channel_contribution_breakdown()
        contrib_dict = contrib.to_dict(orient="records")
    except Exception:
        contrib_dict = []

    ctx = {
        "channels": mmm.channel_cols,
        "target": mmm.target_col,
        "channel_contributions": contrib_dict,
    }
    return json.dumps(ctx, indent=2)


def render():
    st.header("AI Insights — Powered by Claude")
    st.caption("Ask natural-language questions about your MMM results.")

    api_key = st.text_input(
        "Anthropic API key",
        type="password",
        help="Your key is never stored — it lives only in this session.",
    )
    if not api_key:
        st.info("Enter your Anthropic API key above to enable AI-powered insights.")
        return

    has_model = "idata" in st.session_state and st.session_state["idata"] is not None
    if not has_model:
        st.warning("Fit a model first so Claude has real results to analyse.")

    preset_questions = [
        "Which channel has the best marginal ROAS right now?",
        "Where is budget being wasted (over-saturated channels)?",
        "What is the confidence level in the top-performing channel?",
        "Suggest a budget reallocation to improve total revenue by 10%.",
        "Explain the adstock effect detected in TV spend.",
    ]
    question = st.selectbox("Quick questions", ["(type your own…)"] + preset_questions)
    user_input = st.text_area(
        "Your question",
        value="" if question == "(type your own…)" else question,
        height=80,
    )

    if st.button("Ask Claude", type="primary") and user_input.strip():
        context = ""
        if has_model:
            context = f"\n\nModel results (JSON):\n{_build_context(st.session_state['mmm'], st.session_state['idata'])}"

        client = anthropic.Anthropic(api_key=api_key)

        with st.spinner("Claude is thinking…"):
            try:
                response = client.messages.create(
                    model="claude-opus-4-8",
                    max_tokens=1024,
                    system=_SYSTEM_PROMPT,
                    messages=[
                        {
                            "role": "user",
                            "content": user_input + context,
                        }
                    ],
                )
                answer = response.content[0].text
                st.session_state.setdefault("chat_history", []).append(
                    {"q": user_input, "a": answer}
                )
            except anthropic.AuthenticationError:
                st.error("Invalid API key. Please check and try again.")
                return
            except Exception as e:
                st.error(f"API call failed: {e}")
                return

    if "chat_history" in st.session_state:
        st.divider()
        for entry in reversed(st.session_state["chat_history"]):
            st.markdown(f"**You:** {entry['q']}")
            st.markdown(f"**Claude:** {entry['a']}")
            st.divider()
