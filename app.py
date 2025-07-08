import streamlit as st
import json
import pandas as pd
from io import StringIO
from core import (
    PromptEnhancer,
    get_ollama_models,
    call_ollama,
    choose_enhancement_strategy,
    ENHANCEMENT_PATTERNS,
)


def main():
    st.set_page_config(
        page_title="Advanced Prompt Enhancer AI Agent (Ollama Edition)",
        page_icon="🚀",
        layout="wide",
    )

    st.title("🚀 Advanced Prompt Enhancer AI Agent (Ollama Edition)")
    st.markdown(
        "Let an AI agent automatically analyze, enhance, and execute your prompts using local Ollama models."
    )

    # --- INITIALIZE SESSION STATE ---
    if "enhancer" not in st.session_state:
        st.session_state.enhancer = PromptEnhancer()
    if "enhanced_prompt" not in st.session_state:
        st.session_state.enhanced_prompt = ""
    if "llm_response" not in st.session_state:
        st.session_state.llm_response = ""
    if "critique" not in st.session_state:
        st.session_state.critique = ""

    # --- SIDEBAR ---
    with st.sidebar:
        st.header("⚙️ Configuration")

        available_models = get_ollama_models()
        if available_models:
            selected_model = st.selectbox(
                "Select an Ollama Model:",
                available_models,
                index=0,
                key="ollama_model",
                help="Models detected from your local Ollama instance.",
            )
        else:
            st.warning("Could not connect to Ollama. Please ensure it's running.")
            selected_model = st.text_input(
                "Ollama Model Name",
                value="llama3",
                key="ollama_model",
                help="Could not detect models. Please enter one manually.",
            )

        st.info(f"Using model: **{selected_model}**")

    # --- MAIN CONTENT ---
    col1, col2 = st.columns([1, 1])

    with col1:
        st.header("📝 Input Prompt")
        user_prompt = st.text_area(
            "Enter your initial prompt:",
            height=250,
            placeholder="Enter the prompt you want the AI agent to enhance...",
            key="user_prompt_input"
        )

        enhance_button = st.button(
            "✨ Automatically Enhance Prompt",
            type="primary",
            use_container_width=True,
        )

    with col2:
        st.header("🎯 Enhanced Output")

        if enhance_button:
            if not user_prompt:
                st.warning("⚠️ Please enter a prompt to enhance.")
            else:
                st.session_state.llm_response = ""  # Clear previous response
                st.session_state.critique = "" # Clear previous critique

                with st.spinner("🤖 AI agent is thinking... Choosing enhancement strategy..."):
                    enhancement_type = choose_enhancement_strategy(user_prompt, selected_model)
                
                st.info(f"**Chosen Strategy:** {ENHANCEMENT_PATTERNS[enhancement_type]['name']}")
                st.markdown(f"_{ENHANCEMENT_PATTERNS[enhancement_type]['description']}_")

                with st.spinner("🧠 Enhancing prompt..."):
                    enhanced_prompt = st.session_state.enhancer.enhance_prompt(
                        user_prompt, enhancement_type, model=selected_model
                    )
                    st.session_state.enhanced_prompt = enhanced_prompt

        if st.session_state.critique:
            with st.expander("🔍 View AI Critique", expanded=False):
                st.markdown(st.session_state.critique)

        if st.session_state.enhanced_prompt:
            st.subheader("✨ Enhanced Prompt")
            st.code(st.session_state.enhanced_prompt, language="markdown")
            
            generate_button = st.button("🚀 Generate Response", use_container_width=True)
            if generate_button:
                with st.spinner(f"🚀 Generating response from **{selected_model}**..."):
                    llm_response = call_ollama(st.session_state.enhanced_prompt, selected_model)
                    st.session_state.llm_response = llm_response

        if st.session_state.llm_response:
            st.subheader("💬 AI Response")
            st.markdown(st.session_state.llm_response, unsafe_allow_html=True)


if __name__ == "__main__":
    main()