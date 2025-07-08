import re
import json
import requests
from datetime import datetime
from typing import List, Dict
import streamlit as st

# Enhanced prompt engineering patterns based on Anthropic's best practices
ENHANCEMENT_PATTERNS = {
    "ai_rewrite": {
        "name": "AI-Driven Rewrite",
        "description": "Uses an AI to critique and rewrite the prompt from scratch for maximum effectiveness.",
        "template": None,  # This strategy does not use a simple template
    },
    "xml_structure": {
        "name": "XML Structure Enhancement",
        "description": "Uses XML tags for clear prompt organization (Anthropic's signature technique)",
        "template": """<instructions>
{original_prompt}
</instructions>

<context>
Provide any relevant background information that helps with understanding the task.
</context>

<examples>
<example>
<input>Sample input here</input>
<output>Expected output format</output>
</example>
</examples>

<formatting>
Please structure your response clearly and follow the examples provided.
</formatting>""",
    },
    "chain_of_thought": {
        "name": "Chain of Thought with XML",
        "description": "Encourages step-by-step reasoning with structured thinking",
        "template": """<task>
{original_prompt}
</task>

<instructions>
Before providing your final answer, please think through this step-by-step inside <thinking> tags:
1. Break down the problem
2. Consider different approaches
3. Work through the solution
4. Verify your reasoning
</instructions>

<thinking>
[Your step-by-step reasoning will go here]
</thinking>

Please provide your response after showing your thinking process.""",
    },
    "role_prompting": {
        "name": "Role-Based Prompting",
        "description": "Assigns specific expert roles for domain expertise",
        "template": """<role>
You are a world-class {persona} with {years_experience} years of experience in {domain}. You are known for your {key_strengths} and have a reputation for {reputation_traits}.
</role>

<task>
{original_prompt}
</task>

<approach>
As an expert {persona}, please:
1. Apply your specialized knowledge and experience
2. Consider industry best practices and standards
3. Provide insights that only an expert would know
4. Structure your response professionally
</approach>""",
    },
    "multishot_examples": {
        "name": "Multishot Example Enhancement",
        "description": "Provides multiple high-quality examples with consistent formatting",
        "template": """<task>
{original_prompt}
</task>

<examples>
<example_1>
<input>{example_input_1}</input>
<output>{example_output_1}</output>
</example_1>

<example_2>
<input>{example_input_2}</input>
<output>{example_output_2}</output>
</example_2>
</examples>

<instructions>
Following the pattern shown in the examples above, please process the actual task.
</instructions>""",
    },
}


class PromptEnhancer:
    def __init__(self):
        self.enhancement_history = []

    def enhance_prompt(self, prompt: str, enhancement_type: str, **kwargs) -> str:
        """Enhance the prompt using the specified strategy"""
        if enhancement_type not in ENHANCEMENT_PATTERNS:
            return prompt

        # Handle the AI rewrite strategy separately
        if enhancement_type == "ai_rewrite":
            critique, rewritten_prompt = rewrite_prompt_with_ai(
                prompt, kwargs.get("model", "llama3")
            )
            # We store the critique for display in the UI
            st.session_state.critique = critique
            return rewritten_prompt

        pattern = ENHANCEMENT_PATTERNS[enhancement_type]
        enhanced = pattern["template"].format(original_prompt=prompt, **kwargs)

        self.enhancement_history.append(
            {
                "timestamp": datetime.now().isoformat(),
                "original": prompt,
                "enhanced": enhanced,
                "type": enhancement_type,
                "parameters": kwargs,
            }
        )
        return enhanced


def get_ollama_models() -> List[str]:
    """
    Get a list of available models from the Ollama API.
    """
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=10)
        response.raise_for_status()
        data = response.json()
        models = sorted([model.get("name", "") for model in data.get("models", [])])
        if not models:
            return ["llama3:latest"]  # Default fallback
        return models
    except requests.exceptions.RequestException:
        return []


def call_ollama(prompt: str, model: str = "llama3") -> str:
    """
    Call the local Ollama API to generate a completion for the given prompt.
    """
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=90,  # Increased timeout for potentially longer generation
        )
        response.raise_for_status()
        data = response.json()
        return data.get("response", "[No response from Ollama]").strip()
    except Exception as e:
        return f"[Ollama Error: {e}]"


def rewrite_prompt_with_ai(prompt: str, model: str) -> (str, str):
    """Critiques and rewrites a prompt using an LLM."""
    critique_prompt = f"""<task>
You are a world-class prompt engineering expert. Your task is to analyze and critique the following user-provided prompt.
Identify its weaknesses based on criteria like clarity, specificity, context, constraints, and desired output format.

**User Prompt:**
`{prompt}`

Provide your critique in a <critique> XML tag. Be specific and constructive.
</task>"""

    critique = call_ollama(critique_prompt, model)

    rewrite_prompt = f"""<task>
You are a world-class prompt engineering expert. You will be given an original prompt and a critique of that prompt.
Your task is to rewrite the original prompt to be a much more effective, "best-in-class" prompt, addressing all the points in the critique.
The new prompt should be significantly more detailed and structured, incorporating principles like XML tagging, clear instructions, context, and examples where appropriate.

**Original Prompt:**
`{prompt}`

**Critique:**
{critique}

Now, provide the new, rewritten prompt inside a <rewritten_prompt> XML tag. Output only the content for the new prompt, without the XML tag itself.
</task>"""

    rewritten_prompt = call_ollama(rewrite_prompt, model)

    # Clean up the output to remove the XML tags if they are present
    critique = critique.replace("<critique>", "").replace("</critique>", "").strip()
    rewritten_prompt = (
        rewritten_prompt.replace("<rewritten_prompt>", "")
        .replace("</rewritten_prompt>", "")
        .strip()
    )

    return critique, rewritten_prompt


def choose_enhancement_strategy(prompt: str, model: str) -> str:
    """
    Uses an LLM to choose the best enhancement strategy for a given prompt.
    """
    # Give the AI a strong preference for the rewrite strategy for complex prompts
    if len(prompt.split()) > 15:
        return "ai_rewrite"

    strategy_descriptions = "\n".join(
        [
            f"- **{key}**: {details['name']} - {details['description']}"
            for key, details in ENHANCEMENT_PATTERNS.items()
        ]
    )

    meta_prompt = f"""<task>
You are an expert in prompt engineering. Your task is to analyze the following user prompt and choose the single best enhancement strategy from the list provided.

**User Prompt:**
\"{prompt}\"

**Available Enhancement Strategies:**
{strategy_descriptions}

**Instructions:**
1. Read the user prompt carefully.
2. For simple, short prompts, a template-based approach is fine.
3. For more complex or vague prompts, `ai_rewrite` is usually the best choice.
4. Respond with ONLY the identifier of your chosen strategy (e.g., xml_structure, ai_rewrite). Do not add any other text or explanation.
</task>

Chosen strategy identifier:"""

    chosen_strategy_raw = call_ollama(meta_prompt, model).strip().lower()

    for key in ENHANCEMENT_PATTERNS.keys():
        if key in chosen_strategy_raw:
            return key

    return "xml_structure"  # Fallback
