from typing import Dict, Any, List

from agents.llm_framework import Agent, Gemini, InMemorySessionService, Runner
from utils.config import APP_NAME, DEFAULT_USER_ID, DEFAULT_SESSION_ID
from agents.ingestion_agent import IngestionAgent
from agents.similarity_agent import SimilarityAgent

# --- Tools (Python) ---
_ingestor_tool = IngestionAgent()
_similarity_tool = SimilarityAgent()

# --- Sub-agents (LLM, for observability / reasoning) ---

similarity_explainer_agent = Agent(
    model=Gemini(model="gemini-2.5-flash-lite"),
    name="similarity_explainer_agent",
    description="Explains similarity results in plain English.",
    instruction=(
        "You receive JSON with a list of similar Jira issues (issue_key, score, summary, description, components).\n"
        "Explain briefly what patterns you see and which issues are more likely duplicates vs just related."
    ),
    tools=[],
)

report_agent = Agent(
    model=Gemini(model="gemini-2.5-flash-lite"),
    name="report_agent",
    description="Produces the final triage summary for QA.",
    instruction=(
        "You are a senior QA lead.\n"
        "Given the cleaned text of a new Jira issue and a list of similar issues (JSON), "
        "write a short triage summary:\n"
        "- Mention top likely duplicates\n"
        "- Mention related issues if helpful\n"
        "- Suggest a recommended action (e.g., 'link as duplicate of X', "
        "'investigate separately in module Y').\n"
        "Be concise and professional."
    ),
    tools=[],
)

# Root agent (conceptual orchestrator, like in Kaggle)
root_agent = Agent(
    model=Gemini(model="gemini-2.5-flash-lite"),
    name="smart_jisa_root_agent",
    description="Root orchestrator for Smart JISA.",
    instruction=(
        "You orchestrate Smart JISA. Logically, you:\n"
        "1) clean the issue text,\n"
        "2) find similar issues,\n"
        "3) create a final triage report.\n"
        "Actual tool execution is handled in Python, but you represent the root decision-maker."
    ),
    tools=[],
)


def run_smart_jisa_pipeline(
    title: str,
    description: str,
    app_name: str = APP_NAME,
    user_id: str = DEFAULT_USER_ID,
    session_id: str = DEFAULT_SESSION_ID,
) -> Dict[str, Any]:
    """
    Root → tools → sub-agents → output.
    """

    # For LLM parts we use Runner + InMemorySessionService 
    session_service = InMemorySessionService()
    report_runner = Runner(agent=report_agent, app_name=app_name, session_service=session_service)

    # 1. Ingestion (tool)
    cleaned_text = _ingestor_tool.run(title, description)

    # 2. Similarity search (tool)
    similar_issues: List[Dict[str, Any]] = _similarity_tool.find_similar(cleaned_text, top_k=5)

    # 3. Final triage report (LLM agent)
    context = {
        "cleaned_issue_text": cleaned_text,
        "similar_issues": similar_issues,
    }

    llm_report = report_runner.run(
        user_id=user_id,
        session_id=session_id,
        user_input="Analyze the similar issues and provide a triage summary.",
        context=context,
    )

    return {
        "cleaned_text": cleaned_text,
        "similar_issues": similar_issues,
        "llm_report": llm_report,
    }
