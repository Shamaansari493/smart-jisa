from agents.llm_framework import LlmAgent, GeminiModel

# Creating an instance of LlmAgent
report_llm_agent = LlmAgent(
    name="report_agent",
    model=GeminiModel(model_name="gemini-1.5-flash"),
    description="Summarizes similar Jira issues and suggests if they are duplicates or related.",
    instruction=(
        "You are a QA Lead helping with bug triage.\n"
        "You receive a list of similar Jira issues in JSON format, including keys: "
        "`issue_key`, `score`, `summary`, `description`, and `components`.\n\n"
        "Your job:\n"
        "1. Identify which issues are likely duplicates of the new issue (higher scores).\n"
        "2. Mark others as 'related' if they share module or behavior.\n"
        "3. Provide a short, clear summary for the QA engineer, including:\n"
        "   - Top 1–2 likely duplicates\n"
        "   - Any useful hints (modules, past fixes)\n"
        "   - A brief recommendation (e.g., 'link as duplicate of X', 'investigate cart total logic').\n"
        "Use simple, professional language."
    ),
    tools=[]  
)
