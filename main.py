# main.py
from pprint import pprint

from agents.jisa_agents import run_smart_jisa_pipeline

if __name__ == "__main__":
    print("=== Smart JISA (root agent → sub-agents → tools → output) ===\n")

    title = input("Enter Jira issue TITLE:\n> ").strip()
    description = input("\nEnter Jira issue DESCRIPTION:\n> ").strip()

    result = run_smart_jisa_pipeline(title, description)

    print("\n--- Cleaned text ---")
    print(result["cleaned_text"])

    print("\n--- Similar issues (raw) ---")
    pprint(result["similar_issues"])

    print("\n--- LLM triage report ---\n")
    print(result["llm_report"])
