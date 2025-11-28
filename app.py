import json
from agents.ingestion_agent import IngestionAgent
from agents.embedding_agent import EmbeddingAgent
from agents.similarity_agent import SimilarityAgent
from agents.report_agent import ReportAgent
from utils.config import GOOGLE_API_KEY

def run_smart_jisa(new_issue_description):
    ingestor = IngestionAgent()
    cleaned = ingestor.run(new_issue_description)

    embedder = EmbeddingAgent(api_key=GOOGLE_API_KEY)
    new_embedding = embedder.embed(cleaned)

    similarity = SimilarityAgent()
    ids, scores = similarity.find_similar(new_embedding)

    with open("data/jira_issues.json") as f:
        dataset = json.load(f)

    matches = list(zip(ids, scores))
    reporter = ReportAgent()
    report = reporter.create_report(matches, dataset)

    return report

if __name__ == "__main__":
    issue = input("Enter your Jira issue description:\n")
    result = run_smart_jisa(issue)

    print("\n🔍 Similar Issues Found:")
    for r in result:
        print(f"{r['issue_id']}  | {r['title']}  | Score: {r['similarity_score']}")
