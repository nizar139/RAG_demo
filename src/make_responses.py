import argparse
import json

from src.backend import SimpleRAG
from src.config import LLM_PROVIDER, MODEL_REPO

def main():
    parser = argparse.ArgumentParser(description="Run RAG Agent")
    parser.add_argument("--input_path", type=str, required=True, help="Path to input questions.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save answers.")
    parser.add_argument("--llm_provider", type=str, default=LLM_PROVIDER, help="LLM provider to use.")
    parser.add_argument("--model_repo", type=str, default=MODEL_REPO, help="Model repository to use.")
    args = parser.parse_args()
    
    with open(args.input_path, 'r', encoding="utf-8") as f:
        questions = json.load(f)
        
    rag_agent = SimpleRAG(
        llm_provider=LLM_PROVIDER,
        model_repo=MODEL_REPO,
    )
    
    for i,q in enumerate(questions["questions"]):
        answer = rag_agent.answer_query(q["question"])
        questions["questions"][i]["predicted_answer"] = answer
        
    with open(args.output_path, 'w') as f:
        json.dump(questions, f, indent=4)
        
if __name__ == "__main__":
    main()