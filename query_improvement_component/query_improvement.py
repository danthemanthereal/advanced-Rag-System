import os
import json
from data_loader_component.data_loader import DataLoader
from pathlib import Path
import requests
from dotenv import load_dotenv

from retrieval.hybrid_retriever import HybridRetriever

load_dotenv()


class QueryImprovementComponent:

    def __init__(self, data_loader: DataLoader,
                 model_name: str,
                 hybrid_retriever: HybridRetriever,
                 ):
        self.data_loader = data_loader
        self.model_name = model_name
        self.hybrid_retriever = hybrid_retriever

    def multi_query_retrival(self, query):

        multi_query_system_prompt = self.data_loader.load_prompt(Path(__file__).parents[1]
                                                                 / "query_improvement_component"
                                                                 / "query_augment_system_prompt.txt")

        multi_query_user_prompt = self.data_loader.load_prompt(Path(__file__).parents[1]
                                                               / "query_improvement_component"
                                                               / "query_augment_user_prompt.txt").format(
            query=query
        )

        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": multi_query_system_prompt},
                {"role": "user", "content": multi_query_user_prompt},
            ]

        }
        groq_api_key = os.getenv("GROQ_API_KEY")
        response = requests.post(
            url="https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {groq_api_key}",
                "Content-Type": "application/json; charset=utf-8",
            },
            data=json.dumps(payload),
        )

        if response.status_code != 200:
            print("error ")
            print(response.text)
        response.raise_for_status()

        llm_answer = response.json()

        json_llm_answer = json.loads(llm_answer["choices"][0]["message"]["content"]).get("alternative_queries", [])

        total_results = []
        for query in json_llm_answer:
            current_results = self.hybrid_retriever.get_top_k_hybrid_retrieval(query, 5)
            total_results.extend(current_results)

        return list(set(id for id, score in total_results))

    def create_query_with_HyDE(self, query):

        hyde_system_prompt = self.data_loader.load_prompt(Path(__file__).parents[1]
                                                          / "query_improvement_component"
                                                          / "HyDE_system_prompt.txt")

        hyde_user_prompt = self.data_loader.load_prompt(Path(__file__).parents[1]
                                                        / "query_improvement_component"
                                                        / "HyDE_user_prompt.txt").format(
            query=query
        )

        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": hyde_system_prompt},
                {"role": "user", "content": hyde_user_prompt},
            ]

        }
        groq_api_key = os.getenv("GROQ_API_KEY")
        response = requests.post(
            url="https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {groq_api_key}",
                "Content-Type": "application/json; charset=utf-8",
            },
            data=json.dumps(payload),
        )

        llm_answer = response.json()

        json_llm_answer = json.loads(llm_answer["choices"][0]["message"]["content"]).get("possible_answer", "")

        self.hybrid_retriever.get_top_k_hybrid_retrieval(json_llm_answer, 5)
