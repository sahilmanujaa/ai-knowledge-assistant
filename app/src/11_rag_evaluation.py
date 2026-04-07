"""
11 — RAG Benchmark and Evaluation
======================================================

This script benchmarks multiple retrieval strategies against the same
evaluation dataset so you can compare them side by side.

Supported systems:
  - base         : plain Chroma similarity retrieval
  - multi_query  : LangChain MultiQueryRetriever
  - hybrid       : BM25 + vector ensemble
  - rerank       : topic router + broad retrieval + cross-encoder reranking
  - self_query   : metadata-aware self-query retrieval

Metrics:
  - Hit@k
  - MRR
  - Optional answer correctness / faithfulness (LLM judge)

Dataset format:
[
  {
    "question": "What is this document about?",
    "ground_truth_answer": "A concise reference answer.",
    "expected_sources": ["example.pdf"],
    "expected_topic": "python"
  }
]

Examples:
    python3 app/src/11_rag_evaluation.py
    RAG_EVAL_SYSTEMS=base,multi_query,hybrid python3 app/src/11_rag_evaluation.py
"""

import json
import os
from statistics import mean

from dotenv import load_dotenv
from retrieval_systems import (
    build_base_retriever,
    build_hybrid_retriever,
    build_llm,
    build_multi_query_retriever,
    build_rerank_retriever,
    build_self_query_retriever,
    build_vectorstore,
)


load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(BASE_DIR, "data", "rag_eval_dataset.sample.json")
SELECTED_SYSTEMS = [
    system.strip()
    for system in os.getenv(
        "RAG_EVAL_SYSTEMS",
        "base,multi_query,hybrid,rerank,self_query",
    ).split(",")
    if system.strip()
]

ANSWER_PROMPT = """Answer the question based only on the following context.

{context}

---

Question: {question}
"""


JUDGE_PROMPT = """You are evaluating a RAG system output.

Score the answer on two dimensions from 1 to 5:
- correctness: How well the answer matches the reference answer.
- faithfulness: How well the answer stays grounded in the retrieved context.

Return valid JSON only with this schema:
{{
  "correctness": 1,
  "faithfulness": 1,
  "reason": "short explanation"
}}

Question: {question}
Reference answer: {reference_answer}
Retrieved context:
{context}

Model answer:
{model_answer}
"""


def load_dataset(dataset_path: str):
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list) or not data:
        raise ValueError("Evaluation dataset must be a non-empty JSON list.")

    for index, item in enumerate(data, start=1):
        if "question" not in item:
            raise ValueError(f"Dataset row {index} is missing 'question'.")

    return data


def reciprocal_rank(retrieved_sources, expected_sources):
    for rank, source in enumerate(retrieved_sources, start=1):
        if source in expected_sources:
            return 1 / rank
    return 0.0


def build_answer(llm, question, docs):
    context = "\n\n---\n\n".join(doc.page_content for doc in docs)
    answer = llm.invoke(ANSWER_PROMPT.format(context=context, question=question))
    return answer.content, context


def judge_answer(llm, question, reference_answer, context, model_answer):
    raw = llm.invoke(
        JUDGE_PROMPT.format(
            question=question,
            reference_answer=reference_answer,
            context=context,
            model_answer=model_answer,
        )
    ).content
    return json.loads(raw)


def init_shared_components():
    embedding, vectorstore = build_vectorstore()
    llm = build_llm()
    return embedding, vectorstore, llm


def init_systems(vectorstore, llm):
    systems = {}
    skipped = {}

    base_retriever = build_base_retriever(vectorstore)
    hybrid_retriever = build_hybrid_retriever(vectorstore)
    multi_query_retriever = None
    rerank_retriever = None
    self_query_retriever = None

    if llm:
        multi_query_retriever = build_multi_query_retriever(vectorstore, llm)
        rerank_retriever = build_rerank_retriever(vectorstore, llm)
        self_query_retriever = build_self_query_retriever(vectorstore, llm)

    available = {
        "base": lambda question: base_retriever.invoke(question),
        "hybrid": (lambda question: hybrid_retriever.invoke(question))
        if hybrid_retriever else None,
        "multi_query": (lambda question: multi_query_retriever.invoke(question))
        if multi_query_retriever else None,
        "rerank": rerank_retriever,
        "self_query": (lambda question: self_query_retriever.invoke(question))
        if self_query_retriever else None,
    }

    for name in SELECTED_SYSTEMS:
        if name not in available:
            skipped[name] = "unknown system"
            continue
        if available[name] is None:
            skipped[name] = "requires OPENAI_API_KEY"
            continue
        systems[name] = available[name]

    return systems, skipped


def print_summary(results):
    headers = [
        ("System", 14),
        ("Hit@k", 8),
        ("MRR", 8),
        ("SrcCov", 8),
        ("TopicAcc", 10),
        ("Correct", 10),
        ("Faithful", 10),
    ]

    def fmt(value, digits=3):
        if value is None:
            return "-"
        return f"{value:.{digits}f}"

    line = " ".join(label.ljust(width) for label, width in headers)
    print("\n" + "=" * len(line))
    print(line)
    print("=" * len(line))

    for name, stats in results.items():
        row = [
            name.ljust(14),
            fmt(stats["hit_at_k"]).ljust(8),
            fmt(stats["mrr"]).ljust(8),
            fmt(stats["source_coverage"]).ljust(8),
            fmt(stats["topic_accuracy"]).ljust(10),
            fmt(stats["avg_correctness"], 2).ljust(10),
            fmt(stats["avg_faithfulness"], 2).ljust(10),
        ]
        print(" ".join(row))


def main():
    print("Loading evaluation dataset...")
    try:
        dataset = load_dataset(DATASET_PATH)
    except Exception as exc:
        print(f"Failed to load dataset: {exc}")
        print(f"Expected file: {DATASET_PATH}")
        return

    try:
        _, vectorstore, llm = init_shared_components()
    except Exception as exc:
        print(f"Failed to initialise shared components: {exc}")
        return

    print("Initialising benchmark systems...")
    systems, skipped = init_systems(vectorstore, llm)

    if skipped:
        for name, reason in skipped.items():
            print(f"Skipping '{name}': {reason}")

    if not systems:
        print("No benchmark systems available to evaluate.")
        return

    judge_llm = llm
    results = {
        name: {
            "hit_scores": [],
            "mrr_scores": [],
            "source_coverage_scores": [],
            "topic_accuracy_scores": [],
            "correctness_scores": [],
            "faithfulness_scores": [],
        }
        for name in systems
    }

    print(f"Running benchmark on {len(dataset)} examples...\n")

    for index, item in enumerate(dataset, start=1):
        question = item["question"]
        expected_sources = {
            os.path.basename(source) for source in item.get("expected_sources", [])
        }
        expected_topic = item.get("expected_topic")
        reference_answer = item.get("ground_truth_answer")

        print(f"[{index}] {question}")

        for system_name, retrieve in systems.items():
            try:
                docs = retrieve(question)
            except Exception as exc:
                print(f"  - {system_name}: failed during retrieval: {exc}")
                continue

            retrieved_sources = [
                os.path.basename(doc.metadata.get("source", "Unknown")) for doc in docs
            ]
            retrieved_topics = {
                doc.metadata.get("topic") for doc in docs if doc.metadata.get("topic")
            }

            hit = 1.0 if expected_sources and any(
                source in expected_sources for source in retrieved_sources
            ) else 0.0

            rr = reciprocal_rank(retrieved_sources, expected_sources) if expected_sources else 0.0

            source_coverage = 0.0
            if expected_sources:
                matched = len(expected_sources.intersection(set(retrieved_sources)))
                source_coverage = matched / len(expected_sources)

            topic_accuracy = None
            if expected_topic:
                topic_accuracy = 1.0 if expected_topic in retrieved_topics else 0.0

            results[system_name]["hit_scores"].append(hit)
            results[system_name]["mrr_scores"].append(rr)
            results[system_name]["source_coverage_scores"].append(source_coverage)
            if topic_accuracy is not None:
                results[system_name]["topic_accuracy_scores"].append(topic_accuracy)

            print(
                f"  - {system_name:<11} "
                f"Hit={hit:.0f} MRR={rr:.3f} Sources={retrieved_sources}"
            )

            if judge_llm and reference_answer and docs:
                try:
                    model_answer, context = build_answer(judge_llm, question, docs)
                    judged = judge_answer(
                        judge_llm,
                        question,
                        reference_answer,
                        context,
                        model_answer,
                    )
                    results[system_name]["correctness_scores"].append(judged["correctness"])
                    results[system_name]["faithfulness_scores"].append(judged["faithfulness"])
                except Exception as exc:
                    print(f"    answer judging skipped for {system_name}: {exc}")

        print()

    aggregated = {}
    for system_name, metric_lists in results.items():
        aggregated[system_name] = {
            "hit_at_k": mean(metric_lists["hit_scores"]) if metric_lists["hit_scores"] else None,
            "mrr": mean(metric_lists["mrr_scores"]) if metric_lists["mrr_scores"] else None,
            "source_coverage": (
                mean(metric_lists["source_coverage_scores"])
                if metric_lists["source_coverage_scores"] else None
            ),
            "topic_accuracy": (
                mean(metric_lists["topic_accuracy_scores"])
                if metric_lists["topic_accuracy_scores"] else None
            ),
            "avg_correctness": (
                mean(metric_lists["correctness_scores"])
                if metric_lists["correctness_scores"] else None
            ),
            "avg_faithfulness": (
                mean(metric_lists["faithfulness_scores"])
                if metric_lists["faithfulness_scores"] else None
            ),
        }

    print_summary(aggregated)

    if not judge_llm:
        print("\nAnswer quality scoring was skipped because OPENAI_API_KEY is not set.")


if __name__ == "__main__":
    main()
