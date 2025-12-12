"""
Evaluate a RAG agent with DeepEval metrics on a Q&A dataset.

What this script does
---------------------
Given a JSON dataset of test cases with:

  [
    {
      "query": "...",
      "expected_answer": "..."
    },
    ...
  ]

it will:

  1. Build the movie search agent (text or multimodal mode).
  2. For each query:
       - Run the agent and capture the final answer (`llm_output`).
       - Extract retrieved text snippets from the agent's tool calls
         (`movie_multimodal_search`, `image_search`).
       - Attach:
           * "llm_output": agent's final answer
           * "relevant_snippets": collected text snippets
  3. Evaluate each test case with DeepEval metrics:
       - ContextualRecallMetric
       - ContextualRelevancyMetric
       - AnswerRelevancyMetric
       - FaithfulnessMetric
  4. Save an enriched dataset with the added fields.
  5. Print a small table with the average score for each metric.

CLI flags
---------
  --dataset PATH        (required) JSON file with 'query' and 'expected_answer'.
  --mode [text|multimodal]
                        Retrieval mode. 'multimodal' nudges the agent to
                        use image-aware search prompts.
  --k INT               Max number of text snippets to keep per query (default: 4).
  --inplace             Overwrite the input dataset in place.
  --output PATH         Optional output path (ignored if --inplace is set).
  --eval-model NAME     LLM name used by DeepEval metrics
                        (e.g. 'gpt-4o-mini').
"""

from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    FaithfulnessMetric,
)
from deepeval.test_case import LLMTestCase

from langchain_core.messages import ToolMessage

from src.agent.agent import Context, build_movie_agent


def load_dataset(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_dataset(data: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def _try_json_loads(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def extract_movies_from_messages(messages) -> List[Dict[str, Any]]:
    """
    Collect movie results from the agent's tool calls.
    Looks for ToolMessages named:
      - movie_multimodal_search
      - image_search
    """
    collected: List[Dict[str, Any]] = []

    for msg in messages:
        msg_type = getattr(msg, "type", None)
        if msg_type != "tool" and not isinstance(msg, ToolMessage):
            continue

        tool_name = getattr(msg, "name", "") or ""
        if tool_name not in {"movie_multimodal_search", "image_search"}:
            continue

        content = msg.content
        payload: Optional[Any] = None
        movie_list: List[Dict[str, Any]] = []

        if isinstance(content, str):
            payload = _try_json_loads(content)

        elif isinstance(content, list):
            for block in content:
                text = None
                if isinstance(block, dict):
                    text = block.get("text") or block.get("content")
                else:
                    text = getattr(block, "text", None)

                if not text:
                    continue

                block_payload = _try_json_loads(text)
                if isinstance(block_payload, dict):
                    movie_list.extend(block_payload.get("results") or [])
                elif isinstance(block_payload, list):
                    movie_list.extend(block_payload)

        elif isinstance(content, dict):
            payload = content

        if payload is not None:
            if isinstance(payload, dict):
                movie_list = payload.get("results") or []
            elif isinstance(payload, list):
                movie_list = payload

        collected.extend(movie_list)

    return collected


def run_agent_query(agent, query: str, use_image_prompt: bool, k: int) -> Tuple[str, List[str]]:
    """
    Execute the agent on a single query and return:
      - llm_output: final answer string
      - snippets: up to k text snippets from retrieved articles
    """
    session_id = f"eval:{uuid.uuid4()}"

    if use_image_prompt:
        input_text = f"{query}\n[Evaluator note: include image query if helpful.]"
    else:
        input_text = f"{query}\n[Evaluator note: please do not use image query, only use text queries]"

    response = agent.invoke(
        {"messages": [{"role": "user", "content": input_text}]},
        config={"configurable": {"thread_id": session_id}},
        context=Context(session_id=session_id),
    )

    messages = response.get("messages") or []
    movies = extract_movies_from_messages(messages)

    snippets: List[str] = []
    for movie in movies:
        for snip in movie.get("text_snippets") or []:
            if snip and snip not in snippets:
                snippets.append(snip)
            if len(snippets) >= k:
                break

        if len(snippets) >= k:
            break

        overview = movie.get("overview")
        if overview and overview not in snippets:
            snippets.append(overview)

        if len(snippets) >= k:
            break

    llm_output = ""
    if messages:
        llm_output = getattr(messages[-1], "content", None) or ""
    if not llm_output:
        llm_output = response.get("output", "") or ""

    return llm_output, snippets[:k]


def format_row(values: Tuple[Any, ...], widths: Tuple[int, ...]) -> str:
    return " | ".join(str(v).ljust(w) for v, w in zip(values, widths))


def summarize(scores: Sequence[float]) -> float:
    return sum(scores) / len(scores) if scores else 0.0


def evaluate_dataset(dataset_path: Path, mode: str, k: int, inplace: bool, output: Optional[Path], eval_model: Optional[str]) -> None:
    data = load_dataset(dataset_path)
    if not isinstance(data, list):
        raise ValueError(f"Dataset must be a JSON array, got {type(data)}")

    use_image_prompt = mode == "multimodal"
    out_path = dataset_path if inplace else (output or dataset_path.with_name(f"{dataset_path.stem}_with_results.json"))

    print(f"[INFO] Building agent (mode={mode}, k={k}, use_image_prompt={use_image_prompt})")
    agent = build_movie_agent()

    metrics = [
        ContextualRecallMetric(model=eval_model) if eval_model else ContextualRecallMetric(),
        ContextualRelevancyMetric(model=eval_model) if eval_model else ContextualRelevancyMetric(),
        AnswerRelevancyMetric(model=eval_model) if eval_model else AnswerRelevancyMetric(),
        FaithfulnessMetric(model=eval_model) if eval_model else FaithfulnessMetric(),
    ]
    metric_names = [
        getattr(m, "name", getattr(m, "metric_name", m.__class__.__name__))
        for m in metrics
    ]
    metric_scores: Dict[str, List[float]] = {name: [] for name in metric_names}

    for entry in data:
        query = (entry.get("query") or "").strip()
        expected_answer = (entry.get("expected_answer") or "").strip()

        if not query:
            raise ValueError("Each dataset entry must have a non-empty 'query' string.")
        if not expected_answer:
            raise ValueError("Each dataset entry must have a non-empty 'expected_answer' string.")

        try:
            llm_output, snippets = run_agent_query(
                agent=agent,
                query=query,
                use_image_prompt=use_image_prompt,
                k=k,
            )
        except Exception as e:
            print(f"[ERROR] Agent failed for query '{query[:80]}': {e}")
            entry["llm_output"] = ""
            entry["relevant_snippets"] = []
            entry["error"] = f"agent: {e}"
            continue

    
        entry["llm_output"] = llm_output
        entry["relevant_snippets"] = snippets

        test_case = LLMTestCase(
            input=query,
            actual_output=llm_output,
            expected_output=expected_answer,
            retrieval_context=snippets,
        )


        for m, name in zip(metrics, metric_names):
            try:
                m.measure(test_case)
                metric_scores[name].append(m.score)
            except Exception as e:
                print(f"[ERROR] Metric {name} failed for query '{query[:80]}': {e}")
                metric_scores[name].append(0.0)
                entry.setdefault("metric_errors", {})[name] = str(e)


    save_dataset(data, out_path)
    print(f"[INFO] Wrote annotated dataset with agent outputs to: {out_path}")


    widths = (28, 8)
    header = ("Metric", "Avg")
    divider = "---".join("-" * w for w in widths)

    print()
    print(format_row(header, widths))
    print(divider)
    for name in metric_names:
        avg = summarize(metric_scores[name])
        print(format_row((name, f"{avg:.3f}"), widths))
    print()




def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a RAG agent with DeepEval metrics on a JSON Q&A dataset.",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        type=Path,
        help="Path to JSON dataset with 'query' and 'expected_answer' fields.",
    )
    parser.add_argument(
        "--mode",
        choices=["text", "multimodal"],
        default="text",
        help="Retrieval mode. 'multimodal' nudges the agent to include image search.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=4,
        help="Max number of snippets to keep per query.",
    )
    parser.add_argument(
        "--inplace",
        action="store_true",
        help="Overwrite the input dataset instead of writing a new file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional explicit output path. Ignored if --inplace is set.",
    )
    parser.add_argument(
        "--eval-model",
        type=str,
        help="LLM name for DeepEval metrics (e.g. 'gpt-4o-mini'). "
             "If omitted, metric defaults are used (currently gpt-4.1).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluate_dataset(
        dataset_path=args.dataset,
        mode=args.mode,
        k=args.k,
        inplace=args.inplace,
        output=args.output,
        eval_model=args.eval_model,
    )


if __name__ == "__main__":
    main()
