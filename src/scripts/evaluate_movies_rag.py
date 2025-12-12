"""
Evaluate retrieval quality for the Multimodal Movie RAG system.

Features:
  - Loads a dataset of queries with expected movie_ids.
  - Runs the LangGraph-based movie agent end-to-end.
  - Extracts movie results from movie_multimodal_search / image_search tool calls.
  - Appends 'retrieved_relevant_movie_ids' for each record.
  - Computes Recall@K, Precision@K, MRR, and HitRate@K and prints a table.

Usage examples:

  # Text-only dataset, save annotated copy next to the input file
  python -m src.scripts.evaluate_movies_rag \
      --dataset data_test/metrics/movies_questions.json \
      --mode text \
      --k 5

  # Multimodal dataset (nudges agent to use poster/image search)
  python -m src.scripts.evaluate_movies_rag \
      --dataset data_test/metrics/movies_questions.json \
      --mode multimodal \
      --k 5 \
      --inplace
"""

from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.agent.agent import build_movie_agent, Context
from langchain_core.messages import AIMessage, ToolMessage



def load_dataset(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_dataset(data: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")



def compute_metrics(expected: Sequence[str], retrieved: Sequence[str], k: int) -> Dict[str, float]:
    """
    Compute basic retrieval metrics at cutoff k.
      - Recall@K: fraction of expected ids found in top-k.
      - Precision@K: fraction of retrieved top-k that are relevant.
      - MRR: reciprocal rank of first relevant in top-k (0 if none).
      - HitRate@K: 1 if any relevant in top-k else 0.
    """
    expected_set = set(expected)
    topk = list(retrieved[:k])

    hits = [mid for mid in topk if mid in expected_set]

    recall = (len(set(hits)) / len(expected_set)) if expected_set else 0.0
    precision = (len(hits) / len(topk)) if topk else 0.0

    mrr = 0.0
    for idx, mid in enumerate(topk, start=1):
        if mid in expected_set:
            mrr = 1.0 / idx
            break

    hit_rate = 1.0 if hits else 0.0

    return {
        "recall": recall,
        "precision": precision,
        "mrr": mrr,
        "hit_rate": hit_rate,
    }



def _try_json_loads(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def extract_movies_from_messages(messages) -> List[Dict[str, Any]]:
    """
    Pull movie results from tool messages in the agent's final state.

    I look for ToolMessage-like objects where:
      - msg.type == "tool"
      - msg.name in {"movie_multimodal_search", "image_search"}

    Each such message's .content is interpreted as:
      - a JSON string with {"results": [...]} or
      - a JSON array of movie dicts.
    """
    collected: List[Dict[str, Any]] = []

    for msg in messages:
        msg_type = getattr(msg, "type", None)
        msg_name = getattr(msg, "name", None)

        if msg_type != "tool":
            continue
        if msg_name not in {"movie_multimodal_search", "image_search"}:
            continue

        content = getattr(msg, "content", None)

        payload: Optional[Any] = None
        movie_list: List[Dict[str, Any]] = []

        if isinstance(content, str):
            payload = _try_json_loads(content)

        elif isinstance(content, list):
            for block in content:
                text = None
                if isinstance(block, dict):
                    text = block.get("text")
                else:
                    text = getattr(block, "text", None)

                if not text:
                    continue

                block_payload = _try_json_loads(text)
                if isinstance(block_payload, dict):
                    movie_list.extend(block_payload.get("results") or [])
                elif isinstance(block_payload, list):
                    movie_list.extend(block_payload)

        if payload is not None:
            if isinstance(payload, dict):
                movie_list = payload.get("results") or []
            elif isinstance(payload, list):
                movie_list = payload

        collected.extend(movie_list)

    return collected



def run_agent_query(agent, query: str, use_image_prompt: bool) -> List[str]:
    """
    Execute the movie agent for a query and return ordered movie_ids
    extracted from tool outputs.
    """
    session_id = f"eval:{uuid.uuid4()}"

    input_text = (
        f"{query}\n[Evaluator note: include image query if helpful.]"
        if use_image_prompt
        else f"{query}\n[Evaluator note: please do not use image query, only use text queries]"
    )

    config = {"configurable": {"thread_id": session_id}}
    response = agent.invoke(
        {"messages": [{"role": "user", "content": input_text}]},
        config=config,
        context=Context(session_id=session_id),
    )

    messages = response.get("messages") or []
            
    ai_msg = messages[-1]

    print("[ASSISTANT] > ", ai_msg.content)
    print()
    
    print("---- DEBUG: tool calls this turn ----")
    for msg in reversed(messages):
        if getattr(msg, "type", None) == "human":
            break

        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
            print("[AI] tool calls:")
            for tc in msg.tool_calls:
                print("   name:", tc.get("name"), "args:", tc.get("args"))

        if isinstance(msg, ToolMessage):
            print("[TOOL]", msg.name, "->", msg.content)
        print("-------------------------------------")
        print()
    movies = extract_movies_from_messages(messages)

    return [m.get("id") for m in movies if m.get("id")]



def summarize_metrics(metrics: List[Dict[str, float]]) -> Dict[str, float]:
    if not metrics:
        return {"recall": 0.0, "precision": 0.0, "mrr": 0.0, "hit_rate": 0.0}

    def avg(key: str) -> float:
        return sum(m[key] for m in metrics) / len(metrics)

    return {
        "recall": avg("recall"),
        "precision": avg("precision"),
        "mrr": avg("mrr"),
        "hit_rate": avg("hit_rate"),
    }


def format_row(values: Tuple[Any, ...], widths: Tuple[int, ...]) -> str:
    return " | ".join(str(v).ljust(w) for v, w in zip(values, widths))


def print_report(
    per_query_metrics: List[Dict[str, float]],
    data: List[Dict[str, Any]],
    k: int,
) -> None:
    widths = (4, 10, 12, 8, 10, 40)
    header = ("#", f"Recall@{k}", f"Precision@{k}", "MRR", f"Hit@{k}", "Query (truncated)")
    divider = "-+-".join("-" * w for w in widths)

    print()
    print(format_row(header, widths))
    print(divider)

    for idx, (entry, metrics) in enumerate(zip(data, per_query_metrics), start=1):
        q = entry.get("query", "")
        query_preview = q[:37] + "..." if len(q) > 40 else q
        row = (
            idx,
            f"{metrics['recall']:.3f}",
            f"{metrics['precision']:.3f}",
            f"{metrics['mrr']:.3f}",
            f"{metrics['hit_rate']:.3f}",
            query_preview,
        )
        print(format_row(row, widths))

    agg = summarize_metrics(per_query_metrics)
    print(divider)
    print(
        format_row(
            (
                "AVG",
                f"{agg['recall']:.3f}",
                f"{agg['precision']:.3f}",
                f"{agg['mrr']:.3f}",
                f"{agg['hit_rate']:.3f}",
                "",
            ),
            widths,
        )
    )
    print()



def evaluate_dataset(dataset_path: Path, mode: str, k: int, inplace: bool, output: Path | None) -> None:
    data = load_dataset(dataset_path)
    if not isinstance(data, list):
        raise ValueError(f"Dataset must be a JSON array, got {type(data)}")

    use_image_prompt = mode == "multimodal"
    out_path = dataset_path if inplace else (
        output or dataset_path.with_name(f"{dataset_path.stem}_with_results.json")
    )

    print(f"[INFO] Building movie agent (mode={mode}, k={k}, use_image_prompt={use_image_prompt})")
    agent = build_movie_agent()

    per_query_metrics: List[Dict[str, float]] = []

    for entry in data:
        query = (entry.get("query") or "").strip()
        expected = entry.get("expected_relevant_movie_ids") or []
        if not query:
            raise ValueError("Each dataset entry must have a non-empty 'query' string.")

        retrieved = run_agent_query(agent, query=query, use_image_prompt=use_image_prompt)[:k]
        entry["retrieved_relevant_movie_ids"] = retrieved

        per_query_metrics.append(compute_metrics(expected, retrieved, k))

    save_dataset(data, out_path)
    print(f"[INFO] Wrote annotated dataset with retrieval results to: {out_path}")

    print_report(per_query_metrics, data, k=k)



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate movie retrieval quality and annotate datasets."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        type=Path,
        help="Path to JSON dataset with 'query' and 'expected_relevant_movie_ids' fields.",
    )
    parser.add_argument(
        "--mode",
        choices=["text", "multimodal"],
        default="text",
        help="Retrieval mode. 'multimodal' nudges the agent to include poster/image search.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=4,
        help="Cutoff for Recall@K, Precision@K, HitRate@K, and ranking depth for MRR.",
    )
    parser.add_argument(
        "--inplace",
        action="store_true",
        help="If set, overwrite the input dataset; otherwise write to *_with_results.json.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional explicit output path. Ignored if --inplace is provided.",
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
    )


if __name__ == "__main__":
    main()
