import argparse
import json
import os
import time
from sentence_transformers import SentenceTransformer
from load_document import (
    load_scanned_pdf, chunk_text, find_top_chunks, generate_answer, MODEL_NAME
)

CACHE_PATH = "answer_cache.json"
REFUSAL_PHRASES = [
    "not enough", "does not contain", "doesn't contain", "no mention",
    "not mentioned", "no information", "not provided", "cannot answer",
    "cannot be determined",
]


def load_cache() -> dict:
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, "r") as f:
            return json.load(f)
    return {}


def save_cache(cache: dict):
    with open(CACHE_PATH, "w") as f:
        json.dump(cache, f, indent=2)


def classify(answer: str) -> str:
    a = answer.lower()
    if a.startswith("[quota exhausted") or a.startswith("[generation failed"):
        return "error"
    if any(p in a for p in REFUSAL_PHRASES):
        return "refused"
    return "answered"


def run_evaluation(pdf_path: str, k: int, chunk_size: int, overlap: int):
    with open("Data/eval_set.json", "r") as f:
        eval_set = json.load(f)

    text = load_scanned_pdf(pdf_path)
    chunks = chunk_text(text, chunk_size, overlap)
    print(f"{len(chunks)} chunks (size={chunk_size}, overlap={overlap}), retrieving top {k}")

    model = SentenceTransformer("all-MiniLM-L6-v2")
    chunk_embeddings = model.encode(chunks)  # embed once, reuse for every question
    cache = load_cache()

    results = []
    for i, item in enumerate(eval_set):
        question, expected = item["question"], item["answer"]
        top = find_top_chunks(question, chunks, chunk_embeddings, model, k)

        # the key includes every setting that changes the answer
        key = f"{pdf_path}|{MODEL_NAME}|size{chunk_size}|ov{overlap}|k{k}|{question}"
        if key in cache:
            answer = cache[key]
            print(f"[{i+1}/{len(eval_set)}] (cached)")
        else:
            answer = generate_answer(question, [c for c, _ in top])
            if classify(answer) != "error":
                cache[key] = answer
                save_cache(cache)
            print(f"[{i+1}/{len(eval_set)}] done")
            time.sleep(13)

        results.append({
            "question": question,
            "expected": expected,
            "top_score": top[0][1],
            "generated_answer": answer,
            "status": classify(answer),
        })
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf_path")
    parser.add_argument("output_file")
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--overlap", type=int, default=50)
    args = parser.parse_args()

    results = run_evaluation(args.pdf_path, args.k, args.chunk_size, args.overlap)

    counts = {s: sum(1 for r in results if r["status"] == s)
              for s in ("answered", "refused", "error")}
    avg_score = sum(r["top_score"] for r in results) / len(results)

    print(f"\n=== SUMMARY: k={args.k}, chunk_size={args.chunk_size} ===")
    print(f"Answered: {counts['answered']} | Refused: {counts['refused']} | Errors: {counts['error']}")
    print(f"Average top retrieval score: {avg_score:.4f}")

    for r in results:
        print(f"\nQ: {r['question']}")
        print(f"Expected: {r['expected']}")
        print(f"[{r['status']}] {r['generated_answer']}")
        print("-" * 40)

    with open(args.output_file, "w") as f:
        json.dump({"settings": vars(args), "counts": counts, "results": results}, f, indent=2)
    print(f"\nSaved to {args.output_file}")