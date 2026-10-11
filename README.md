# RAG Study Assistant over Scanned Handwritten Notes

A Retrieval-Augmented Generation (RAG) pipeline that answers questions from my scanned, handwritten Data Structures & Algorithms notes, plus a small evaluation harness to measure how well it works.

## Problem

Plain LLMs answer confidently even when they have never seen your material. I wanted answers grounded in my own lecture notes, and I wanted a way to measure whether the system is actually good instead of just trusting how the answers look.

The notes are scanned handwriting, so there is no selectable text. That made the input messy, which turned out to be the most interesting part of the project.

## How it works

```
Scanned PDF -> OCR (Tesseract) -> fixed-size chunks -> embeddings (MiniLM)
                                                          |
Question -> embedding -> cosine similarity -> top-k chunks -> Gemini -> grounded answer
```

1. **OCR:** `pdf2image` (Poppler) turns each page into an image and `pytesseract` reads it. The result is cached to disk because OCR is slow.
2. **Chunking:** fixed-size character chunks (default 500 characters, 50 overlap).
3. **Embeddings:** `all-MiniLM-L6-v2` from `sentence-transformers`, run locally.
4. **Retrieval:** cosine similarity between the question and every chunk, returning the top-k.
5. **Generation:** Gemini gets the retrieved chunks and is told to answer using only that context, or say it does not have enough information.

## Tech stack

Python, Tesseract OCR, Poppler, sentence-transformers (PyTorch), scikit-learn, Google Gemini API (`google-genai`).

## Setup

1. Install [Tesseract](https://github.com/UB-Mannheim/tesseract/wiki) and [Poppler for Windows](https://github.com/oschwartz10612/poppler-windows/releases/).
2. Edit the two paths at the top of `src/load_document.py` (`tesseract_cmd` and `POPPLER_PATH`) to match your install.
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Get a free API key from Google AI Studio and set it in your terminal (never put it in a file):
   ```
   $env:GEMINI_API_KEY="your-key"
   ```
5. Put your PDF in `Data/`. My notes are not included in this repo.

## Usage

Run the evaluation on a PDF, choosing how many chunks to retrieve:

```
python src/evaluate.py Data/DSA_Lecture.pdf run_k3.json --k 3
```

Options: `--k` (chunks retrieved), `--chunk-size`, `--overlap`. Questions and expected answers live in `Data/eval_set.json`. Answers are cached in `answer_cache.json`, keyed by all settings, so reruns do not waste API calls.

## Evaluation

I wrote 10 questions from my notes with known answers (heap insertion, preorder traversal, AVL balance factor, load factor, merge sort, quick sort, and so on) and graded the generated answers by hand.

| Retrieval | Correct | Wrong | Refused |
|---|---|---|---|
| top-1 chunk | 1 / 10 | 0 | 9 |
| top-3 chunks | 4 / 10 | 1 | 5 |

Same model, same prompt, same chunks. The only change was how many chunks reached the model, and correct answers went from 1 to 4. So retrieval, not the LLM, was the bottleneck. This is a small test (10 questions, one run), so treat it as a strong hint, not proof.

What I learned from the results:

- **The grounding prompt works.** When the retrieved text did not contain the answer, the model said so instead of guessing.
- **More context also made one answer wrong.** For "null links in a binary tree with n nodes" (correct: n + 1), top-3 gave `2n - 1`, because it got tangled in a garbled proof in my notes.
- **A keyword refusal detector is not reliable.** It missed phrasings like "the information is not sufficient". Correctness needs a better judge.

## Limitations

- OCR on handwriting is poor, for example "Binary Tree" comes out as "Buinaony tneec". Everything downstream is limited by this.
- Fixed-size chunking cuts text mid-thought, so one chunk often mixes several topics.
- The model sometimes seems to repair garbled text from what it already knows, so answers are not always purely from the notes.
- Only 10 evaluation questions and a single document. Results are indicative, not statistically solid.
- Brute-force similarity search over every chunk. Fine for about 20 chunks, would not scale.
- Free-tier API limits (per-minute and per-day) slowed experiments. The code retries temporary server errors, stops immediately on quota errors, and caches results.

## Future improvements

- LLM-as-judge scoring of answer correctness, instead of the keyword refusal check.
- Better OCR for handwriting (a vision-model transcription step, or a handwriting-specific model).
- Smarter chunking (by topic or paragraph), and a comparison of chunk sizes.
- Hybrid search and reranking, then a vector database for larger collections.
- Page-number citations in answers.
- A larger evaluation set, and a comparison against clean typed notes.
