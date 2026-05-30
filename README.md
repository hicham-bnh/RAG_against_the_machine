*This project has been created as part of the 42 curriculum by mobenhab.*

# RAG Against the Machine

## Description

This project implements a **Retrieval-Augmented Generation (RAG)** system capable of answering questions about the vLLM codebase. The system ingests the vLLM repository, builds a searchable index, retrieves relevant code snippets and documentation for any given question, and generates natural language answers using a local LLM (Qwen3-0.6B).

The pipeline covers the full RAG workflow: document ingestion and chunking, BM25-based indexing and retrieval, context-augmented answer generation, and recall@k evaluation.

## Instructions

### Requirements

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) (package and project manager)

### Installation

```bash
git clone <your-repo-url>
cd RAG
uv sync
```

### Setup: place the vLLM source

The vLLM repository must be present at:

```
data/raw/vllm-0.10.1/
```

### Indexing

```bash
uv run python -m student index --max_chunk_size 2000
```

This processes all `.py` and `.md` files and saves the BM25 index to `data/processed/`.

### Searching

Single query:

```bash
uv run python -m student search "How to configure OpenAI server?" --k 10
```

Dataset:

```bash
uv run python -m student search_dataset \
  --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json \
  --k 10 \
  --save_directory data/output/search_results
```

### Answering

Single query:

```bash
uv run python -m student answer "How to configure OpenAI server?" --k 10
```

Dataset (requires prior search_dataset output):

```bash
uv run python -m student answer_dataset \
  --student_search_results_path data/output/search_results/dataset_docs_public.json \
  --save_directory data/output/search_results_and_answer
```

### Evaluation

```bash
uv run python -m student evaluate \
  --student_answer_path data/output/search_results/dataset_docs_public.json \
  --dataset_path data/datasets/AnsweredQuestions/dataset_docs_public.json
```

## System Architecture

The RAG pipeline is composed of four main components that interact sequentially:

**Chunker** (`src/chunker/chunker.py`) walks the vLLM repository and splits files into fixed-size chunks using file-type-aware splitters. Chunks are plain Python objects (Pydantic models) carrying the file path and character offsets.

**Indexer** (`src/indexer/indexer.py`) receives the list of chunks, tokenizes their content with a custom smart tokenizer, builds a BM25 index with `bm25s`, and persists both the index and the chunk metadata to disk under `data/processed/`.

**Search** (`src/search/search.py`) loads the saved index and chunks, tokenizes the query with the same smart tokenizer, runs BM25 retrieval, and returns a ranked list of `MinimalSource` objects (file path + character offsets).

**Answer** (`src/answer/answer.py`) takes a question and the retrieved source chunks, formats them into a chat-template prompt for Qwen3-0.6B, and generates a concise answer using HuggingFace Transformers.

All data models (chunks, search results, answers) are defined as Pydantic models in `src/validation/validation.py`, ensuring type safety and clean JSON serialization throughout the pipeline.

```
User query
    │
    ▼
Search ──── BM25 Index ◄──── Indexer ◄──── Chunker ◄──── vLLM repo
    │
    ▼
Answer ──── Qwen3-0.6B
    │
    ▼
JSON output (StudentSearchResultsAndAnswer)
```

## Chunking Strategy

Two strategies are applied depending on file type:

**Python files** use `PythonCodeTextSplitter` from `langchain_text_splitters`, which is aware of Python syntax and tries to split at function and class boundaries rather than cutting arbitrarily through code. This preserves semantic units (functions, methods) in each chunk.

**Markdown files** use `RecursiveCharacterTextSplitter`, which splits on paragraph and section boundaries before falling back to character-level splitting, preserving document structure.

Both strategies use a configurable `max_chunk_size` (default 2000 characters) with no overlap. Character offsets (`first_character_index`, `last_character_index`) are tracked precisely by finding each chunk's position in the original file content using `str.find()`, which is necessary for the recall@k evaluation based on IoU overlap.

## Retrieval Method

The retrieval system is built on **BM25** (Best Match 25), a classical probabilistic ranking function that scores documents based on term frequency and inverse document frequency.

A key improvement over default BM25 tokenization is a **custom smart tokenizer** applied at both indexing and query time:

1. CamelCase splitting: `AsyncLLMEngine` → `async llm engine`
2. snake_case splitting: `async_llm_engine` → `async llm engine`
3. Punctuation removal
4. Lowercasing

This significantly improves matching for code-specific queries where identifiers appear in different forms across documentation and source files.

At query time, the same tokenizer is applied to the user question before passing it to `bm25s.BM25.retrieve()`, which returns the top-k chunk indices ranked by BM25 score. Results are returned as `MinimalSource` objects with file path and character offsets.

## Performance Analysis

Results measured with the official moulinette evaluator (IoU threshold 5%, k=10, max_context_length=2000):

| Dataset | Recall@1 | Recall@3 | Recall@5 | Recall@10 | Minimum required |
|---------|----------|----------|----------|-----------|-----------------|
| Docs    | 0.570    | 0.760    | **0.810**| 0.890     | 0.80 ✅          |
| Code    | 0.290    | 0.480    | **0.540**| 0.590     | 0.50 ✅          |

Both datasets pass the required thresholds. The docs dataset scores significantly higher than code, which is expected: documentation uses natural language close to query phrasing, while code uses identifiers and patterns that are harder to match with BM25 even with the smart tokenizer.

**Timing benchmarks** (on 42 school hardware, CPU only):

| Operation | Result | Limit |
|-----------|--------|-------|
| Indexing | ~15s | 5 min ✅ |
| Cold start search | ~7.6s | 60s ✅ |
| 100 questions search | ~7s | 90s/1000q ✅ |
| Throughput | ~1800 q/s | — |

## Design Decisions

**BM25 over dense embeddings.** BM25 was chosen as the primary retrieval method for its simplicity, speed, and strong baseline performance. It requires no GPU for indexing or retrieval, respects the 5-minute indexing constraint easily, and achieves the required recall scores without additional complexity.

**Custom tokenizer.** The default `bm25s.tokenize` uses simple whitespace splitting, which misses identifiers like `AsyncLLMEngine` or `get_supported_mm_limits`. The custom tokenizer bridges the gap between natural language queries and code identifiers.

**Separate chunking strategies.** Python code and Markdown documentation have fundamentally different structure. Using syntax-aware splitting for Python preserves function boundaries, which directly improves the chance that a retrieved chunk contains the complete answer to a code-related question.

**Qwen3-0.6B with chat template.** The model is used via its native chat template (`apply_chat_template` with `enable_thinking=False`) rather than raw prompt formatting. This gives significantly better instruction-following behavior compared to a plain prompt string, which was causing the model to hallucinate or continue the context instead of answering.

**Pydantic models for all data.** Using Pydantic throughout (chunks, search results, answers) ensures that the output JSON always matches the format expected by the moulinette, and provides clear validation errors when something goes wrong.

## Challenges Faced

**Disk space on 42 machines.** The home directory quota (18GB) was quickly exhausted by PyTorch CUDA dependencies (~4GB). The solution was to move `~/.cache/uv` and `~/.cache/huggingface` to `/sgoinfre` using symlinks, which has no quota on 42 machines.

**Moulinette field name mismatch.** The moulinette expected a `question_str` field in `MinimalSearchResults`, while the Pydantic models originally used `question`. This caused 100 validation errors. The fix was renaming the field in the model and updating all construction sites in `student.py`.

**LLM answer quality.** The initial prompt format caused Qwen3-0.6B to reproduce the context or hallucinate rather than extract the answer. Switching to the model's native chat template with `enable_thinking=False` and extracting only the generated tokens (not the prompt) resolved this.

**`answer_dataset` parameter name.** The original method signature used `dataset_path`, but the moulinette calls it with `--student_search_results_path`. The parameter was renamed and the input was changed to read `StudentSearchResults` (already-retrieved sources) rather than re-running search.

## Example Usage

Complete end-to-end workflow:

```bash
# 1. Index the vLLM repository
uv run python -m student index --max_chunk_size 2000

# 2. Search a single question
uv run python -m student search "What HTTP endpoint loads a LoRA adapter?" --k 5

# 3. Answer a single question
uv run python -m student answer "What HTTP endpoint loads a LoRA adapter?" --k 5

# 4. Process the full docs dataset
uv run python -m student search_dataset \
  --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json \
  --k 10

# 5. Generate answers for all retrieved results
uv run python -m student answer_dataset \
  --student_search_results_path data/output/search_results/dataset_docs_public.json \
  --save_directory data/output/search_results_and_answer

# 6. Evaluate retrieval quality
uv run python -m student evaluate \
  --student_answer_path data/output/search_results/dataset_docs_public.json \
  --dataset_path data/datasets/AnsweredQuestions/dataset_docs_public.json
```

## Resources

**RAG and retrieval:**
- Lewis et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.* https://arxiv.org/abs/2005.11401
- Robertson & Zaragoza (2009). *The Probabilistic Relevance Framework: BM25 and Beyond.* https://www.nowpublishers.com/article/Details/INR-019
- bm25s library documentation: https://github.com/xhluca/bm25s

**LLMs and generation:**
- Qwen3 model card: https://huggingface.co/Qwen/Qwen3-0.6B
- HuggingFace Transformers documentation: https://huggingface.co/docs/transformers

**Libraries used:**
- LangChain text splitters: https://python.langchain.com/docs/how_to/recursive_text_splitter/
- Pydantic: https://docs.pydantic.dev
- Python Fire: https://github.com/google/python-fire

**How AI was used in this project:**

Claude (Anthropic) was used for resolving disk space issues on 42 machines by redirecting caches to `/sgoinfre`, and generating this README. All generated code was reviewed, tested, and understood before integration.