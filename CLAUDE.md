# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

UniDoc-Bench is a unified benchmark for document-centric multimodal retrieval-augmented generation (MM-RAG). It evaluates multimodal document understanding systems across 8 domains (healthcare, finance, legal, education, energy, construction, commerce/manufacturing, CRM) using 70k real-world PDF pages and 1,600 multimodal QA pairs.

## Key Commands

### Environment Setup
```bash
conda create -n unidoc-bench python=3.9
conda activate unidoc-bench
pip install -r requirements.txt  # Note: requirements.txt may need to be created
```

Required API keys:
```bash
export OPENAI_API_KEY="your-openai-key"
export GOOGLE_API_KEY="your-google-key"  # For Gemini
export VOYAGE_API_KEY="your-key"  # Optional
```

### Document Tagging
```bash
python src/vllm_tagging.py \
    --input_path /path/to/documents \
    --output_path /path/to/tagged_output \
    --model_name "Qwen/Qwen-VL-Chat"
```

### QA Dataset Synthesis Pipeline
```bash
# Run complete 6-stage pipeline
bash src/qa_synthesize/scripts/create_dataset.sh

# Individual stages (from src/qa_synthesize/):
python 1_kg_create.py --name_str <domain> --database_path <path>
python 2_qa_synthesize.py --folder_elements <path> --name_str <domain> --testset_size <size> --output_file <file>
python 3_filter_qa.py --qa_path <file> --folder_elements <path>
python 4_filter_similarities.py --file_path <file>
python 5_balance.py --file_path <file> --max_diff 0
python 6_rewriting.py --folder_elements <path> --file_path <file> --file_path_save <file> --mode full
```

### Running Baselines
```bash
bash src/baselines/scripts/text_rag.sh           # Text-only RAG
bash src/baselines/scripts/img_rag.sh            # Image-only RAG
bash src/baselines/scripts/img_text_rag.sh       # Multimodal text-image fusion
bash src/baselines/scripts/voyage_rag.sh         # Voyage AI embeddings
```

### Evaluation
```bash
bash src/evaluation/scripts/eval_e2e.sh
# Or directly:
python src/evaluation/evaluation_ragas.py --input_file <file> --output_file <file> --testsize 500
```

## Architecture

### 1. Document Tagging (`src/vllm_tagging.py`, `src/icl_tagging.py`, `src/pdf_layout_parser.py`)
- Uses VLLM with Qwen-VL models for multimodal document processing
- LayoutParser for PDF layout analysis
- Ray for distributed processing
- Extracts: domain, language, date, modality (text/images/tables)

### 2. QA Synthesis Pipeline (`src/qa_synthesize/`)
Six-stage pipeline for generating QA pairs:
- **1_kg_create.py**: Knowledge graph creation from documents
- **2_qa_synthesize.py**: Core QA generation using RAGAS framework
- **3_filter_qa.py**: Quality filtering
- **4_filter_similarities.py**: Deduplication
- **5_balance.py**: Question type balancing
- **6_rewriting.py**: Question improvement

### 3. Baseline Implementations (`src/baselines/`)
Four RAG paradigms:
- **simple_rag.py**: Text-only retrieval
- **image_rag.py**: Image-only retrieval
- **image_text_rag.py**: Multimodal text-image fusion (recommended)
- **image_rag_gme.py**: Gemini integration
- **image_rag_voyage.py**: Voyage AI embeddings

### 4. Evaluation Framework (`src/evaluation/`)
- **evaluation_ragas.py**: Main evaluation using RAGAS metrics
- **retrieval_eval.py**: Retrieval quality metrics
- **correctness.py**: Answer accuracy

## Important Implementation Notes

### External Data Compatibility
When working with external retrieval data (e.g., from keyword search), the `chunk_match_back` function in `2_qa_synthesize.py` handles missing `source` fields by falling back to `document_keyword` field. This is critical for image/table loading in QA synthesis.

### Answer Types in QA Synthesis
The system supports four answer types, each with configurable visitation patterns:
- `text_as_answer`: Text-only answers
- `image_as_answer`: Image-only answers
- `table_as_answer`: Table-only answers
- `image_plus_text_as_answer`: Combined image and text answers

Configurable via arguments like `--different_file_text_as_answer different_files_visited`

### Supported Models
- LLMs: GPT-4, GPT-3.5, Gemini Pro, Qwen-VL
- Embeddings: OpenAI Embeddings, Voyage AI
- RAG Framework: LlamaIndex
- Evaluation: RAGAS

## Data Structure

```
data/
├── final_database/
│   ├── <domain>/           # PDF files per domain (download separately from HuggingFace)
│   └── <domain>_database/  # Processed chunks
├── QA/
│   └── filtered/           # QA datasets by domain
└── elements/               # Extracted figures/tables metadata
```

## Test Mode
QA synthesis supports test mode for debugging:
```bash
python src/qa_synthesize/2_qa_synthesize.py \
    --test mode02 \
    --chunks_json_path <external_retrieval.json> \
    --debug
```

## Recent Changes
- Fixed compatibility with external retrieval data when `source` field is missing (uses `document_keyword` fallback)
- Added test mode support for different answer types (image, text, table, image+text)

## License
CC BY-NC 4.0 - This dataset was generated using GPT-4.1 and should not be used to develop models that compete with OpenAI.
