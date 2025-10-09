# UNIDOC-BENCH

A unified benchmark for document-centric multimodal retrieval-augmented generation (MM-RAG). This project provides the first large-scale, realistic benchmark for MM-RAG built from 70k real-world PDF pages across eight domains, with tools for document tagging, dataset synthesis, baseline implementations, and evaluation frameworks.

## Overview

UNIDOC-BENCH is designed to evaluate and benchmark multimodal document understanding systems across various domains including healthcare, finance, legal, education, and more. The benchmark extracts and links evidence from text, tables, and figures, then generates 1,600 multimodal QA pairs spanning factual retrieval, comparison, summarization, and logical reasoning queries. It supports apples-to-apples comparison across four paradigms: (1) text-only, (2) image-only, (3) multimodal text-image fusion, and (4) multimodal joint retrieval.

## Key Findings

Based on the comprehensive evaluation in the [UNIDOC-BENCH paper](https://arxiv.org/abs/2510.03663), several important insights emerge:

### 🎯 **Performance Insights**
- **Multimodal text-image fusion RAG systems consistently outperform** both unimodal and jointly multimodal embedding-based retrieval
- **Neither text nor images alone are sufficient** for optimal document understanding
- **Current multimodal embeddings remain inadequate** for complex document-centric tasks

### 📊 **Benchmark Characteristics**
- **70,000 real-world PDF pages** across 8 diverse domains
- **1,600 multimodal QA pairs** with 20% expert validation
- **Four query types**: factual retrieval, comparison, summarization, and logical reasoning
- **Unified evaluation protocol** with standardized candidate pools, prompts, and metrics

### 🔍 **Analysis Capabilities**
- **When and how visual context complements textual evidence**
- **Systematic failure mode identification**
- **Actionable guidance for developing robust MM-RAG pipelines**

## Project Structure

```
UNIDOC-BENCH/
├── src/
│   ├── vllm_tagging.py          # Document tagging and classification
│   ├── qa_synthesize/           # QA dataset synthesis pipeline
│   ├── baselines/               # Baseline RAG implementations
│   ├── evaluation/              # Evaluation metrics and tools
│   └── tagging_prompts/         # Prompts for document tagging
├── data/
│   └── QA/filtered/             # Filtered QA datasets by domain
└── docs/                        # Documentation and examples
```

## Main Components

### 1. Document Tagging (`src/vllm_tagging.py`)

Automated document tagging and classification system that processes PDF documents and extracts:
- **Domain classification** (healthcare, finance, legal, etc.)
- **Language detection**
- **Date extraction**
- **Modality identification** (text, images, tables)
- **Format analysis**

**Key Features:**
- Multi-modal document processing using VLLM
- Layout analysis with LayoutParser
- Support for both single PDFs and batch processing
- Integration with Qwen-VL models

### 2. Dataset Synthesis (`src/qa_synthesize/`)

Complete pipeline for generating high-quality question-answer pairs from document collections:

**Pipeline Steps:**
1. **Knowledge Graph Creation** (`1_kg_create.py`) - Build knowledge graphs from documents
2. **QA Synthesis** (`2_qa_synthesize.py`) - Generate QA pairs using RAGAS framework
3. **Quality Filtering** (`3_filter_qa.py`) - Filter out low-quality pairs
4. **Similarity Filtering** (`4_filter_similarities.py`) - Remove duplicate/similar questions
5. **Balancing** (`5_balance.py`) - Balance question types and difficulty
6. **Rewriting** (`6_rewriting.py`) - Improve question quality and diversity

**Supported Domains:**
- Healthcare
- Finance
- Legal
- Education
- Energy
- Construction
- Commerce & Manufacturing
- CRM

**Dataset Statistics:**
- 70,000 real-world PDF pages
- 1,600 multimodal QA pairs
- 20% validated by multiple annotators and expert adjudication
- Covers factual retrieval, comparison, summarization, and logical reasoning queries

### 3. Baseline Implementations (`src/baselines/`)

Multiple RAG baseline implementations supporting four paradigms:

- **Text-only RAG** (`simple_rag.py`) - Traditional text-based retrieval
- **Image-only RAG** (`image_rag.py`) - Visual document retrieval
- **Multimodal Text-Image Fusion** (`image_text_rag.py`) - Combined text and image retrieval
- **Multimodal Joint Retrieval** - Joint embedding-based retrieval
- **Gemini Integration** (`gemini_call.py`) - Google Gemini model integration
- **Voyage Embeddings** (`image_rag_voyage.py`) - Voyage AI embeddings

**Supported Models:**
- GPT-4, GPT-3.5
- Gemini Pro
- Qwen-VL
- Various embedding models (OpenAI, Voyage, etc.)

### 4. Evaluation Framework (`src/evaluation/`)

Comprehensive evaluation suite using RAGAS metrics:

- **Correctness Evaluation** (`correctness.py`) - Answer accuracy assessment
- **Retrieval Evaluation** (`retrieval_eval.py`) - Retrieval quality metrics
- **RAGAS Integration** (`evaluation_ragas.py`) - Full RAGAS evaluation pipeline
- **Analysis Tools** (`analysis.py`) - Results analysis and visualization

**Evaluation Metrics:**
- Answer Correctness
- Context Precision
- Context Recall
- Faithfulness
- Answer Relevancy

## Quick Start

### Prerequisites

```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment variables
export OPENAI_API_KEY="your-openai-key"
export GOOGLE_API_KEY="your-google-key"  # For Gemini
```

### 1. Document Tagging

```bash
python src/vllm_tagging.py \
    --input_path /path/to/documents \
    --output_path /path/to/tagged_output \
    --model_name "Qwen/Qwen-VL-Chat"
```

### 2. Dataset Synthesis

```bash
# Run the complete synthesis pipeline
bash src/qa_synthesize/scripts/create_dataset.sh
```

### 3. Run Baselines

```bash
# Text RAG baseline
bash src/baselines/scripts/text_rag.sh

# Multimodal RAG baseline
bash src/baselines/scripts/img_text_rag.sh
```

### 4. Evaluation

```bash
# Run end-to-end evaluation
bash src/evaluation/scripts/eval_e2e.sh
```

## Configuration

### Environment Setup

1. **Conda Environment:**
```bash
conda create -n unidoc-bench python=3.9
conda activate unidoc-bench
pip install -r requirements.txt
```

2. **API Keys:**
```bash
export OPENAI_API_KEY="your-key"
export GOOGLE_API_KEY="your-key"
export VOYAGE_API_KEY="your-key"  # Optional
```

### Data Preparation

1. **Document Structure:**
```
data/
├── final_database/
│   ├── healthcare_database/     # Text chunks
│   ├── healthcare_elements/     # Visual elements
│   └── ...
└── QA/
    └── filtered/               # Processed QA datasets
```

## Usage Examples

### Document Processing

```python
from src.vllm_tagging import process_documents

# Process a single PDF
results = process_documents(
    input_path="document.pdf",
    model_name="Qwen/Qwen-VL-Chat"
)

# Process a folder of documents
results = process_documents(
    input_path="documents/",
    model_name="Qwen/Qwen-VL-Chat",
    batch_size=10
)
```

### QA Generation

```python
from src.qa_synthesize.kg_create import create_knowledge_graph

# Create knowledge graph
kg = create_knowledge_graph(
    name_str="healthcare",
    database_path="data/final_database/healthcare_database"
)
```

### Evaluation

```python
from src.evaluation.evaluation_ragas import evaluate_ragas

# Evaluate a RAG system
results = evaluate_ragas(
    input_file="results.json",
    output_file="evaluation.json",
    testsize=500
)
```

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Citation

If you use UNIDOC-BENCH in your research, please cite:

```bibtex
@article{peng2025unidoc,
  title={UNIDOC-BENCH: A Unified Benchmark for Document-Centric Multimodal RAG},
  author={Peng, Xiangyu and Qin, Can and Chen, Zeyuan and Xu, Ran and Xiong, Caiming and Wu, Chien-Sheng},
  journal={arXiv preprint arXiv:2510.03663},
  year={2025}
}
```

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE.txt) file for details.

## Support

For questions and support:
- Open an issue on GitHub
- Contact: [your-email@domain.com]

## Acknowledgments

- Built on top of RAGAS framework
- Uses LlamaIndex for RAG implementations
- Integrates with various LLM providers (OpenAI, Google, etc.)

