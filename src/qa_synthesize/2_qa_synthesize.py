import os
import ast
import json
import time
import base64
import argparse
from tqdm import tqdm
import copy
from openai import OpenAI
from ragas.testset.graph import KnowledgeGraph
from utils import flatten_unique_ignore_case
from chunks_extraction import chunk_match_back, extract_relevant_chunks, extract_chunks
from prompts.query_syn_prompt import obtain_user_prompt
from prompts.templates import CHOOSE_TEMPLATE_PROMPT_USER, MESSAGE_WITH_EXAMPLE, choose_fixed_templates

MESSAGE_WITH_EXAMPLE_ONCE = copy.deepcopy(MESSAGE_WITH_EXAMPLE)
MODEL = "gpt-4o"
# Note: client initialization moved to after argument parsing for CLI fallback support


def initialize_openai_client(args):
    """
    Initialize OpenAI client with environment variables and CLI fallback.
    Priority: env var > CLI arg

    Args:
        args: Parsed command-line arguments

    Returns:
        Initialized OpenAI client
    """
    # Get API key with priority: env var > CLI arg
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key and hasattr(args, 'openai_api_key') and args.openai_api_key:
        api_key = args.openai_api_key

    if not api_key:
        raise ValueError("OPENAI_API_KEY not set. Set environment variable or use --openai_api_key")

    # Get base URL with priority: env var > CLI arg
    base_url = os.environ.get("OPENAI_BASE_URL")
    if not base_url and hasattr(args, 'openai_base_url') and args.openai_base_url:
        base_url = args.openai_base_url

    # Initialize client
    client_kwargs = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url

    return OpenAI(**client_kwargs)


def load_chunks_lib(chunks_lib_path):
    """
    Load chunks from JSON library file.

    Args:
        chunks_lib_path: Path to batch_retrieve_results.json

    Returns:
        List of query result dictionaries with chunks
    """
    with open(chunks_lib_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Validate structure
    if "batch_results" not in data:
        raise ValueError("Invalid chunks library format: missing 'batch_results'")

    chunks_lib = []
    for result in data["batch_results"]:
        if result.get("code") == 0 and "data" in result and "chunks" in result["data"]:
            chunks_lib.append({
                "query_question": result.get("query_question", ""),
                "chunks": result["data"]["chunks"]
            })

    return chunks_lib


def classify_chunk_type(chunk):
    """
    Classify chunk type based on doc_type_kwd and content.

    Args:
        chunk: Chunk dictionary

    Returns:
        One of: "text", "image", "table"
    """
    doc_type = chunk.get("doc_type_kwd", "")
    content = chunk.get("content", "")

    if doc_type == "":
        return "text"
    elif doc_type == "image":
        # Check if content contains <table> marker
        if "<table>" in content.lower():
            return "table"
        else:
            return "image"
    else:
        # Default to text for unknown types
        return "text"


def get_min_chunk_count(answer_type):
    """Get minimum chunk count for answer type."""
    min_counts = {
        "image_as_answer": 1,
        "table_as_answer": 1,
        "text_as_answer": 1,
        "image_plus_text_as_answer": 2
    }
    return min_counts.get(answer_type, 1)


def has_both_text_and_image(chunks):
    """Check if group has both text and image chunks (no tables)."""
    has_text = False
    has_image = False
    for chunk in chunks:
        chunk_type = classify_chunk_type(chunk)
        if chunk_type == "text":
            has_text = True
        elif chunk_type == "image":
            has_image = True
    return has_text and has_image


def adapt_chunk_metadata(chunk):
    """
    Adapt JSON chunk metadata to format expected by chunk_match_back().

    Args:
        chunk: Chunk from JSON library

    Returns:
        Metadata dictionary compatible with chunk_match_back()
    """
    filename = chunk.get("document_keyword", "unknown")
    doc_id = chunk.get("document_id", "unknown")
    chunk_id = chunk.get("id", "unknown")

    # Create source path in expected format
    # Note: page info not available in JSON, using chunk_id as approximation
    source = f"{filename}_id_{doc_id}_pg_{chunk_id}.txt"

    return {
        "source": source,
        "chunk_id": chunk_id,
        "document_id": doc_id,
        "image_id": chunk.get("image_id", ""),
        "document_keyword": filename
    }


def form_chunk_groups_for_answer_type(query_chunks, answer_type, chunk_count, file_constraint):
    """
    Form chunk groups for a specific answer type from a single query result.

    Args:
        query_chunks: List of chunks from a single query
        answer_type: One of the four answer types
        chunk_count: Desired number of chunks per group
        file_constraint: None, "different_files_visited_random", or "different_files_visited"

    Returns:
        List of chunk groups (each group is a list of chunks) - max 1 group per query
    """
    import random

    # Filter chunks by answer type
    filtered_chunks = []
    for chunk in query_chunks:
        chunk_type = classify_chunk_type(chunk)

        if answer_type == "image_as_answer" and chunk_type == "image":
            filtered_chunks.append(chunk)
        elif answer_type == "table_as_answer" and chunk_type == "table":
            filtered_chunks.append(chunk)
        elif answer_type == "text_as_answer" and chunk_type == "text":
            filtered_chunks.append(chunk)
        elif answer_type == "image_plus_text_as_answer":
            # Need both text and images (no tables)
            if chunk_type in ["text", "image"]:
                filtered_chunks.append(chunk)

    groups = []
    min_count = get_min_chunk_count(answer_type)

    if file_constraint is None or file_constraint == "same_file":
        # Group chunks from same file
        file_groups = {}
        for chunk in filtered_chunks:
            doc_id = chunk.get("document_id", "unknown")
            if doc_id not in file_groups:
                file_groups[doc_id] = []
            file_groups[doc_id].append(chunk)

        # Create groups of specified size
        for doc_id, chunks in file_groups.items():
            if len(groups) >= 1:
                break
            for i in range(0, len(chunks), chunk_count):
                group = chunks[i:i+chunk_count]
                if len(group) >= min_count:
                    if answer_type == "image_plus_text_as_answer":
                        if has_both_text_and_image(group):
                            groups.append(group)
                            break
                    else:
                        groups.append(group)
                        break

    elif file_constraint == "different_files_visited":
        # All chunks must be from different files
        file_used = set()
        current_group = []

        for chunk in filtered_chunks:
            if len(groups) >= 1:
                break
            doc_id = chunk.get("document_id", "unknown")
            if doc_id not in file_used:
                current_group.append(chunk)
                file_used.add(doc_id)

                if len(current_group) == chunk_count:
                    if answer_type == "image_plus_text_as_answer":
                        if has_both_text_and_image(current_group):
                            groups.append(current_group)
                            break
                    else:
                        groups.append(current_group)
                        break

    elif file_constraint == "different_files_visited_random":
        # Mix of same and different files
        random.shuffle(filtered_chunks)

        for i in range(0, len(filtered_chunks), chunk_count):
            if len(groups) >= 1:
                break
            group = filtered_chunks[i:i+chunk_count]
            if len(group) >= min_count:
                if answer_type == "image_plus_text_as_answer":
                    if has_both_text_and_image(group):
                        groups.append(group)
                        break
                else:
                    groups.append(group)
                    break

    # Return at most 1 group per query
    return groups[:1] if groups else []


def load_chunks_from_lib(chunks_lib, distribution, args):
    """
    Load chunks from JSON chunks library based on distribution and constraints.

    Args:
        chunks_lib: Loaded chunks library from load_chunks_lib()
        distribution: Dictionary mapping answer types to desired counts
        args: Command-line arguments with constraints

    Returns:
        Tuple of (chunks_all, chunks_metadata_all, chunks_overlapped_items)
    """
    chunks_all = {}
    chunks_metadata_all = {}
    chunks_overlapped_items = {}

    # Define answer types and their configurations
    answer_type_configs = {
        "image_as_answer": {
            "chunk_count": getattr(args, 'image_as_answer_count', 2),
            "file_constraint": getattr(args, 'different_file_image_as_answer', None),
        },
        "table_as_answer": {
            "chunk_count": getattr(args, 'table_as_answer_count', 1),
            "file_constraint": getattr(args, 'different_file_table_as_answer', None),
        },
        "text_as_answer": {
            "chunk_count": getattr(args, 'text_as_answer_count', 2),
            "file_constraint": getattr(args, 'different_file_text_as_answer', None),
        },
        "image_plus_text_as_answer": {
            "chunk_count": getattr(args, 'image_plus_text_as_answer_count', 3),
            "file_constraint": getattr(args, 'different_file_image_plus_text_as_answer', None),
        }
    }

    # Process each answer type
    for answer_type, config in answer_type_configs.items():
        chunk_groups = []
        chunk_metadata_groups = []

        target_count = distribution.get(answer_type, 0)

        # Iterate through query results
        for query_result in chunks_lib:
            if len(chunk_groups) >= target_count:
                break

            query_chunks = query_result["chunks"]

            # Form groups for this answer type
            groups = form_chunk_groups_for_answer_type(
                query_chunks,
                answer_type,
                config["chunk_count"],
                config["file_constraint"]
            )

            # Convert to expected format
            for group in groups:
                if len(chunk_groups) >= target_count:
                    break

                # Extract content and metadata
                chunk_contents = [chunk["content"] for chunk in group]
                chunk_metadatas = [adapt_chunk_metadata(chunk) for chunk in group]

                chunk_groups.append(chunk_contents)
                chunk_metadata_groups.append(chunk_metadatas)

        chunks_all[answer_type] = chunk_groups
        chunks_metadata_all[answer_type] = chunk_metadata_groups
        chunks_overlapped_items[answer_type] = ["" for _ in chunk_groups]

    return chunks_all, chunks_metadata_all, chunks_overlapped_items


def parse_qa_types(qa_types_str):
    """
    Parse QA types filter string.

    Args:
        qa_types_str: Comma-separated string of answer types or None

    Returns:
        Set of answer types to process, or None for all
    """
    if not qa_types_str:
        return None

    valid_types = {"image_as_answer", "table_as_answer", "text_as_answer", "image_plus_text_as_answer"}
    requested_types = set(t.strip() for t in qa_types_str.split(","))

    # Validate
    invalid = requested_types - valid_types
    if invalid:
        raise ValueError(f"Invalid QA types: {invalid}. Valid types: {valid_types}")

    return requested_types


def run_test_mode_01(chunks_all, chunks_metadata_all):
    """
    Run test mode 01: display statistics and output chunks to files.

    Args:
        chunks_all: Dictionary of chunks by answer type
        chunks_metadata_all: Dictionary of chunk metadata by answer type
    """
    import os

    # Create test output directory
    test_output_dir = "test_mode_01_output"
    os.makedirs(test_output_dir, exist_ok=True)

    # Statistics for each answer type
    stats = {}

    for answer_type in ["image_as_answer", "table_as_answer", "text_as_answer", "image_plus_text_as_answer"]:
        chunks = chunks_all.get(answer_type, [])
        chunks_metadata = chunks_metadata_all.get(answer_type, [])

        # Calculate statistics
        total_groups = len(chunks)
        total_chunks = sum(len(group) for group in chunks)

        # Count by file
        file_counts = {}
        for group in chunks_metadata:
            files = set()
            for meta in group:
                doc_id = meta.get("document_id", "unknown")
                files.add(doc_id)
            for doc_id in files:
                file_counts[doc_id] = file_counts.get(doc_id, 0) + 1

        stats[answer_type] = {
            "total_groups": total_groups,
            "total_chunks": total_chunks,
            "unique_files": len(file_counts),
            "avg_chunks_per_group": total_chunks / total_groups if total_groups > 0 else 0
        }

        # Output chunks to file
        output_file = os.path.join(test_output_dir, f"{answer_type}_chunks.jsonl")
        with open(output_file, 'w', encoding='utf-8') as f:
            for chunk_group, metadata_group in zip(chunks, chunks_metadata):
                output_data = {
                    "answer_type": answer_type,
                    "chunks": chunk_group,
                    "metadata": metadata_group
                }
                f.write(json.dumps(output_data, ensure_ascii=False) + "\n")

        print(f"\n{answer_type}:")
        print(f"  Total groups: {stats[answer_type]['total_groups']}")
        print(f"  Total chunks: {stats[answer_type]['total_chunks']}")
        print(f"  Unique files: {stats[answer_type]['unique_files']}")
        print(f"  Avg chunks per group: {stats[answer_type]['avg_chunks_per_group']:.2f}")
        print(f"  Output file: {output_file}")

    # Save summary statistics
    stats_file = os.path.join(test_output_dir, "statistics.json")
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"\nSummary statistics saved to: {stats_file}")


def choose_distribution(testset_size):
    answer_types = [
        "image_as_answer",
        "image_plus_text_as_answer",
        "text_as_answer",
        "table_as_answer",
    ]

    base_size = testset_size // 4
    remainder = testset_size % 4

    distribution = {atype: base_size for atype in answer_types}

    for i in range(remainder):
        distribution[answer_types[i]] += 1

    distribution["image_as_answer"] *= 3  # image_only is HARD to obtain

    return distribution


def load_chunks(
    loaded_kg,
    distribution,
    different_files_in_cluster,
    different_files_visited,
    no_tab_in_chunk_text,
    no_tab_in_chunk_img
):
    chunks_all, chunks_metadata_all, chunks_overlapped_items = dict(), dict(), dict()

    chunks, chunks_metadata = extract_chunks(
        loaded_kg,
        image_num_min=1,
        image_num_max=4,  # was 2
        different_file=False, #different_files_visited,
        table_num_min=0,
        table_num_max=2,
    )
    chunks_all["image_as_answer"] = chunks[:distribution["image_as_answer"]]
    chunks_metadata_all["image_as_answer"] = chunks_metadata[
        : distribution["image_as_answer"]
    ]
    chunks_overlapped_items["image_as_answer"] = [
        "" for _ in range(len(chunks[: distribution["image_as_answer"]]))
    ]

    chunks, chunks_metadata = extract_chunks(
        loaded_kg,
        image_num_min=0,
        image_num_max=0,
        different_file=different_files_visited,
        table_num_min=1,
        table_num_max=2,
    )
    chunks_all["table_as_answer"] = chunks[:distribution["table_as_answer"]]
    chunks_metadata_all["table_as_answer"] = chunks_metadata[
        : distribution["table_as_answer"]
    ]
    chunks_overlapped_items["table_as_answer"] = [
        "" for _ in range(len(chunks[: distribution["table_as_answer"]]))
    ]

    chunks, chunks_metadata, chunks_overlapped_item = extract_relevant_chunks(
        None,
        loaded_kg,
        overlapping_items_minimum=2,
        testset_size=distribution["text_as_answer"]*2,
        different_files=different_files_in_cluster,
        image_number_min=0,
        image_number_max=2,
        page_diff=1,
        different_files_visited=different_files_visited,
        no_tab=no_tab_in_chunk_text,
        table_number_min=0,
        table_number_max=0,
    )
    chunks_all["text_as_answer"] = chunks[: distribution["text_as_answer"]]
    chunks_metadata_all["text_as_answer"] = chunks_metadata[
        : distribution["text_as_answer"]
    ]
    chunks_overlapped_items["text_as_answer"] = chunks_overlapped_item[
        : distribution["text_as_answer"]
    ]

    chunks, chunks_metadata, chunks_overlapped_item = extract_relevant_chunks(
        None,
        loaded_kg,
        overlapping_items_minimum=1,
        testset_size=distribution["image_plus_text_as_answer"],
        different_files=different_files_in_cluster,
        image_number_min=1,
        image_number_max=2,
        page_diff=1,
        different_files_visited=different_files_visited,
        no_tab=no_tab_in_chunk_img,
        table_number_min=0,
        table_number_max=0,
    )
    chunks_all["image_plus_text_as_answer"] = chunks[
        : distribution["image_plus_text_as_answer"]
    ]
    chunks_metadata_all["image_plus_text_as_answer"] = chunks_metadata[
        : distribution["image_plus_text_as_answer"]
    ]
    chunks_overlapped_items["image_plus_text_as_answer"] = chunks_overlapped_item[
        : distribution["image_plus_text_as_answer"]
    ]

    return chunks_all, chunks_metadata_all, chunks_overlapped_items


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def choose_templates(chunks, chunks_metadata, domain_name):
    messages = copy.deepcopy(MESSAGE_WITH_EXAMPLE_ONCE)
    user_prompt = [
        {
            "type": "text",
            "text": CHOOSE_TEMPLATE_PROMPT_USER.replace(
                "{text_contexts}", json.dumps(chunks, indent=4)).replace(
                "{{TEMPLATES}}", choose_fixed_templates(domain_name)
            )
            + "\n\nThese are the tables and images in the above chunks:",
        }
    ]
    images = {}
    for chunk, chunk_metadata in zip(chunks, chunks_metadata):
        _, img = chunk_match_back(chunk, chunk_metadata, args.folder_elements)
        for im, p in img.items():
            images[f"<<fig-{im}>>"] = [encode_image(p), p]

    for fig, img in images.items():
        user_prompt += [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{img[0]}",
                    "name": f"This is the image for {fig} in the above context.",
                },
            }
        ]
    messages.append({"role": "user", "content": user_prompt})
    templates = None
    for j in range(3):
        try:
            response = (
                client.chat.completions.create(
                    model=MODEL,
                    messages=messages,
                    temperature=0.85,
                )
                .choices[0]
                .message.content
            )
            response = response.replace("```json", "```").split("```")[1]
            templates = ast.literal_eval(response)
            break
        except Exception as e:
            time.sleep(j+1)
    return templates


def build_prompt(chunks, chunks_metadata, hint, answer_type, query_type, templates):
    images = {}
    tables = {}

    for chunk, chunk_metadata in zip(chunks, chunks_metadata):
        tab, img = chunk_match_back(chunk, chunk_metadata, args.folder_elements)
        for t, p in tab.items():
            tables[f"<<tab-{t}>>"] = [encode_image(p), p]
        for im, p in img.items():
            images[f"<<fig-{im}>>"] = [encode_image(p), p]

    combined_chunk_message = "\n\n".join(
        [f"**Chunk {i}:**\n\n{chunk}" for i, chunk in enumerate(chunks, start=1)]
    )

    user_prompt = [{"type": "text", "text": combined_chunk_message}]

    for tab, img in tables.items():
        user_prompt += [
            {
                "type": "text",
                "text": f"Below is the image for the TABLE: {tab} in the above context.",
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{img[0]}",
                    "name": f"This is the image for the TABLE: {tab} in the above context.",
                },
            },
        ]

    for fig, img in images.items():
        user_prompt += [
            {
                "type": "text",
                "text": f"Below is the image for the FIGURE: {fig} in the above context.",
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{img[0]}",
                    "name": f"This is the image for the FIGURE: {fig} in the above context.",
                },
            },
        ]
    messages = []
    for template in templates:
        template = json.dumps(template, indent=4)
        user_query = obtain_user_prompt(answer_type, template, query_type).replace(
            "{{hints}}", "/ ".join(hint)
        )

        user_prompt_each = user_prompt + [{"type": "text", "text": user_query}]

        messages.append(
            [
                {
                    "role": "system",
                    "content": "You are a helpful assistant that responds Python dictionary.",
                },
                {"role": "user", "content": user_prompt_each},
            ]
        )

    return messages, tables, images


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Original arguments
    parser.add_argument(
        "--name_str", type=str, required=True, help="name of the subdomain"
    )
    parser.add_argument(
        "--folder_elements", type=str, required=True, help="Database folder elements"
    )
    parser.add_argument("--output_file", type=str, required=True, help="output_file")
    parser.add_argument("--testset_size", type=int, required=True, help="# of datasets")
    parser.add_argument(
        "--different_files_in_cluster", action="store_true", help="# of datasets"
    )
    parser.add_argument(
        "--different_files_visited", action="store_true", help="# of datasets"
    )
    parser.add_argument("--no_tab_in_chunk_text", action="store_true", help="# of datasets")
    parser.add_argument("--no_tab_in_chunk_img", action="store_true", help="# of datasets")
    parser.add_argument("--domain_name", type=str, required=True)

    # NEW: Chunks library path
    parser.add_argument("--chunks_lib", type=str, default=None,
                        help="Path to chunks library JSON file (e.g., batch_retrieve_results.json)")

    # NEW: Chunk count controls
    parser.add_argument("--image_as_answer_count", type=int, default=2,
                        help="Number of chunks for image_as_answer (1-4, default: 2)")
    parser.add_argument("--table_as_answer_count", type=int, default=1,
                        help="Number of chunks for table_as_answer (1-2, default: 1)")
    parser.add_argument("--text_as_answer_count", type=int, default=2,
                        help="Number of chunks for text_as_answer (1-4, default: 2)")
    parser.add_argument("--image_plus_text_as_answer_count", type=int, default=3,
                        help="Number of chunks for image_plus_text_as_answer (2-5, default: 3)")

    # NEW: File diversity controls
    parser.add_argument("--different_file_image_as_answer", type=str, default=None,
                        choices=["different_files_visited_random", "different_files_visited"],
                        help="File diversity constraint for image_as_answer")
    parser.add_argument("--different_file_table_as_answer", type=str, default=None,
                        choices=["different_files_visited_random", "different_files_visited"],
                        help="File diversity constraint for table_as_answer")
    parser.add_argument("--different_file_text_as_answer", type=str, default=None,
                        choices=["different_files_visited_random", "different_files_visited"],
                        help="File diversity constraint for text_as_answer")
    parser.add_argument("--different_file_image_plus_text_as_answer", type=str, default=None,
                        choices=["different_files_visited_random", "different_files_visited"],
                        help="File diversity constraint for image_plus_text_as_answer")

    # NEW: QA generation control
    parser.add_argument("--qa_types", type=str, default=None,
                        help="Comma-separated list of QA types to generate (e.g., image_as_answer,table_as_answer)")

    # NEW: Test mode
    parser.add_argument("--test", type=str, default=None,
                        help="Test mode: 'mode01' for statistics and file output")

    # NEW: OpenAI API configuration (fallback when env vars not set)
    parser.add_argument("--openai_api_key", type=str, default=None,
                        help="OpenAI API Key (fallback when OPENAI_API_KEY env var not set)")
    parser.add_argument("--openai_base_url", type=str, default=None,
                        help="OpenAI Base URL (fallback when OPENAI_BASE_URL env var not set)")

    args = parser.parse_args()

    # NEW: Initialize OpenAI client with environment variables and CLI fallback
    client = initialize_openai_client(args)

    # Declare client as global for use in other functions
    globals()['client'] = client

    output_file = open(
        args.output_file,
        "a",
    )

    # NEW: Load chunks from JSON library or KnowledgeGraph (backward compatibility)
    if args.chunks_lib:
        # Use new JSON library loading
        chunks_lib = load_chunks_lib(args.chunks_lib)
        answer_type_distribution = choose_distribution(args.testset_size)
        chunks_all, chunks_metadata_all, chunks_overlapped_items = load_chunks_from_lib(
            chunks_lib,
            answer_type_distribution,
            args
        )
    else:
        # Use existing KnowledgeGraph loading (backward compatibility)
        kg_path = f"YOUR_QA_DATA_PATH/kg/{args.name_str}_database.json"  # e.g., "/path/to/qa/kg"
        loaded_kg = KnowledgeGraph.load(kg_path)
        answer_type_distribution = choose_distribution(args.testset_size)
        chunks_all, chunks_metadata_all, chunks_overlapped_items = load_chunks(
            loaded_kg,
            answer_type_distribution,
            args.different_files_in_cluster,
            args.different_files_visited,
            args.no_tab_in_chunk_text,
            args.no_tab_in_chunk_img
        )

    # NEW: Test mode 01 - run and exit
    if args.test == "mode01":
        output_file.close()
        run_test_mode_01(chunks_all, chunks_metadata_all)
        print("\nTest mode 01 completed. Exiting.")
        exit(0)

    # NEW: Parse QA types filter
    qa_types_filter = parse_qa_types(args.qa_types) if args.qa_types else None

    # QA generation loop
    lines = 0
    for answer_type in [
        "image_as_answer",
        "image_plus_text_as_answer",
        "text_as_answer",
        "table_as_answer",
    ]:
        # NEW: Skip if not in filter
        if qa_types_filter and answer_type not in qa_types_filter:
            continue

        chunks_answer, chunks_metadata_answer, hints_answer = (
            chunks_all[answer_type],
            chunks_metadata_all[answer_type],
            chunks_overlapped_items[answer_type],
        )

        for chunks, chunks_metadata, hints in tqdm(
            zip(chunks_answer, chunks_metadata_answer, hints_answer), desc=f"Generating for {answer_type}"
        ):
            hints = flatten_unique_ignore_case(hints)
            try:
                templates = choose_templates(chunks, chunks_metadata, args.domain_name)
                templates = [templates[-1]]
            except Exception as e:
                templates = []

            if not templates:
                continue

            messages_lst, tables, images = build_prompt(
                chunks,
                chunks_metadata,
                hints,
                answer_type,
                query_type=None,
                templates=templates,
            )

            for idx, messages in enumerate(messages_lst):

                try:
                    response = (
                        client.chat.completions.create(
                            model=MODEL,
                            messages=messages,
                            temperature=0.85,
                        )
                        .choices[0]
                        .message.content
                    )
                    response = (
                        response
                        if "```" not in response
                        else response.replace("```json", "```").split("```")[1]
                    )

                except Exception as error:
                    time.sleep(idx+1)

                try:
                    questions = ast.literal_eval(response.strip())
                    for element in questions["questions"]:
                        element["answer_type"] = answer_type
                        element["contexts"] = chunks
                        element["template"] = templates[idx]
                        element["chunks_metadata"] = chunks_metadata
                        element["hints"] = hints
                        element["tables"] = {t: p[1] for t, p in tables.items()}
                        element["images"] = {t: p[1] for t, p in images.items()}

                        output_file.write(json.dumps(element) + "\n")
                        output_file.flush()
                        lines += 1

                except Exception as error:
                    pass

    output_file.close()

