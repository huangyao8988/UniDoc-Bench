"""
Test script for test mode 01 functionality.
This script tests the chunks library loading and grouping without requiring full dependencies.
"""

import json
import os
import sys

# Add qa_synthesize to path
sys.path.insert(0, os.path.dirname(__file__))


def classify_chunk_type(chunk):
    """Classify chunk type based on doc_type_kwd and content."""
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
    """Adapt JSON chunk metadata to format expected by chunk_match_back()."""
    filename = chunk.get("document_keyword", "unknown")
    doc_id = chunk.get("document_id", "unknown")
    chunk_id = chunk.get("id", "unknown")

    source = f"{filename}_id_{doc_id}_pg_{chunk_id}.txt"

    return {
        "source": source,
        "chunk_id": chunk_id,
        "document_id": doc_id,
        "image_id": chunk.get("image_id", ""),
        "document_keyword": filename
    }


def form_chunk_groups_for_answer_type(query_chunks, answer_type, chunk_count, file_constraint):
    """Form chunk groups for a specific answer type from a single query result."""
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


def load_chunks_lib(chunks_lib_path):
    """Load chunks from JSON library file."""
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


def load_chunks_from_lib(chunks_lib, distribution, args_dict):
    """Load chunks from JSON chunks library based on distribution and constraints."""

    class Args:
        def __init__(self, d):
            for k, v in d.items():
                setattr(self, k, v)

    args = Args(args_dict)

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


def choose_distribution(testset_size):
    """Choose answer type distribution."""
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


def run_test_mode_01(chunks_all, chunks_metadata_all):
    """Run test mode 01: display statistics and output chunks to files."""
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


if __name__ == "__main__":
    print("Test Mode 01 - Chunks Library Test")
    print("=" * 50)

    # Find test file
    test_lib_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "batch_retrieve_results_test.json"
    )

    if not os.path.exists(test_lib_path):
        print(f"Test file not found: {test_lib_path}")
        sys.exit(1)

    print(f"Loading chunks from: {test_lib_path}")

    # Load chunks library
    chunks_lib = load_chunks_lib(test_lib_path)
    print(f"Loaded {len(chunks_lib)} query results")

    # Set distribution
    answer_type_distribution = choose_distribution(10)
    print(f"Distribution: {answer_type_distribution}")

    # Set args
    args_dict = {
        'image_as_answer_count': 2,
        'table_as_answer_count': 1,
        'text_as_answer_count': 2,
        'image_plus_text_as_answer_count': 3,
        'different_file_image_as_answer': None,
        'different_file_table_as_answer': None,
        'different_file_text_as_answer': None,
        'different_file_image_plus_text_as_answer': None,
    }

    # Load chunks from lib
    chunks_all, chunks_metadata_all, chunks_overlapped_items = load_chunks_from_lib(
        chunks_lib,
        answer_type_distribution,
        args_dict
    )

    # Run test mode 01
    run_test_mode_01(chunks_all, chunks_metadata_all)

    print("\n" + "=" * 50)
    print("Test Mode 01 completed successfully!")
    print(f"Output files saved in: {os.path.join(os.getcwd(), 'test_mode_01_output')}")
