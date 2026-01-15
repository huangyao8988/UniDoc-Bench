"""
Import output.jsonl to Langfuse dataset.

This script reads a JSONL file containing QA pairs and imports them
into a Langfuse dataset for evaluation and experimentation.
"""

import os
import json
import argparse
import sys
from typing import Dict, Any, Optional


try:
    from langfuse import Langfuse
except ImportError:
    print("Error: langfuse package is not installed.")
    print("Please install it using: pip install langfuse")
    sys.exit(1)


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments following codebase patterns."""
    parser = argparse.ArgumentParser(
        description="Import output.jsonl to Langfuse dataset"
    )
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Path to output.jsonl file"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help="Name of Langfuse dataset to create/update"
    )
    parser.add_argument(
        "--description",
        type=str,
        default="Imported from UniDoc-Bench",
        help="Description of the dataset"
    )
    parser.add_argument(
        "--metadata",
        type=str,
        default="{}",
        help="Dataset metadata as JSON string"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode for verbose output"
    )
    return parser.parse_args()


def initialize_langfuse(debug: bool = False) -> Langfuse:
    """
    Initialize Langfuse client using environment variables.

    Args:
        debug: Enable debug mode for verbose output

    Returns:
        Langfuse client instance
    """
    public_key = os.environ.get("LANGFUSE_PUBLIC_KEY")
    secret_key = os.environ.get("LANGFUSE_SECRET_KEY")
    host = os.environ.get("LANGFUSE_HOST")

    if not public_key or not secret_key:
        print("Error: LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY environment variables must be set.")
        print("Please set them using:")
        print("  export LANGFUSE_PUBLIC_KEY='your-public-key'")
        print("  export LANGFUSE_SECRET_KEY='your-secret-key'")
        sys.exit(1)

    if debug:
        print(f"Initializing Langfuse client (host: {host or 'default'})")

    return Langfuse(
        public_key=public_key,
        secret_key=secret_key,
        host=host
    )


def create_or_get_dataset(
    langfuse: Langfuse,
    name: str,
    description: str,
    metadata: Dict[str, Any],
    debug: bool = False
) -> Any:
    """
    Create dataset or get existing one.

    Args:
        langfuse: Langfuse client instance
        name: Dataset name
        description: Dataset description
        metadata: Dataset metadata
        debug: Enable debug mode for verbose output

    Returns:
        Dataset object
    """
    try:
        # Try to get existing dataset
        dataset = langfuse.get_dataset(name=name)
        if debug:
            print(f"Using existing dataset: {name}")
        return dataset
    except Exception:
        # Dataset doesn't exist, create new one
        if debug:
            print(f"Creating new dataset: {name}")
        return langfuse.create_dataset(
            name=name,
            description=description,
            metadata=metadata
        )


def prepare_dataset_item(line_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert JSONL line to Langfuse dataset item format.

    Args:
        line_data: Dictionary containing data from a single JSONL line

    Returns:
        Dictionary with input, expected_output, and metadata fields
    """
    # Extract question and answer
    question = line_data.pop("question")
    answer = line_data.pop("answer")

    # All remaining fields become metadata
    metadata = line_data

    return {
        "input": {"question": question},
        "expected_output": {"answer": answer},
        "metadata": metadata
    }


def import_jsonl_to_langfuse(args: argparse.Namespace) -> Dict[str, Any]:
    """
    Main import function with streaming and error handling.

    Args:
        args: Parsed command line arguments

    Returns:
        Statistics dictionary with import results
    """
    # Initialize Langfuse client
    langfuse = initialize_langfuse(debug=args.debug)

    # Parse dataset metadata
    try:
        dataset_metadata = json.loads(args.metadata)
    except json.JSONDecodeError as e:
        print(f"Error parsing metadata JSON: {e}")
        dataset_metadata = {}

    # Create or get dataset
    dataset = create_or_get_dataset(
        langfuse,
        name=args.dataset_name,
        description=args.description,
        metadata=dataset_metadata,
        debug=args.debug
    )

    # Statistics tracking following codebase patterns
    stats = {
        "total": 0,
        "success": 0,
        "failed": 0,
        "errors": []
    }

    if args.debug:
        print(f"Starting import from: {args.input_file}")

    # Stream read JSONL file
    with open(args.input_file, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue

            stats["total"] += 1

            try:
                # Parse JSON line
                line_data = json.loads(line.strip())

                # Validate required fields
                if "question" not in line_data:
                    raise ValueError("Missing 'question' field")
                if "answer" not in line_data:
                    raise ValueError("Missing 'answer' field")

                # Prepare dataset item
                item = prepare_dataset_item(line_data.copy())

                # Create dataset item in Langfuse
                langfuse.create_dataset_item(
                    dataset_name=args.dataset_name,
                    input=item["input"],
                    expected_output=item["expected_output"],
                    metadata=item["metadata"]
                )

                stats["success"] += 1

                if args.debug and stats["success"] % 10 == 0:
                    print(f"Progress: {stats['success']} items imported...")

            except Exception as e:
                stats["failed"] += 1
                error_msg = f"Line {line_num}: {str(e)}"
                stats["errors"].append(error_msg)
                if args.debug:
                    print(f"Error: {error_msg}")

    return stats


def print_statistics(stats: Dict[str, Any]) -> None:
    """Print import statistics."""
    print("\n" + "=" * 50)
    print("Import Statistics")
    print("=" * 50)
    print(f"Total items processed: {stats['total']}")
    print(f"Successfully imported: {stats['success']}")
    print(f"Failed: {stats['failed']}")

    if stats['errors']:
        print(f"\nErrors ({len(stats['errors'])}):")
        for error in stats['errors'][:10]:  # Show first 10 errors
            print(f"  - {error}")
        if len(stats['errors']) > 10:
            print(f"  ... and {len(stats['errors']) - 10} more errors")

    print("=" * 50)


def main() -> None:
    """Main entry point."""
    args = parse_arguments()

    # Validate input file exists
    if not os.path.exists(args.input_file):
        print(f"Error: Input file not found: {args.input_file}")
        sys.exit(1)

    # Run import
    stats = import_jsonl_to_langfuse(args)

    # Print statistics
    print_statistics(stats)

    # Exit with error code if any failures
    if stats['failed'] > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
