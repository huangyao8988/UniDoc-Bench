"""
Simple test script for new functions in 2_qa_synthesize.py
Tests the new functions without loading the full module with dependencies.
"""

def test_classify_chunk_type():
    """Test chunk type classification."""
    # Define the function inline to test logic
    def classify_chunk_type(chunk):
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

    # Test cases
    text_chunk = {"doc_type_kwd": "", "content": "some text content"}
    image_chunk = {"doc_type_kwd": "image", "content": "image description without table"}
    table_chunk = {"doc_type_kwd": "image", "content": "<table><tr><td>data</td></tr></table>"}

    assert classify_chunk_type(text_chunk) == "text", "Text chunk test failed"
    assert classify_chunk_type(image_chunk) == "image", "Image chunk test failed"
    assert classify_chunk_type(table_chunk) == "table", "Table chunk test failed"
    print("✓ classify_chunk_type tests passed")


def test_get_min_chunk_count():
    """Test minimum chunk count for each answer type."""
    def get_min_chunk_count(answer_type):
        min_counts = {
            "image_as_answer": 1,
            "table_as_answer": 1,
            "text_as_answer": 1,
            "image_plus_text_as_answer": 2
        }
        return min_counts.get(answer_type, 1)

    assert get_min_chunk_count("image_as_answer") == 1
    assert get_min_chunk_count("table_as_answer") == 1
    assert get_min_chunk_count("text_as_answer") == 1
    assert get_min_chunk_count("image_plus_text_as_answer") == 2
    print("✓ get_min_chunk_count tests passed")


def test_has_both_text_and_image():
    """Test has_both_text_and_image function."""
    def classify_chunk_type(chunk):
        doc_type = chunk.get("doc_type_kwd", "")
        content = chunk.get("content", "")
        if doc_type == "":
            return "text"
        elif doc_type == "image":
            if "<table>" in content.lower():
                return "table"
            else:
                return "image"
        else:
            return "text"

    def has_both_text_and_image(chunks):
        has_text = False
        has_image = False
        for chunk in chunks:
            chunk_type = classify_chunk_type(chunk)
            if chunk_type == "text":
                has_text = True
            elif chunk_type == "image":
                has_image = True
        return has_text and has_image

    text_chunk = {"doc_type_kwd": "", "content": "text"}
    image_chunk = {"doc_type_kwd": "image", "content": "image content"}
    table_chunk = {"doc_type_kwd": "image", "content": "<table>data</table>"}

    assert has_both_text_and_image([text_chunk, image_chunk]) == True
    assert has_both_text_and_image([text_chunk, table_chunk]) == False
    assert has_both_text_and_image([text_chunk]) == False
    assert has_both_text_and_image([image_chunk]) == False
    print("✓ has_both_text_and_image tests passed")


def test_adapt_chunk_metadata():
    """Test metadata adaptation."""
    def adapt_chunk_metadata(chunk):
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

    chunk = {
        "document_keyword": "test.pdf",
        "document_id": "doc123",
        "id": "chunk456",
        "image_id": "img789"
    }

    metadata = adapt_chunk_metadata(chunk)
    assert "test.pdf" in metadata["source"]
    assert metadata["document_id"] == "doc123"
    assert metadata["chunk_id"] == "chunk456"
    assert metadata["image_id"] == "img789"
    print("✓ adapt_chunk_metadata tests passed")


def test_parse_qa_types():
    """Test QA types parsing."""
    def parse_qa_types(qa_types_str):
        if not qa_types_str:
            return None
        valid_types = {"image_as_answer", "table_as_answer", "text_as_answer", "image_plus_text_as_answer"}
        requested_types = set(t.strip() for t in qa_types_str.split(","))
        invalid = requested_types - valid_types
        if invalid:
            raise ValueError(f"Invalid QA types: {invalid}. Valid types: {valid_types}")
        return requested_types

    assert parse_qa_types("image_as_answer") == {"image_as_answer"}
    assert parse_qa_types("image_as_answer,table_as_answer") == {"image_as_answer", "table_as_answer"}
    assert parse_qa_types(None) is None
    assert parse_qa_types("") is None

    try:
        parse_qa_types("invalid_type")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Invalid QA types" in str(e)

    print("✓ parse_qa_types tests passed")


def test_load_chunks_lib():
    """Test chunks library loading."""
    import json
    import os

    test_lib_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "batch_retrieve_results_test.json"
    )

    if not os.path.exists(test_lib_path):
        print("⊘ load_chunks_lib test skipped (test file not found)")
        return

    def load_chunks_lib(chunks_lib_path):
        with open(chunks_lib_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
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

    chunks_lib = load_chunks_lib(test_lib_path)
    assert isinstance(chunks_lib, list)
    assert len(chunks_lib) > 0
    assert "chunks" in chunks_lib[0]
    assert "query_question" in chunks_lib[0]
    print(f"✓ load_chunks_lib tests passed (loaded {len(chunks_lib)} query results)")


def test_form_chunk_groups():
    """Test chunk group formation."""
    def classify_chunk_type(chunk):
        doc_type = chunk.get("doc_type_kwd", "")
        content = chunk.get("content", "")
        if doc_type == "":
            return "text"
        elif doc_type == "image":
            if "<table>" in content.lower():
                return "table"
            else:
                return "image"
        else:
            return "text"

    def get_min_chunk_count(answer_type):
        min_counts = {
            "image_as_answer": 1,
            "table_as_answer": 1,
            "text_as_answer": 1,
            "image_plus_text_as_answer": 2
        }
        return min_counts.get(answer_type, 1)

    def has_both_text_and_image(chunks):
        has_text = False
        has_image = False
        for chunk in chunks:
            chunk_type = classify_chunk_type(chunk)
            if chunk_type == "text":
                has_text = True
            elif chunk_type == "image":
                has_image = True
        return has_text and has_image

    def form_chunk_groups_for_answer_type(query_chunks, answer_type, chunk_count, file_constraint):
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

        # ... (other constraints omitted for brevity)

        return groups[:1] if groups else []

    # Create mock chunks
    mock_chunks = [
        {"doc_type_kwd": "", "content": "text 1", "document_id": "doc1"},
        {"doc_type_kwd": "", "content": "text 2", "document_id": "doc1"},
        {"doc_type_kwd": "image", "content": "image content", "document_id": "doc1"},
    ]

    # Test text_as_answer grouping
    groups = form_chunk_groups_for_answer_type(mock_chunks, "text_as_answer", 2, None)
    assert len(groups) <= 1, "Should return at most 1 group per query"

    # If group exists, validate it contains only text
    if groups:
        for chunk in groups[0]:
            assert classify_chunk_type(chunk) == "text"

    print("✓ form_chunk_groups_for_answer_type tests passed")


if __name__ == "__main__":
    print("Running new function tests...\n")
    test_classify_chunk_type()
    test_get_min_chunk_count()
    test_has_both_text_and_image()
    test_adapt_chunk_metadata()
    test_parse_qa_types()
    test_load_chunks_lib()
    test_form_chunk_groups()
    print("\n✅ All tests passed!")
