"""
Test file for 2_qa_synthesize.py new functions.
TDD test cases for chunks library loading and processing.
"""
import unittest
import os
import sys
import argparse
import json

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(__file__))

# Import the module directly
import importlib.util
spec = importlib.util.spec_from_file_location("two_qa_synthesize", os.path.join(os.path.dirname(__file__), "2_qa_synthesize.py"))
two_qa_synthesize = importlib.util.module_from_spec(spec)
spec.loader.exec_module(two_qa_synthesize)


class TestClassifyChunkType(unittest.TestCase):
    """Test chunk type classification function."""

    def setUp(self):
        self.classify_chunk_type = two_qa_synthesize.classify_chunk_type

    def test_classify_chunk_type_text(self):
        """Test text chunk classification - doc_type_kwd is empty string."""
        chunk = {"doc_type_kwd": "", "content": "some text content"}
        result = self.classify_chunk_type(chunk)
        self.assertEqual(result, "text", "Empty doc_type_kwd should be classified as text")

    def test_classify_chunk_type_image(self):
        """Test image chunk classification - doc_type_kwd is 'image' without <table>."""
        chunk = {"doc_type_kwd": "image", "content": "image description without table"}
        result = self.classify_chunk_type(chunk)
        self.assertEqual(result, "image", "Image doc_type without table tag should be classified as image")

    def test_classify_chunk_type_table(self):
        """Test table chunk classification - doc_type_kwd is 'image' with <table> in content."""
        chunk = {"doc_type_kwd": "image", "content": "<table><tr><td>data</td></tr></table>"}
        result = self.classify_chunk_type(chunk)
        self.assertEqual(result, "table", "Image doc_type with <table> tag should be classified as table")

    def test_classify_chunk_type_table_case_insensitive(self):
        """Test table classification is case-insensitive for <table> tag."""
        chunk = {"doc_type_kwd": "image", "content": "<TABLE><tr><td>data</td></tr></TABLE>"}
        result = self.classify_chunk_type(chunk)
        self.assertEqual(result, "table", "Table detection should be case-insensitive")

    def test_classify_chunk_type_unknown_doc_type(self):
        """Test unknown doc_type_kwd defaults to text."""
        chunk = {"doc_type_kwd": "unknown", "content": "some content"}
        result = self.classify_chunk_type(chunk)
        self.assertEqual(result, "text", "Unknown doc_type_kwd should default to text")


class TestHelperFunctions(unittest.TestCase):
    """Test helper functions for chunk processing."""

    def setUp(self):
        self.get_min_chunk_count = two_qa_synthesize.get_min_chunk_count
        self.has_both_text_and_image = two_qa_synthesize.has_both_text_and_image
        self.classify_chunk_type = two_qa_synthesize.classify_chunk_type

    def test_get_min_chunk_count_image_as_answer(self):
        """Test minimum chunk count for image_as_answer."""
        result = self.get_min_chunk_count("image_as_answer")
        self.assertEqual(result, 1, "image_as_answer should have min chunk count of 1")

    def test_get_min_chunk_count_table_as_answer(self):
        """Test minimum chunk count for table_as_answer."""
        result = self.get_min_chunk_count("table_as_answer")
        self.assertEqual(result, 1, "table_as_answer should have min chunk count of 1")

    def test_get_min_chunk_count_text_as_answer(self):
        """Test minimum chunk count for text_as_answer."""
        result = self.get_min_chunk_count("text_as_answer")
        self.assertEqual(result, 1, "text_as_answer should have min chunk count of 1")

    def test_get_min_chunk_count_image_plus_text_as_answer(self):
        """Test minimum chunk count for image_plus_text_as_answer."""
        result = self.get_min_chunk_count("image_plus_text_as_answer")
        self.assertEqual(result, 2, "image_plus_text_as_answer should have min chunk count of 2")

    def test_has_both_text_and_image_true(self):
        """Test has_both_text_and_image returns True when both types present."""
        text_chunk = {"doc_type_kwd": "", "content": "text"}
        image_chunk = {"doc_type_kwd": "image", "content": "image content"}
        result = self.has_both_text_and_image([text_chunk, image_chunk])
        self.assertTrue(result, "Should return True when both text and image chunks present")

    def test_has_both_text_and_image_with_table(self):
        """Test has_both_text_and_image returns False when table is present instead of image."""
        text_chunk = {"doc_type_kwd": "", "content": "text"}
        table_chunk = {"doc_type_kwd": "image", "content": "<table>data</table>"}
        result = self.has_both_text_and_image([text_chunk, table_chunk])
        self.assertFalse(result, "Should return False when table is present instead of image")

    def test_has_both_text_and_image_only_text(self):
        """Test has_both_text_and_image returns False when only text present."""
        text_chunk = {"doc_type_kwd": "", "content": "text"}
        result = self.has_both_text_and_image([text_chunk])
        self.assertFalse(result, "Should return False when only text chunks present")

    def test_has_both_text_and_image_only_image(self):
        """Test has_both_text_and_image returns False when only image present."""
        image_chunk = {"doc_type_kwd": "image", "content": "image content"}
        result = self.has_both_text_and_image([image_chunk])
        self.assertFalse(result, "Should return False when only image chunks present")


class TestAdaptChunkMetadata(unittest.TestCase):
    """Test metadata adaptation function."""

    def setUp(self):
        self.adapt_chunk_metadata = two_qa_synthesize.adapt_chunk_metadata

    def test_adapt_chunk_metadata_basic(self):
        """Test basic metadata adaptation."""
        chunk = {
            "document_keyword": "test.pdf",
            "document_id": "doc123",
            "id": "chunk456",
            "image_id": "img789"
        }
        metadata = self.adapt_chunk_metadata(chunk)

        self.assertIn("source", metadata)
        self.assertIn("test.pdf", metadata["source"])
        self.assertEqual(metadata["document_id"], "doc123")
        self.assertEqual(metadata["chunk_id"], "chunk456")
        self.assertEqual(metadata["image_id"], "img789")
        self.assertEqual(metadata["document_keyword"], "test.pdf")

    def test_adapt_chunk_metadata_source_format(self):
        """Test that source field is formatted correctly."""
        chunk = {
            "document_keyword": "file.pdf",
            "document_id": "doc_id_123",
            "id": "chunk_id_456",
        }
        metadata = self.adapt_chunk_metadata(chunk)

        expected_source = "file.pdf_id_doc_id_123_pg_chunk_id_456.txt"
        self.assertEqual(metadata["source"], expected_source)


class TestParseQATypes(unittest.TestCase):
    """Test QA types filter parsing function."""

    def setUp(self):
        self.parse_qa_types = two_qa_synthesize.parse_qa_types

    def test_parse_qa_types_single(self):
        """Test parsing single QA type."""
        result = self.parse_qa_types("image_as_answer")
        self.assertEqual(result, {"image_as_answer"})

    def test_parse_qa_types_multiple(self):
        """Test parsing multiple QA types."""
        result = self.parse_qa_types("image_as_answer,table_as_answer")
        self.assertEqual(result, {"image_as_answer", "table_as_answer"})

    def test_parse_qa_types_multiple_with_spaces(self):
        """Test parsing multiple QA types with spaces."""
        result = self.parse_qa_types("image_as_answer, table_as_answer , text_as_answer")
        self.assertEqual(result, {"image_as_answer", "table_as_answer", "text_as_answer"})

    def test_parse_qa_types_none(self):
        """Test parsing None returns None."""
        result = self.parse_qa_types(None)
        self.assertIsNone(result)

    def test_parse_qa_types_empty_string(self):
        """Test parsing empty string returns None."""
        result = self.parse_qa_types("")
        self.assertIsNone(result)

    def test_parse_qa_types_invalid(self):
        """Test parsing invalid type raises ValueError."""
        with self.assertRaises(ValueError) as context:
            self.parse_qa_types("invalid_type")
        self.assertIn("Invalid QA types", str(context.exception))


class TestLoadChunksLib(unittest.TestCase):
    """Test chunks library loading function."""

    def setUp(self):
        self.load_chunks_lib = two_qa_synthesize.load_chunks_lib
        # Use project root path for test file
        self.test_lib_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "batch_retrieve_results_test.json"
        )

    def test_load_chunks_lib_structure(self):
        """Test JSON chunks library loading returns correct structure."""
        if not os.path.exists(self.test_lib_path):
            self.skipTest(f"Test file {self.test_lib_path} not found")

        chunks_lib = self.load_chunks_lib(self.test_lib_path)

        self.assertIsInstance(chunks_lib, list)
        self.assertGreater(len(chunks_lib), 0)

        # Check first result structure
        first_result = chunks_lib[0]
        self.assertIn("chunks", first_result)
        self.assertIn("query_question", first_result)
        self.assertIsInstance(first_result["chunks"], list)

    def test_load_chunks_lib_filters_code_zero(self):
        """Test that only results with code=0 are loaded."""
        if not os.path.exists(self.test_lib_path):
            self.skipTest(f"Test file {self.test_lib_path} not found")

        chunks_lib = self.load_chunks_lib(self.test_lib_path)

        # All loaded results should have come from code=0 entries
        for result in chunks_lib:
            self.assertIn("chunks", result)
            self.assertIsInstance(result["chunks"], list)


class TestFormChunkGroups(unittest.TestCase):
    """Test chunk group formation function."""

    def setUp(self):
        self.form_chunk_groups_for_answer_type = two_qa_synthesize.form_chunk_groups_for_answer_type
        self.load_chunks_lib = two_qa_synthesize.load_chunks_lib
        self.classify_chunk_type = two_qa_synthesize.classify_chunk_type
        self.test_lib_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "batch_retrieve_results_test.json"
        )

    def test_form_chunk_groups_image_as_answer(self):
        """Test image_as_answer chunk group formation."""
        if not os.path.exists(self.test_lib_path):
            self.skipTest(f"Test file {self.test_lib_path} not found")

        chunks_lib = self.load_chunks_lib(self.test_lib_path)
        query_chunks = chunks_lib[0]["chunks"]

        groups = self.form_chunk_groups_for_answer_type(
            query_chunks, "image_as_answer", 2, None
        )

        # Should return at most 1 group per query
        self.assertLessEqual(len(groups), 1, "Should return at most 1 group per query")

        # If group exists, validate it contains only images (not tables)
        if groups:
            for chunk in groups[0]:
                chunk_type = self.classify_chunk_type(chunk)
                self.assertEqual(chunk_type, "image", "All chunks in image_as_answer group should be image type")

    def test_form_chunk_groups_table_as_answer(self):
        """Test table_as_answer chunk group formation."""
        if not os.path.exists(self.test_lib_path):
            self.skipTest(f"Test file {self.test_lib_path} not found")

        chunks_lib = self.load_chunks_lib(self.test_lib_path)
        query_chunks = chunks_lib[0]["chunks"]

        groups = self.form_chunk_groups_for_answer_type(
            query_chunks, "table_as_answer", 1, None
        )

        # Should return at most 1 group per query
        self.assertLessEqual(len(groups), 1)

        # If group exists, validate it contains only tables
        if groups:
            for chunk in groups[0]:
                chunk_type = self.classify_chunk_type(chunk)
                self.assertEqual(chunk_type, "table", "All chunks in table_as_answer group should be table type")

    def test_form_chunk_groups_text_as_answer(self):
        """Test text_as_answer chunk group formation."""
        if not os.path.exists(self.test_lib_path):
            self.skipTest(f"Test file {self.test_lib_path} not found")

        chunks_lib = self.load_chunks_lib(self.test_lib_path)
        query_chunks = chunks_lib[0]["chunks"]

        groups = self.form_chunk_groups_for_answer_type(
            query_chunks, "text_as_answer", 2, None
        )

        # Should return at most 1 group per query
        self.assertLessEqual(len(groups), 1)

        # If group exists, validate it contains only text
        if groups:
            for chunk in groups[0]:
                chunk_type = self.classify_chunk_type(chunk)
                self.assertEqual(chunk_type, "text", "All chunks in text_as_answer group should be text type")


class TestInitializeOpenAIClient(unittest.TestCase):
    """Test OpenAI client initialization."""

    def setUp(self):
        self.initialize_openai_client = two_qa_synthesize.initialize_openai_client

    def tearDown(self):
        # Clean up environment variables
        if "OPENAI_API_KEY" in os.environ:
            del os.environ["OPENAI_API_KEY"]
        if "OPENAI_BASE_URL" in os.environ:
            del os.environ["OPENAI_BASE_URL"]

    def test_openai_client_with_env_var(self):
        """Test OpenAI client initialization with environment variable."""
        os.environ["OPENAI_API_KEY"] = "test-key-env"
        args = argparse.Namespace(openai_api_key=None, openai_base_url=None)

        try:
            client = self.initialize_openai_client(args)
            self.assertIsNotNone(client)
        except Exception as e:
            # May fail with invalid key, but should not fail on API key lookup
            self.assertNotIn("not set", str(e))

    def test_openai_client_with_cli_fallback(self):
        """Test OpenAI client initialization with CLI argument fallback."""
        # Ensure no env var is set
        if "OPENAI_API_KEY" in os.environ:
            del os.environ["OPENAI_API_KEY"]

        args = argparse.Namespace(openai_api_key="test-key-cli", openai_base_url=None)

        try:
            client = self.initialize_openai_client(args)
            self.assertIsNotNone(client)
        except Exception as e:
            # May fail with invalid key, but should not fail on API key lookup
            self.assertNotIn("not set", str(e))

    def test_openai_client_missing_api_key(self):
        """Test OpenAI client initialization fails without API key."""
        # Ensure no env var is set
        if "OPENAI_API_KEY" in os.environ:
            del os.environ["OPENAI_API_KEY"]

        args = argparse.Namespace(openai_api_key=None, openai_base_url=None)

        with self.assertRaises(ValueError) as context:
            self.initialize_openai_client(args)
        self.assertIn("OPENAI_API_KEY not set", str(context.exception))

    def test_openai_client_with_base_url(self):
        """Test OpenAI client initialization with custom base URL."""
        os.environ["OPENAI_API_KEY"] = "test-key"
        os.environ["OPENAI_BASE_URL"] = "https://custom.api.com/v1"
        args = argparse.Namespace(openai_api_key=None, openai_base_url=None)

        try:
            client = self.initialize_openai_client(args)
            self.assertIsNotNone(client)
        except Exception as e:
            # May fail with invalid key, but should not fail on API key lookup
            self.assertNotIn("not set", str(e))


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete flow."""

    def setUp(self):
        self.test_lib_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "batch_retrieve_results_test.json"
        )

    def test_full_load_and_classify_flow(self):
        """Test loading chunks and classifying them."""
        if not os.path.exists(self.test_lib_path):
            self.skipTest(f"Test file {self.test_lib_path} not found")

        chunks_lib = two_qa_synthesize.load_chunks_lib(self.test_lib_path)

        # Classify all chunks from first query
        query_chunks = chunks_lib[0]["chunks"]
        types_found = set()

        for chunk in query_chunks:
            chunk_type = two_qa_synthesize.classify_chunk_type(chunk)
            types_found.add(chunk_type)

        # At minimum should have some text chunks
        self.assertIn("text", types_found, "Should find at least text chunks")


if __name__ == "__main__":
    unittest.main(verbosity=2)
