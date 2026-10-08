"""
Comprehensive Test Suite for Gemini RAG ReAct Agent.
Tests PDF extraction, chunking, retrieval tools, and ReAct agent loop.
"""

import os
import sys
import unittest
from dotenv import load_dotenv

# Ensure UTF-8 console output on Windows
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rag.indexer import DocumentIndexer
from rag.retriever import HybridRetriever
from tools.rag_tools import RAGToolKit
from agent.react_agent import GeminiReActAgent


class TestRAGComponents(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        load_dotenv(override=True)
        cls.pdf_path = os.path.join(os.path.dirname(__file__), "nila.pdf")
        if not os.path.exists(cls.pdf_path):
            alt_path = os.path.join(os.path.dirname(__file__), "nila.pdf.pdf")
            if os.path.exists(alt_path):
                cls.pdf_path = alt_path

        # Setup indexer without client for deterministic offline testing
        cls.indexer = DocumentIndexer(
            pdf_path=cls.pdf_path,
            client=None,
            cache_dir=os.path.join(os.path.dirname(__file__), ".test_cache"),
        )
        cls.chunks = cls.indexer.load_and_index()
        cls.retriever = HybridRetriever(
            chunks=cls.chunks,
            pages_text=cls.indexer.pages_text,
            sections_found=cls.indexer.sections_found,
            client=None,
        )
        cls.toolkit = RAGToolKit(retriever=cls.retriever)

    def test_pdf_extraction(self):
        """Verify that PDF pages were parsed and text extracted."""
        self.assertGreaterEqual(len(self.indexer.pages_text), 2, "Expected at least 2 pages in nila.pdf")
        p1_text = self.indexer.pages_text[0]
        self.assertIn("NILA HOTEL", p1_text)
        self.assertIn("Masala Dosa", p1_text)

    def test_section_detection(self):
        """Verify detected menu sections."""
        sections = self.indexer.sections_found
        self.assertIn("BREAKFAST & TIFFIN", sections)
        self.assertIn("RICE & MEALS", sections)
        self.assertIn("BIRYANI & FRIED SPECIALS", sections)

    def test_keyword_search(self):
        """Verify exact keyword search finds expected items."""
        results = self.retriever.search_keyword("Chicken Biryani")
        self.assertGreater(len(results), 0, "Chicken Biryani should be found")
        self.assertEqual(results[0]["page_number"], 1)
        self.assertIn("190", results[0]["text"])

        # Check Page 2 item: Parotta or Chilli Chicken
        results_p2 = self.retriever.search_keyword("Parotta")
        self.assertGreater(len(results_p2), 0, "Parotta should be found on page 2")
        self.assertEqual(results_p2[0]["page_number"], 2)

    def test_get_page_content_tool(self):
        """Verify get_page_content returns full page text."""
        p1 = self.toolkit.get_page_content(1)
        self.assertIn("BREAKFAST & TIFFIN", p1)
        p2 = self.toolkit.get_page_content(2)
        self.assertIn("BREADS & DRINKS", p2)

    def test_calculate_bill_tool(self):
        """Verify order bill calculation tool accurately multiplies and totals."""
        order = "2 Chicken Biryani @ 190, 4 Parotta @ 25, 2 Filter Coffee @ 35"
        # 2*190=380, 4*25=100, 2*35=70 -> Total = 550
        result = self.toolkit.calculate_bill(order)
        self.assertIn("550.00", result)
        self.assertIn("380.00", result)
        self.assertIn("100.00", result)


class TestReActAgentLive(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        load_dotenv(override=True)
        cls.api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not cls.api_key:
            raise unittest.SkipTest("GEMINI_API_KEY not found; skipping live API test.")

        pdf_path = os.path.join(os.path.dirname(__file__), "nila.pdf")
        if not os.path.exists(pdf_path):
            pdf_path = os.path.join(os.path.dirname(__file__), "nila.pdf.pdf")

        from google import genai
        client = genai.Client(api_key=cls.api_key)
        indexer = DocumentIndexer(
            pdf_path=pdf_path,
            client=client,
            cache_dir=os.path.join(os.path.dirname(__file__), ".cache"),
        )
        chunks = indexer.load_and_index()
        retriever = HybridRetriever(
            chunks=chunks,
            pages_text=indexer.pages_text,
            sections_found=indexer.sections_found,
            client=client,
        )
        toolkit = RAGToolKit(retriever=retriever)
        cls.agent = GeminiReActAgent(
            api_key=cls.api_key,
            model=os.getenv("GEMINI_MODEL", "gemini-3.8-flash"),
            toolkit=toolkit,
            verbose=True,
        )

    def test_live_react_query(self):
        """Test a live question requiring ReAct tool invocation."""
        query = "What is the price of Mutton Biryani at Nila Hotel, and which page is it on?"
        response = self.agent.ask(query)
        self.assertIn("answer", response)
        ans = response["answer"].lower()
        self.assertTrue("240" in ans or "mutton biryani" in ans)
        # Verify that tools were invoked
        tool_actions = [t for t in response["traces"] if t.get("type") == "action"]
        self.assertGreater(len(tool_actions), 0, "ReAct agent must invoke at least one tool action")


if __name__ == "__main__":
    unittest.main()
