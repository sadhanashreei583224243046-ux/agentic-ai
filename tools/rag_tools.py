"""
RAG Tools module for the Gemini ReAct Agent.
Provides semantic retrieval, exact keyword search, page inspection,
section listing, and arithmetic bill calculation tools.
"""

import json
import re
from typing import List, Callable, Optional, Dict, Any
from rag.retriever import HybridRetriever


class RAGToolKit:
    """
    Toolkit that binds a HybridRetriever to Gemini-compatible callable functions.
    """

    def __init__(self, retriever: HybridRetriever):
        self.retriever = retriever

    def search_document(self, query: str) -> str:
        """
        Search the document using hybrid semantic vector and keyword search.
        Use this tool to find information about menu items, categories, descriptions, or prices.

        Args:
            query: The search query, concept, dish name, or question to look up.

        Returns:
            A formatted string containing the top relevant excerpts and their page numbers.
        """
        results = self.retriever.search_hybrid(query, top_k=4)
        if not results:
            return f"No matching sections found for query: '{query}'."

        output_lines = [f"Found {len(results)} relevant excerpt(s) for '{query}':\n"]
        for idx, res in enumerate(results, 1):
            output_lines.append(
                f"[Result {idx} | Page {res['page_number']} | Section: {res['section']} | Score: {res['score']}]\n"
                f"{res['text']}\n"
            )
        return "\n".join(output_lines)

    def search_keyword(self, keyword: str) -> str:
        """
        Perform an exact keyword search across the document.
        Best for finding specific dish names, ingredients, or exact prices (e.g., 'Parotta', 'Dosa', 'Tea').

        Args:
            keyword: The specific term or item name to find.

        Returns:
            Excerpts containing the keyword with page numbers.
        """
        results = self.retriever.search_keyword(keyword, top_k=5)
        if not results:
            return f"No occurrences of keyword '{keyword}' found in the document."

        output_lines = [f"Found {len(results)} match(es) for keyword '{keyword}':\n"]
        for idx, res in enumerate(results, 1):
            output_lines.append(
                f"[Match {idx} | Page {res['page_number']} | Section: {res['section']}]\n"
                f"{res['text']}\n"
            )
        return "\n".join(output_lines)

    def get_page_content(self, page_number: int) -> str:
        """
        Retrieve the full unabridged text of a specific page in the document.
        Useful when you need to inspect the complete context of a page or browse an entire section.

        Args:
            page_number: The page number (1-indexed) to read.

        Returns:
            The complete text of that page.
        """
        return self.retriever.get_page_content(page_number)

    def list_document_sections(self) -> str:
        """
        List all detected categories and sections in the document.
        Use this to understand what categories or menu headings are available.

        Returns:
            A bulleted list of all detected sections and total pages.
        """
        sections = self.retriever.list_sections()
        total_pages = len(self.retriever.pages_text)
        if not sections:
            return f"Document has {total_pages} page(s) with general content."

        res = [f"Document structure ({total_pages} pages total):"]
        for sec in sections:
            res.append(f" - {sec}")
        return "\n".join(res)

    def calculate_bill(self, order_description: str) -> str:
        """
        Accurately calculates total cost, itemized amounts, and taxes for orders.
        Input should specify item names, unit prices, and quantities (e.g., '2 Chicken Biryani @ 190, 3 Parotta @ 25').

        Args:
            order_description: Text describing the items, quantities, and prices to calculate.

        Returns:
            Itemized breakdown and grand total.
        """
        # Parse patterns like: "2 Chicken Biryani @ 190", "3 x Parotta 25", or arithmetic
        items = []
        total = 0.0

        # Pattern 1: <qty> [x] <name> [@ at] <price>
        lines = [line.strip() for line in order_description.replace(";", "\n").replace(",", "\n").splitlines() if line.strip()]

        for line in lines:
            # Check for pattern like: 2 Chicken Biryani @ 190 or 2 * 190
            match = re.search(r"(\d+)\s*(?:x|\*|\sof\s)?\s*([a-zA-Z\s]+)?\s*[@|at|\:]?\s*[₹$]?\s*(\d+(?:\.\d+)?)", line)
            if match:
                qty = int(match.group(1))
                name = (match.group(2) or "Item").strip()
                unit_price = float(match.group(3))
                subtotal = qty * unit_price
                total += subtotal
                items.append(f"• {qty}x {name} @ ₹{unit_price:.2f} = ₹{subtotal:.2f}")
            else:
                # Fallback to simple number extraction or formula evaluation
                nums = re.findall(r"\d+(?:\.\d+)?", line)
                if len(nums) >= 2:
                    qty = float(nums[0])
                    price = float(nums[1])
                    subtotal = qty * price
                    total += subtotal
                    items.append(f"• {qty}x Item @ ₹{price:.2f} = ₹{subtotal:.2f}")

        if not items:
            # Check if simple arithmetic formula
            clean_expr = re.sub(r"[^0-9\+\-\*\/\.\(\)\s]", "", order_description)
            try:
                val = eval(clean_expr, {"__builtins__": None}, {})
                return f"Calculated Result: ₹{float(val):.2f}"
            except Exception:
                return f"Could not parse order items from: '{order_description}'."

        breakdown = "\n".join(items)
        return (
            f"--- ITEMIZED BILL BREAKDOWN ---\n"
            f"{breakdown}\n"
            f"-------------------------------\n"
            f"Grand Total: ₹{total:.2f}"
        )

    def get_tool_functions(self) -> List[Callable]:
        """Return the list of tool functions for Gemini AFC registration."""
        return [
            self.search_document,
            self.search_keyword,
            self.get_page_content,
            self.list_document_sections,
            self.calculate_bill,
        ]
