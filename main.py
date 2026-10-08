"""
Gemini RAG ReAct Agent - Interactive CLI & Benchmark Suite.

Features:
- PDF Ingestion and Hybrid Vector/Keyword Retrieval for 'nila.pdf'
- ReAct (Reasoning and Action) Autonomous Agent Loop
- Transparent Tool Calling Telemetry (Thought -> Action -> Observation)
- Environment Key Management via .env
- Interactive Shell, Single Query, and Benchmark Demonstration modes
"""

import os
import sys
import argparse
from dotenv import load_dotenv

# Ensure proper encoding and instant flushing on Windows console
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
        sys.stderr.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    except Exception:
        pass

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rag.indexer import DocumentIndexer
from rag.retriever import HybridRetriever
from tools.rag_tools import RAGToolKit
from agent.react_agent import GeminiReActAgent


def resolve_pdf_path(specified_path: str = None) -> str:
    """Locate nila.pdf in current directory or alternative common paths."""
    candidates = []
    if specified_path:
        candidates.append(specified_path)

    env_path = os.getenv("DOCUMENT_PATH")
    if env_path:
        candidates.append(env_path)

    # Standard candidate locations
    candidates.extend([
        "nila.pdf",
        "nila.pdf.pdf",
        os.path.join(os.path.dirname(__file__), "nila.pdf"),
        os.path.join(os.path.dirname(__file__), "nila.pdf.pdf"),
        os.path.expanduser(r"~\Downloads\nila.pdf.pdf"),
        os.path.expanduser(r"~\Downloads\nila.pdf"),
    ])

    for path in candidates:
        if path and os.path.exists(path):
            return os.path.abspath(path)

    return "nila.pdf"


def print_banner():
    banner = r"""
========================================================================
       🚀 GEMINI AGENTIC RAG SYSTEM: ReAct REASONING & RETRIEVAL
      Powered by Google Gemini | nila.pdf Knowledge Base | ReAct Loop
========================================================================
  * Architecture : ReAct Loop (Thought -> Action -> Observation -> Answer)
  * RAG Engine   : Hybrid Dense Semantic Embeddings + Sparse Keyword Match
  * Document     : nila.pdf (Nila Hotel Menu Card)
  * Tools        : search_document, search_keyword, get_page_content,
                   list_document_sections, calculate_bill
========================================================================
"""
    print(banner)


def setup_agent(pdf_path: str, model: str = None, verbose: bool = True) -> GeminiReActAgent:
    """Initialize Indexer, Retriever, Tools, and ReAct Agent."""
    # 1. Load .env
    load_dotenv(override=True)
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

    if not api_key:
        print("\n❌ [ERROR] GEMINI_API_KEY not found in .env file or environment!")
        print("Please edit the .env file in this directory and insert your Gemini API Key:")
        print("   GEMINI_API_KEY=\"your_actual_key_here\"\n")
        sys.exit(1)

    if not os.path.exists(pdf_path):
        print(f"\n❌ [ERROR] Document not found at '{pdf_path}'!")
        print("Please make sure 'nila.pdf' is placed in this root directory.\n")
        sys.exit(1)

    print(f"📖 [1/3] Ingesting & indexing document: '{os.path.basename(pdf_path)}'...")
    from google import genai
    client = genai.Client(api_key=api_key)

    embedding_model = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")
    indexer = DocumentIndexer(
        pdf_path=pdf_path,
        client=client,
        embedding_model=embedding_model,
        cache_dir=os.path.join(os.path.dirname(__file__), ".cache"),
    )
    chunks = indexer.load_and_index()
    print(f"       ✅ Indexed {len(chunks)} chunks across {len(indexer.pages_text)} pages.")
    print(f"       ✅ Detected Categories: {', '.join(indexer.sections_found)}")

    print(f"🔍 [2/3] Initializing Hybrid Retriever & RAG Tools...")
    retriever = HybridRetriever(
        chunks=chunks,
        pages_text=indexer.pages_text,
        sections_found=indexer.sections_found,
        client=client,
        embedding_model=embedding_model,
    )
    toolkit = RAGToolKit(retriever=retriever)

    print(f"🤖 [3/3] Initializing Gemini ReAct Agent...")
    chosen_model = model or os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
    agent = GeminiReActAgent(
        api_key=api_key,
        model=chosen_model,
        toolkit=toolkit,
        verbose=verbose,
    )
    print(f"       ✅ Agent Ready! Active Model: {agent.active_model}\n")
    return agent


def run_benchmark_demo(agent: GeminiReActAgent):
    """Run an automated ReAct reasoning demo across varied queries."""
    benchmark_queries = [
        (
            "Specific Item & Price Lookup",
            "What is the price of Masala Dosa and Pongal at Nila Hotel?",
        ),
        (
            "Category Discovery & Listing",
            "What biryani and fried specials are offered, and what are their prices?",
        ),
        (
            "Multi-Step Order & Bill Calculation",
            "If I order 2 Chicken Biryani, 4 Parotta, and 2 Filter Coffee, what is the exact total bill?",
        ),
        (
            "Vegetarian Diet Filter",
            "What vegetarian rice and meals options are available?",
        ),
        (
            "Page-Level Inspection & Beverages",
            "What hot drinks and fresh juices are served on page 2?",
        ),
    ]

    print("\n--- RUNNING REASONING & ACTION (ReAct) BENCHMARK DEMO ---\n")
    for idx, (title, query) in enumerate(benchmark_queries, 1):
        print(f"\n[{idx}/5] Scenario: {title}")
        print(f"User Query : \"{query}\"")
        print("-" * 65)
        try:
            result = agent.ask(query)
            print(f"\n🎯 [FINAL ANSWER]:\n{result['answer']}")
        except Exception as e:
            print(f"Execution Error: {e}")
        print("=" * 65)


def run_interactive_repl(agent: GeminiReActAgent):
    """Run interactive REPL session."""
    print("Interactive ReAct Agent session started.")
    print("Type your question about 'nila.pdf' (Nila Hotel menu).")
    print("Commands: '/reset' to clear conversation, '/exit' or 'quit' to quit.\n")

    while True:
        try:
            user_input = input("\nYou > ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit", "q", "/exit"):
                print("Exiting session. Thank you for using Nila RAG ReAct Agent!")
                break
            if user_input.lower() in ("reset", "/reset"):
                agent.reset_conversation()
                print("🔄 [System] Conversation history reset.")
                continue

            print("\n[ReAct Agent Reasoning & Executing Tools]...")
            result = agent.ask(user_input)
            print(f"\n🎯 [Agent Response]:\n{result['answer']}\n")

        except (KeyboardInterrupt, EOFError):
            print("\nSession ended by user. Goodbye!")
            break
        except Exception as e:
            print(f"\n[Error]: {e}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Gemini RAG ReAct Agent for nila.pdf"
    )
    parser.add_argument("query", nargs="?", help="Optional single question to query")
    parser.add_argument("--demo", action="store_true", help="Run automated ReAct benchmark demonstration")
    parser.add_argument("--pdf", type=str, default=None, help="Path to PDF document (default: nila.pdf)")
    parser.add_argument("--model", type=str, default=None, help="Gemini model name (e.g. gemini-3.8-flash)")
    parser.add_argument("--quiet", action="store_true", help="Suppress intermediate Thought & Action traces")

    args = parser.parse_args()

    print_banner()

    pdf_path = resolve_pdf_path(args.pdf)
    agent = setup_agent(
        pdf_path=pdf_path,
        model=args.model,
        verbose=not args.quiet,
    )

    if args.demo:
        run_benchmark_demo(agent)
    elif args.query:
        print(f"User Query: \"{args.query}\"\n")
        result = agent.ask(args.query)
        print(f"\n🎯 [FINAL ANSWER]:\n{result['answer']}")
    else:
        run_interactive_repl(agent)


if __name__ == "__main__":
    main()
