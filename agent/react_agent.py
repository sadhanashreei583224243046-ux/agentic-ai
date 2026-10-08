"""
Gemini ReAct (Reasoning and Action) Agent.
Implements the ReAct cycle (Thought -> Action -> Observation -> Final Answer)
using Google GenAI function calling and grounding against document retrievals.
"""

import os
import sys
import time
from typing import List, Callable, Optional, Dict, Any
from google import genai
from google.genai import types
from google.genai.errors import ServerError, ClientError

from tools.rag_tools import RAGToolKit


REACT_SYSTEM_INSTRUCTION = """You are an expert ReAct (Reasoning and Action) Agentic AI specialized in Question Answering and Information Retrieval from the document 'nila.pdf' (Nila Hotel Menu Card).

You MUST strictly follow the ReAct paradigm to answer user questions:
1. **THOUGHT**: Analyze the user's request. Think step-by-step about what information is required, which category or page it might be in, and which tool to use.
2. **ACTION**: Invoke the most appropriate retrieval or calculation tool (search_document, search_keyword, get_page_content, list_document_sections, calculate_bill).
3. **OBSERVATION**: Carefully examine the retrieved facts and citations returned by the tool.
4. **THOUGHT**: Assess whether you have complete, sufficient evidence to answer the user's question. If more details (like prices, drinks, or totals) are needed, invoke another action.
5. **FINAL ANSWER**: Once you have all facts, synthesize a clear, helpful, well-structured response.

STRICT GROUNDING RULES:
- Never assume, fabricate, or hallucinate menu items, categories, or prices.
- Retrieve information directly from the document using your tools before answering.
- Always include citations (e.g., [Page 1, 'BREAKFAST & TIFFIN'] or [Page 2, 'BREADS & DRINKS']) for transparency.
- For bill calculations or arithmetic, always call the 'calculate_bill' tool to ensure 100% precision.
"""

DEFAULT_FALLBACK_MODELS = [
    "gemini-3-flash-preview",
    "gemini-3.8-flash",
    "gemini-flash-latest",
]


class GeminiReActAgent:
    """
    ReAct Agent combining Gemini LLM with RAG Tools and full telemetry.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        toolkit: Optional[RAGToolKit] = None,
        tools: Optional[List[Callable]] = None,
        fallback_models: Optional[List[str]] = None,
        system_instruction: str = REACT_SYSTEM_INSTRUCTION,
        verbose: bool = True,
    ):
        # Configure safe stdout encoding on Windows
        if sys.platform.startswith("win"):
            try:
                sys.stdout.reconfigure(encoding="utf-8", errors="replace")
                sys.stderr.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass

        self.api_key = (
            api_key
            or os.environ.get("GEMINI_API_KEY")
            or os.environ.get("GOOGLE_API_KEY")
        )
        if not self.api_key:
            raise ValueError(
                "Gemini API Key not found! Please set GEMINI_API_KEY in your .env file or environment."
            )

        self.client = genai.Client(api_key=self.api_key)
        self.primary_model = (
            model or os.environ.get("GEMINI_MODEL") or DEFAULT_FALLBACK_MODELS[0]
        )
        self.fallback_models = fallback_models or DEFAULT_FALLBACK_MODELS

        if toolkit:
            self.tools = toolkit.get_tool_functions()
        elif tools:
            self.tools = tools
        else:
            raise ValueError("Either 'toolkit' or 'tools' must be provided.")

        self.system_instruction = system_instruction
        self.verbose = verbose

        self.active_model = self.primary_model
        self.chat = None
        self._init_chat(self.active_model)

    def _init_chat(self, model_name: str) -> None:
        """Create a stateful chat session with the chosen Gemini model and tools."""
        self.chat = self.client.chats.create(
            model=model_name,
            config=types.GenerateContentConfig(
                tools=self.tools,
                system_instruction=self.system_instruction,
                temperature=0.2,  # Low temperature for deterministic grounding
            ),
        )
        self.active_model = model_name

    def reset_conversation(self) -> None:
        """Clear conversation history."""
        self._init_chat(self.active_model)

    def ask(self, query: str, max_retries: int = 3) -> Dict[str, Any]:
        """
        Processes a query through the ReAct loop, executing tool actions,
        capturing telemetry, and returning the structured result.

        Returns:
            Dict containing 'answer', 'model', and 'traces' (list of Thoughts, Actions, Observations).
        """
        models_to_attempt = [self.active_model] + [
            m for m in self.fallback_models if m != self.active_model
        ]
        last_error = None

        for current_model in models_to_attempt:
            for attempt in range(1, max_retries + 1):
                try:
                    if self.active_model != current_model or self.chat is None:
                        if self.verbose and self.active_model != current_model:
                            print(
                                f"[Agent Info] Switching model from {self.active_model} -> {current_model}..."
                            )
                        self._init_chat(current_model)

                    history_before = len(self.chat.get_history())

                    # Send prompt to Gemini (AFC executes tool actions automatically)
                    response = self.chat.send_message(query)

                    # Extract ReAct step traces from chat history turns
                    traces = self._extract_react_traces(history_before)

                    if self.verbose:
                        try:
                            self._print_react_traces(traces)
                        except Exception as print_err:
                            pass

                    return {
                        "answer": response.text,
                        "model": self.active_model,
                        "traces": traces,
                    }

                except ServerError as e:
                    last_error = e
                    backoff = attempt * 1.5
                    if self.verbose:
                        print(
                            f"[Agent Warning] {current_model} busy (503). Retrying in {backoff:.1f}s (Attempt {attempt}/{max_retries})..."
                        )
                    time.sleep(backoff)

                except ClientError as e:
                    last_error = e
                    if self.verbose:
                        print(
                            f"[Agent Warning] {current_model} client error: {e}. Trying fallback model..."
                        )
                    break  # Try next model in fallback list

                except Exception as e:
                    last_error = e
                    if self.verbose:
                        print(f"[Agent Warning] Unexpected API error with {current_model}: {e}")
                    break

        raise RuntimeError(
            f"All candidate models failed to process query. Last error: {last_error}"
        )

    def _extract_react_traces(self, start_idx: int) -> List[Dict[str, Any]]:
        """
        Parses chat turns from start_idx to identify Thought, Action, and Observation cycles.
        """
        traces = []
        try:
            history = self.chat.get_history()
            for msg in history[start_idx:]:
                for part in msg.parts:
                    # Model's Thought or intermediate reasoning text
                    if getattr(part, "text", None):
                        text = part.text.strip()
                        if text:
                            traces.append({"type": "thought", "content": text})

                    # Action: Function call requested by model
                    elif getattr(part, "function_call", None):
                        fc = part.function_call
                        traces.append({
                            "type": "action",
                            "tool_name": fc.name,
                            "arguments": dict(fc.args) if fc.args else {},
                        })

                    # Observation: Function response returned from local tool
                    elif getattr(part, "function_response", None):
                        fr = part.function_response
                        resp = getattr(fr, "response", None)
                        if hasattr(resp, "items"):
                            resp_content = dict(resp)
                        else:
                            resp_content = resp
                        traces.append({
                            "type": "observation",
                            "tool_name": fr.name,
                            "result": resp_content,
                        })
        except Exception:
            pass
        return traces

    def _print_react_traces(self, traces: List[Dict[str, Any]]) -> None:
        """Pretty-print the ReAct execution trace safely to stdout."""
        for step in traces:
            step_type = step.get("type")
            if step_type == "thought":
                print(f"\n  [THOUGHT] {step['content']}")
            elif step_type == "action":
                tool = step["tool_name"]
                args = step["arguments"]
                args_str = ", ".join(f"{k}={repr(v)}" for k, v in args.items())
                print(f"  [ACTION]  {tool}({args_str})")
            elif step_type == "observation":
                tool = step["tool_name"]
                res = step["result"]
                res_str = str(res)
                if len(res_str) > 280:
                    res_str = res_str[:280] + "... [truncated]"
                print(f"  [OBSERVATION ({tool})]\n    {res_str}")
