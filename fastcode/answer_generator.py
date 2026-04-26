"""
Answer Generator - Generate answers using LLM with retrieved context
"""

import logging
import os
import re
from collections.abc import Callable, Generator
from typing import Any, cast

from anthropic import Anthropic
from dotenv import load_dotenv
from openai import OpenAI

from .core import context as _context
from .core import summary as _summary
from .llm_utils import openai_chat_completion
from .utils import count_tokens, truncate_to_tokens


class AnswerGenerator:
    """Generate natural language answers using LLM"""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.gen_config = config.get("generation", {})
        self.logger = logging.getLogger(__name__)

        self.provider = self.gen_config.get("provider", "openai")
        # self.model = self.gen_config.get("model", "openai/gpt-oss-120b")
        # self.base_url = self.gen_config.get("base_url", "https://openrouter.ai/api/v1")
        self.temperature = self.gen_config.get("temperature", 0.4)
        self.max_tokens = self.gen_config.get("max_tokens", 20000)

        self.max_context_tokens = self.gen_config.get("max_context_tokens", 200000)
        self.reserve_tokens = self.gen_config.get("reserve_tokens_for_response", 10000)

        # Multi-turn dialogue settings
        self.enable_multi_turn = self.gen_config.get("enable_multi_turn", False)
        self.context_rounds = self.gen_config.get("context_rounds", 10)

        self.include_file_paths = self.gen_config.get("include_file_paths", True)
        self.include_line_numbers = self.gen_config.get("include_line_numbers", True)
        self.include_related_code = self.gen_config.get("include_related_code", True)

        # Load environment variables from .env file
        load_dotenv()

        # Initialize LLM client
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        self.base_url = os.getenv("BASE_URL")
        self.model = os.getenv("MODEL")
        self.client: OpenAI | Anthropic | None = self._initialize_client()

    def _initialize_client(self):
        """Initialize LLM client based on provider"""
        if self.provider == "openai":
            api_key = self.api_key
            if not api_key:
                self.logger.warning("OPENAI_API_KEY not set")
            return OpenAI(api_key=api_key, base_url=self.base_url)

        if self.provider == "anthropic":
            api_key = self.anthropic_api_key
            if not api_key:
                self.logger.warning("ANTHROPIC_API_KEY not set")
            return Anthropic(api_key=api_key, base_url=self.base_url)

        self.logger.warning(f"Unknown provider: {self.provider}")
        return None

    def generate(
        self,
        query: str,
        retrieved_elements: list[dict[str, Any]],
        query_info: dict[str, Any] | None = None,
        dialogue_history: list[dict[str, Any]] | None = None,
        prompt_builder: Callable[
            [str, str, dict[str, Any] | None, list[dict[str, Any]] | None], str
        ]
        | None = None,
    ) -> dict[str, Any]:
        """
        Generate answer for query using retrieved context

        Args:
            query: User query
            retrieved_elements: List of retrieved code elements
            query_info: Additional query processing info
            dialogue_history: Previous dialogue turns for multi-turn mode
            prompt_builder: Optional callable that returns a prompt given
                (query, prepared_context, query_info, dialogue_history)

        Returns:
            Dictionary with answer, summary (if multi-turn), and metadata
        """
        self.logger.info("Generating answer")

        # Prepare context
        context = self._prepare_context(retrieved_elements)

        # Build prompt (with dialogue history if in multi-turn mode)
        if prompt_builder:
            prompt = prompt_builder(query, context, query_info, dialogue_history)
        else:
            prompt = self._build_prompt(query, context, query_info, dialogue_history)

        # Count tokens
        prompt_tokens = count_tokens(prompt, self.model or "gpt-4")
        self.logger.info(f"Initial prompt tokens: {prompt_tokens}")

        # Calculate available tokens for input
        # Reserve tokens for: output (max_tokens) + safety margin
        available_input_tokens = (
            self.max_context_tokens - self.max_tokens - self.reserve_tokens
        )

        # Truncate if needed - keep front part, truncate from the end
        if prompt_tokens > available_input_tokens:
            self.logger.warning(
                f"Prompt exceeds limit ({prompt_tokens} > {available_input_tokens} tokens). "
                f"Truncating context to fit. (max_context_tokens={self.max_context_tokens}, "
                f"max_tokens={self.max_tokens}, reserve={self.reserve_tokens})"
            )

            # Calculate tokens for each component to determine how much context we can keep
            system_prompt_sample = self._build_prompt(
                query, "", query_info, dialogue_history
            )
            base_tokens = count_tokens(system_prompt_sample, self.model or "gpt-4")
            context_token_budget = (
                available_input_tokens - base_tokens - 100
            )  # Extra safety margin

            if context_token_budget > 0:
                # Truncate context from the end (keep beginning)
                context = self._truncate_context(context, context_token_budget)
                self.logger.info(f"Context truncated to ~{context_token_budget} tokens")
            else:
                self.logger.error(
                    f"Cannot fit prompt even without context! "
                    f"Base tokens: {base_tokens}, available: {available_input_tokens}"
                )
                context = ""  # Last resort

            # Rebuild prompt with truncated context
            if prompt_builder:
                prompt = prompt_builder(query, context, query_info, dialogue_history)
            else:
                prompt = self._build_prompt(
                    query, context, query_info, dialogue_history
                )

            # Verify final token count
            final_prompt_tokens = count_tokens(prompt, self.model or "gpt-4")
            self.logger.info(
                f"Final prompt tokens after truncation: {final_prompt_tokens}"
            )
            prompt_tokens = final_prompt_tokens

        # Generate answer
        try:
            if self.provider == "openai":
                raw_response = self._generate_openai(prompt)
            elif self.provider == "anthropic":
                raw_response = self._generate_anthropic(prompt)
            else:
                raw_response = "Error: LLM provider not configured"
            # print("raw_response: ", raw_response)

            # # Save raw_response to JSON file
            # test_data = {"answer": raw_response}
            # test_file_path = os.path.join(os.path.dirname(__file__), "test_specialized.json")
            # with open(test_file_path, 'w', encoding='utf-8') as f:
            #     json.dump(test_data, f, ensure_ascii=False, indent=2)

            # Parse response to extract answer and summary (if multi-turn mode)
            if self.enable_multi_turn and dialogue_history is not None:
                answer, summary = self._parse_response_with_summary(raw_response)

                # Fallback: Generate summary if parsing failed
                if not summary:
                    self.logger.info(
                        "Generating fallback summary from retrieved elements"
                    )
                    summary = self._generate_fallback_summary(
                        query, answer, retrieved_elements
                    )
            else:
                answer = raw_response
                summary = None

            result = {
                "answer": answer,
                "query": query,
                "context_elements": len(retrieved_elements),
                "prompt_tokens": prompt_tokens,
                "sources": self._extract_sources(retrieved_elements),
            }

            if summary:
                result["summary"] = summary

            return result

        except Exception as e:
            self.logger.error(f"Failed to generate answer: {e}")
            import traceback

            full_error = traceback.format_exc()
            self.logger.error(f"Full error traceback:\n{full_error}")
            return {
                "answer": f"Error generating answer: {e!s}",
                "query": query,
                "context_elements": len(retrieved_elements),
                "error": full_error,
            }

    def generate_stream(
        self,
        query: str,
        retrieved_elements: list[dict[str, Any]],
        query_info: dict[str, Any] | None = None,
        dialogue_history: list[dict[str, Any]] | None = None,
        prompt_builder: Callable[
            [str, str, dict[str, Any] | None, list[dict[str, Any]] | None], str
        ]
        | None = None,
    ) -> Generator[tuple[str | None, dict[str, Any] | None], None, None]:
        """
        Generate answer with streaming support (yields chunks of text)

        Args:
            query: User query
            retrieved_elements: List of retrieved code elements
            query_info: Additional query processing info
            dialogue_history: Previous dialogue turns for multi-turn mode
            prompt_builder: Optional callable for custom prompt building

        Yields:
            Tuples of (chunk_text, metadata_dict or None)
            - First yield includes metadata: (None, {"prompt_tokens": ..., "sources": [...]})
            - Subsequent yields are text chunks: (text_chunk, None)
            - Final yield includes complete response: (remaining_text, {"summary": ..., "complete": True})
        """
        self.logger.info("Generating streaming answer")

        # Prepare context
        context = self._prepare_context(retrieved_elements)

        # Build prompt (with dialogue history if in multi-turn mode)
        if prompt_builder:
            prompt = prompt_builder(query, context, query_info, dialogue_history)
        else:
            prompt = self._build_prompt(query, context, query_info, dialogue_history)

        # Count tokens and truncate if needed (same logic as generate())
        prompt_tokens = count_tokens(prompt, self.model or "gpt-4")
        self.logger.info(f"Initial prompt tokens: {prompt_tokens}")

        available_input_tokens = (
            self.max_context_tokens - self.max_tokens - self.reserve_tokens
        )

        if prompt_tokens > available_input_tokens:
            self.logger.warning(
                f"Prompt exceeds limit ({prompt_tokens} > {available_input_tokens} tokens). Truncating context."
            )

            system_prompt_sample = self._build_prompt(
                query, "", query_info, dialogue_history
            )
            base_tokens = count_tokens(system_prompt_sample, self.model or "gpt-4")
            context_token_budget = available_input_tokens - base_tokens - 100

            if context_token_budget > 0:
                context = self._truncate_context(context, context_token_budget)
                self.logger.info(f"Context truncated to ~{context_token_budget} tokens")
            else:
                self.logger.error("Cannot fit prompt even without context!")
                context = ""

            if prompt_builder:
                prompt = prompt_builder(query, context, query_info, dialogue_history)
            else:
                prompt = self._build_prompt(
                    query, context, query_info, dialogue_history
                )

            final_prompt_tokens = count_tokens(prompt, self.model or "gpt-4")
            self.logger.info(
                f"Final prompt tokens after truncation: {final_prompt_tokens}"
            )
            prompt_tokens = final_prompt_tokens

        # Yield metadata first
        metadata = {
            "prompt_tokens": prompt_tokens,
            "sources": self._extract_sources(retrieved_elements),
            "context_elements": len(retrieved_elements),
            "query": query,
        }
        yield None, metadata

        # Determine if we need to filter summary from streaming output
        filter_summary = self.enable_multi_turn and dialogue_history is not None

        # Generate streaming answer
        try:
            full_response: list[str] = []
            displayed_response: list[str] = []

            if filter_summary:
                # Use buffered streaming to filter out <SUMMARY> section
                for original_chunk, filtered_chunk in self._stream_with_summary_filter(
                    prompt
                ):
                    if original_chunk:
                        full_response.append(original_chunk)
                    if filtered_chunk:
                        displayed_response.append(filtered_chunk)
                        yield filtered_chunk, None
            # Normal streaming without filtering
            elif self.provider == "openai":
                for chunk in self._generate_openai_stream(prompt):
                    full_response.append(chunk)
                    yield chunk, None
            elif self.provider == "anthropic":
                for chunk in self._generate_anthropic_stream(prompt):
                    full_response.append(chunk)
                    yield chunk, None
            else:
                error_msg = "Error: LLM provider not configured"
                yield error_msg, None

            # Parse complete response for summary (multi-turn mode)
            raw_response = "".join(full_response)
            summary = None

            if filter_summary:
                answer, summary = self._parse_response_with_summary(raw_response)
                if not summary:
                    self.logger.info(
                        "Generating fallback summary from retrieved elements"
                    )
                    summary = self._generate_fallback_summary(
                        query, answer, retrieved_elements
                    )

            # Final yield with summary and completion flag
            final_metadata: dict[str, Any] = {"complete": True}
            if summary:
                final_metadata["summary"] = summary
            yield None, final_metadata

        except Exception as e:
            self.logger.error(f"Failed to generate streaming answer: {e}")
            import traceback

            full_error = traceback.format_exc()
            self.logger.error(f"Full error traceback:\n{full_error}")
            error_msg = f"Error generating answer: {e!s}"
            yield error_msg, {"error": full_error, "complete": True}

    def _stream_with_summary_filter(
        self, prompt: str
    ) -> Generator[tuple[str | None, str | None], None, None]:
        """
        Stream LLM response while filtering out <SUMMARY>...</SUMMARY> section.

        Yields:
            Tuples of (original_chunk, filtered_chunk)
            - original_chunk: The raw chunk from LLM (for building full response)
            - filtered_chunk: The chunk to display (None if part of summary)
        """
        # Summary tag patterns - use regex for robust matching
        # Matches variations like: <SUMMARY>, <Summary>, <summary>, <SUMMARY:>, **SUMMARY**, etc.
        summary_start_regex = re.compile(
            r"<\s*[Ss][Uu][Mm][Mm][Aa][Rr][Yy]\s*:?\s*>|"  # <SUMMARY>, <Summary:>, etc.
            r"\*\*\s*[Ss][Uu][Mm][Mm][Aa][Rr][Yy]\s*\*\*\s*:?|"  # **SUMMARY**, **Summary**:
            r"\*\*\s*<\s*[Ss][Uu][Mm][Mm][Aa][Rr][Yy]\s*>\s*\*\*"  # **<SUMMARY>**
        )
        summary_end_regex = re.compile(
            r"<\s*/\s*[Ss][Uu][Mm][Mm][Aa][Rr][Yy]\s*>|"  # </SUMMARY>, </Summary>, etc.
            r"\*\*\s*<\s*/\s*[Ss][Uu][Mm][Mm][Aa][Rr][Yy]\s*>\s*\*\*"  # **</SUMMARY>**
        )

        # Legacy exact patterns for quick initial check
        summary_start_patterns = ["<SUMMARY>", "<summary>", "<Summary>"]
        summary_end_patterns = ["</SUMMARY>", "</summary>", "</Summary>"]

        # Buffer for detecting summary start
        buffer: str = ""
        in_summary = False
        max_buffer_size = 20  # Buffer enough to detect "<SUMMARY>"

        # Choose stream generator based on provider
        if self.provider == "openai":
            stream_generator: Generator[str, None, None] = self._generate_openai_stream(
                prompt
            )
        elif self.provider == "anthropic":
            stream_generator = self._generate_anthropic_stream(prompt)
        else:
            yield (
                "Error: LLM provider not configured",
                "Error: LLM provider not configured",
            )
            return

        for chunk in stream_generator:
            # Always yield original chunk for full response tracking
            original_chunk = chunk

            if in_summary:
                # Check if summary ends in this chunk
                combined_for_end = buffer + chunk

                # Try regex match first for robust detection
                end_match = summary_end_regex.search(combined_for_end)
                if end_match:
                    # Summary ended - but don't output anything from summary
                    end_idx = end_match.end()
                    remaining = combined_for_end[end_idx:]
                    in_summary = False
                    buffer = ""
                    # Output remaining content after summary
                    yield original_chunk, remaining if remaining else None
                else:
                    # Fallback to exact pattern match
                    found_end = False
                    for end_pattern in summary_end_patterns:
                        if end_pattern in combined_for_end:
                            end_idx = combined_for_end.find(end_pattern) + len(
                                end_pattern
                            )
                            remaining = combined_for_end[end_idx:]
                            in_summary = False
                            buffer = ""
                            yield original_chunk, remaining if remaining else None
                            found_end = True
                            break

                    if not found_end:
                        # Still in summary, don't output
                        buffer = (
                            chunk[-max_buffer_size:]
                            if len(chunk) > max_buffer_size
                            else chunk
                        )
                        yield original_chunk, None
            else:
                # Not in summary - check if summary starts
                combined = buffer + chunk

                # Try regex match first for robust detection
                start_match = summary_start_regex.search(combined)
                if start_match:
                    # Summary starts in this chunk
                    summary_start_idx = start_match.start()
                    before_summary = combined[:summary_start_idx]
                    in_summary = True
                    buffer = combined[summary_start_idx:]
                    yield original_chunk, before_summary if before_summary else None
                else:
                    # Fallback to exact pattern match
                    summary_start_idx = -1
                    for start_pattern in summary_start_patterns:
                        idx = combined.find(start_pattern)
                        if idx != -1:
                            summary_start_idx = idx
                            break

                    if summary_start_idx != -1:
                        # Summary starts in this chunk
                        before_summary = combined[:summary_start_idx]
                        in_summary = True
                        buffer = combined[summary_start_idx:]
                        yield original_chunk, before_summary if before_summary else None
                    else:
                        # No summary start detected
                        # Check if chunk might contain partial tag (extended check for regex patterns)
                        might_be_partial = (
                            any(
                                combined.endswith(pattern[:i])
                                for pattern in summary_start_patterns
                                for i in range(1, len(pattern))
                            )
                            or combined.rstrip().endswith("<")
                            or combined.rstrip().endswith("*")
                        )

                        if might_be_partial:
                            # Hold back potential partial tag
                            safe_output = buffer
                            buffer = chunk
                            yield original_chunk, safe_output if safe_output else None
                        else:
                            # Safe to output
                            output = buffer + chunk
                            buffer = ""
                            yield original_chunk, output

        # Flush remaining buffer (if not in summary)
        if buffer and not in_summary:
            yield "", buffer

    def _prepare_context(self, elements: list[dict[str, Any]]) -> str:
        """Prepare context from retrieved elements"""
        return _context.prepare_context(
            elements,
            include_file_paths=self.include_file_paths,
            include_line_numbers=self.include_line_numbers,
        )

    def _build_prompt(
        self,
        query: str,
        context: str,
        query_info: dict[str, Any] | None = None,
        dialogue_history: list[dict[str, Any]] | None = None,
    ) -> str:
        """Build complete prompt for LLM"""

        # Base system prompt
        base_system_prompt = """You are a helpful AI assistant specialized in code understanding and explanation.
Your task is to answer questions about code repositories based on the relevant code snippets provided.
You may be working with code from multiple repositories, so pay attention to repository names.

Guidelines:
1. Focus primarily on answering the question itself.
2. The provided code/file content may be irrelevant to the original question or may contain noise. In this case, do not rely on the provided fragment.
3. Provide clear, accurate, and concise answers
4. Reference specific code snippets when relevant
5. Include repository names, file paths, line numbers and corresponding code snippets when discussing specific code
6. If the provided context doesn't contain enough information, say so
7. Use code examples to illustrate your explanations
8. Be technical but accessible
9. If asked to find something, list all relevant locations with their repositories
10. When comparing code from different repositories, clearly distinguish between them
11. **IMPORTANT: Always respond in the same language as the user's question. For example, if the question is in Chinese, respond in Chinese; If in English, respond in English. Match the user's language exactly**."""

        # Multi-turn mode enhancement
        if self.enable_multi_turn and dialogue_history:
            system_prompt = (
                base_system_prompt
                + """

**Multi-turn Dialogue Instructions:**
At the end of your answer, you MUST provide a structured summary for internal use (not shown to the user).
The summary should be enclosed in <SUMMARY> tags and include:
1. Intent: A sentence describing the user's intent in this turn
2. Files Read: List all the files you have analyzed in this conversation
3. Missing Information: Describe what additional files, classes, functions, or context would help answer the query more completely
4. Key Facts: Stable conclusions that can be relied upon in subsequent turns
5. Symbol Mappings: Map user-mentioned names to actual symbols (e.g., "the function" → "utils.process_data")

**IMPORTANT**: Keep the summary under 500 words. Focus on information that helps with code location and reasoning.

Format:
<SUMMARY>
Files Read:
- [repo_name/file_path_1] - [brief description of what was found]
- [repo_name/file_path_2] - [brief description of what was found]

Missing Information:
- [description of what files or context are still needed]
- [why this information would be helpful]

Key Facts:
- [fact 1]
- [fact 2]

Symbol Mappings:
- [user term] → [actual symbol in codebase]
</SUMMARY>

**STRICT FORMAT REQUIREMENT**: You MUST output the summary exactly in the above `<SUMMARY>...</SUMMARY>` structure. Do NOT place content outside the tags. Regardless of the language you use to respond (Chinese, English, or any other language), always use `<SUMMARY>...</SUMMARY>` as the summary tags — do NOT translate or replace them."""
            )
        else:
            system_prompt = base_system_prompt

        # Build user prompt
        user_parts: list[str] = []

        # Add dialogue history context if available
        if dialogue_history and len(dialogue_history) > 0:
            user_parts.append("**Previous Conversation Context**:")

            # Only include recent turns (limited by context_rounds)
            recent_history = (
                dialogue_history[-self.context_rounds :]
                if len(dialogue_history) > self.context_rounds
                else dialogue_history
            )

            for turn in recent_history:
                turn_num = turn.get("turn_number", 0)
                prev_query = turn.get("query", "")
                prev_summary = turn.get("summary", "")

                user_parts.append(f"\n**Turn {turn_num}**")
                user_parts.append(f"User: {prev_query}")
                if prev_summary:
                    user_parts.append(f"Summary: {prev_summary}")

            user_parts.append("\n---\n")

        # Add current query
        user_parts.append(f"**Current Question**: {query}")

        # Add query intent if available
        # if query_info and "intent" in query_info:
        #     intent = query_info["intent"]
        #     user_parts.append(f"\n*(Detected intent: {intent})*")

        # Add context
        user_parts.append("\n**Relevant Code Context**:\n")
        user_parts.append(context)

        # Add instruction
        if self.enable_multi_turn and dialogue_history is not None:
            instruction = (
                "\n**Instructions**: Please answer the question using the code snippets above only if they are relevant. "
                "The code may not always be helpful, so focus on the question itself and refer to specific files or code elements only when necessary. "
                "Remember to include the summary at the end as specified."
            )
        else:
            instruction = (
                "\n**Instructions**: Please answer the question using the code snippets above only if they are relevant. "
                "The code may not always be helpful, so focus on the question itself and refer to specific files or code elements only when necessary. "
            )

        user_parts.append(instruction)

        user_prompt = "\n".join(user_parts)

        # Combine
        full_prompt = f"{system_prompt}\n\n{user_prompt}"

        # print("full_prompt: ", full_prompt)

        return full_prompt

    def _truncate_context(self, context: str, max_tokens: int) -> str:
        """Truncate context to fit within token limit"""
        return truncate_to_tokens(context, max_tokens, self.model or "gpt-4")

    def _generate_openai(self, prompt: str) -> str:
        """Generate answer using OpenAI"""
        if self.client is None:
            return "Error: OpenAI client not initialized"

        try:
            response = openai_chat_completion(
                cast(OpenAI, self.client),
                model=self.model or "gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            # print("response: ", response)

            # Defensive checks because some providers may return partial/None payloads
            if not response or not getattr(response, "choices", None):
                raise ValueError(f"Empty response or no choices returned: {response}")
            first_choice = response.choices[0]
            message = getattr(first_choice, "message", None)
            content = getattr(message, "content", None) if message else None
            if content is None:
                raise ValueError(f"LLM response has no content: {response}")
            return content

        except Exception as e:
            self.logger.error(f"OpenAI API error: {e}")
            raise

    def _generate_openai_stream(self, prompt: str) -> Generator[str, None, None]:
        """Generate answer using OpenAI with streaming"""
        if self.client is None:
            yield "Error: OpenAI client not initialized"
            return

        try:
            response = openai_chat_completion(
                cast(OpenAI, self.client),
                model=self.model or "gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=True,
            )

            for chunk in response:
                if chunk.choices and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta
                    if hasattr(delta, "content") and delta.content:
                        yield delta.content

        except Exception as e:
            self.logger.error(f"OpenAI streaming API error: {e}")
            yield f"\n\nError: {e!s}"

    def _generate_anthropic(self, prompt: str) -> str:
        """Generate answer using Anthropic Claude"""
        if self.client is None:
            return "Error: Anthropic client not initialized"

        try:
            anthropic_client = cast(Anthropic, self.client)
            response = anthropic_client.messages.create(
                model=self.model or "gpt-4",
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=[{"role": "user", "content": prompt}],
            )

            # Defensive checks because some providers may return partial/None payloads
            if not response or not getattr(response, "content", None):
                raise ValueError(f"Empty response or no content returned: {response}")
            first_block = response.content[0] if response.content else None
            text = getattr(first_block, "text", None) if first_block else None
            if text is None:
                raise ValueError(f"LLM response has no text: {response}")
            return text

        except Exception as e:
            self.logger.error(f"Anthropic API error: {e}")
            raise

    def _generate_anthropic_stream(self, prompt: str) -> Generator[str, None, None]:
        """Generate answer using Anthropic Claude with streaming"""
        if self.client is None:
            yield "Error: Anthropic client not initialized"
            return

        try:
            anthropic_client = cast(Anthropic, self.client)
            with anthropic_client.messages.stream(
                model=self.model or "gpt-4",
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=[{"role": "user", "content": prompt}],
            ) as stream:
                for text in stream.text_stream:
                    yield text

        except Exception as e:
            self.logger.error(f"Anthropic streaming API error: {e}")
            yield f"\n\nError: {e!s}"

    def _parse_response_with_summary(self, raw_response: str) -> tuple[str, str | None]:
        """Parse LLM response to extract answer and summary.

        Args:
            raw_response: Raw response from LLM.

        Returns:
            Tuple of (answer, summary).
        """
        return _context.parse_response_with_summary(raw_response)

    def _generate_fallback_summary(
        self, query: str, answer: str, retrieved_elements: list[dict[str, Any]]
    ) -> str:
        """Generate a fallback summary when LLM doesn't produce one."""
        return _summary.generate_fallback_summary(query, answer, retrieved_elements)

    def _extract_sources(self, elements: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Extract source information from elements."""
        return _summary.extract_sources(elements)

    def format_answer_with_sources(self, result: dict[str, Any]) -> str:
        """Format answer with sources for display."""
        return _summary.format_answer_with_sources(result)
