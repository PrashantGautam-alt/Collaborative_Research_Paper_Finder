import json  # built-in Python library for parsing JSON strings into Python objects
import re    # built-in Python library for cleaning text using patterns

from utils.gemini_client import gemini_client, GEMINI_MODEL


class QueryExpander:
    """
    Agent 1 — Query Expander

    Why this agent exists:
        A user types one raw query like "transformers for time series".
        That single phrasing might miss papers that use different terminology
        for the same idea — e.g. "attention mechanisms for forecasting" or
        "sequence models for temporal data".

        By generating 3 alternative phrasings, the Search Agent gets 4 angles
        to search from. More angles = more relevant papers found.

    Input:  one query string  (the user's raw topic)
    Output: list of 4 strings (original query + 3 Gemini-generated alternatives)

    Fallback: if Gemini fails or returns malformed JSON, returns [original_query]
              so the rest of the pipeline can still run with just 1 query.
    """

    def expand(self, query: str) -> list[str]:
        """
        Expand one query into 4 queries.

        Args:
            query — the user's raw search topic

        Returns:
            A list of 4 strings: [original, expansion1, expansion2, expansion3]
            Returns [original] alone if anything goes wrong.
        """

        # We tell Gemini exactly what format to return.
        # Key rules in the prompt:
        #   1. "Return ONLY a JSON array" — no extra words, no explanation
        #   2. "No markdown fences"      — no ```json ... ``` wrapping
        #   3. We give an example        — LLMs follow examples more reliably than instructions alone
        prompt = f"""You are a research query expander.

Given a research topic, generate exactly 3 alternative search phrasings that cover
different aspects or terminology of the same topic.

Return ONLY a JSON array of exactly 3 strings. No explanation. No markdown fences.

Example output format:
["alternative one", "alternative two", "alternative three"]

Topic: {query}"""

        try:
            response = gemini_client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt
            )

            # .strip() removes any leading/trailing whitespace from the response
            response_text = response.text.strip()

            # Gemini sometimes wraps JSON in markdown code fences like this:
            #   ```json
            #   ["phrase one", "phrase two", "phrase three"]
            #   ```
            # json.loads() would crash on those backticks, so we strip them first.
            # re.sub(pattern, replacement, text) replaces all matches of the pattern.
            # This pattern matches: ```json or ``` at the start/end of the JSON block.
            cleaned = re.sub(r'```(?:json)?\s*|\s*```', '', response_text).strip()

            # json.loads() converts the cleaned JSON string into a Python list.
            # "Transformers for NLP" → '["phrase1", "phrase2", "phrase3"]' → ["phrase1", "phrase2", "phrase3"]
            # If the string is not valid JSON, this raises json.JSONDecodeError.
            expansions = json.loads(cleaned)

            # Validate that Gemini actually returned a list and not a dict or string.
            # isinstance() checks if a variable is a specific type.
            if not isinstance(expansions, list):
                raise ValueError(f"Expected a JSON array, got: {type(expansions)}")

            # Prepend the original query so it's always position 0 in the list.
            # expansions[:3] takes at most 3 items in case Gemini returned more.
            return [query] + expansions[:3]

        except json.JSONDecodeError:
            # Gemini returned something that isn't valid JSON at all.
            # We can't parse it, so we fall back to the original query only.
            # The pipeline will still run — just with 1 query instead of 4.
            print(f"[QueryExpander] Gemini response was not valid JSON. Falling back to original query.")
            return [query]

        except Exception as e:
            # Catches anything else: network error, quota error, ValueError above, etc.
            print(f"[QueryExpander] Unexpected error: {e}. Falling back to original query.")
            return [query]
