import os                          # reads environment variables
import requests                    # makes HTTP requests
from dotenv import load_dotenv     # reads the .env file into environment variables

# load_dotenv() finds the .env file and loads GEMINI_API_KEY, SEMANTIC_SCHOLAR_API_KEY, etc.
# into os.environ so os.environ.get() can read them below.
# This call must happen BEFORE os.environ.get() — otherwise the .env values aren't loaded yet.
load_dotenv()

# Read the API key from .env — gives 100 req/sec instead of 1 req/sec.
# If the key is missing, falls back to "" (empty string = unauthenticated, limited access).
_API_KEY = os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "")

# The base URL for the Semantic Scholar Graph API.
# Storing it as a constant means if the URL ever changes, you fix it in one place.
SEMANTIC_SCHOLAR_BASE_URL = "https://api.semanticscholar.org/graph/v1"


def search_semantic_scholar(query: str, limit: int = 8) -> list[dict]:
    """
    Search Semantic Scholar for papers matching a query.

    Args:
        query  — the topic to search for, e.g. "transformers for time series"
        limit  — how many papers to return (default 8, Semantic Scholar max is 100)

    Returns:
        A list of dicts, one per paper. Each dict has exactly these keys:
            paper_id      — Semantic Scholar's unique ID for the paper
            title         — the paper title
            abstract      — the abstract text (empty string "" if missing)
            authors       — list of author name strings
            year          — publication year (int, or None if unknown)
            citation_count — how many times this paper has been cited (int)
            url           — link to the paper on semanticscholar.org

        Returns an empty list [] if the request fails for any reason.
    """

    # Build the full API endpoint URL:
    # SEMANTIC_SCHOLAR_BASE_URL + "/paper/search"
    # = "https://api.semanticscholar.org/graph/v1/paper/search"
    endpoint = f"{SEMANTIC_SCHOLAR_BASE_URL}/paper/search"

    # "fields" tells Semantic Scholar which data to include in the response.
    # Without this, the API returns only the paperId and title by default.
    # Each field name must exactly match what the API expects.
    params = {
        "query":  query,
        "limit":  limit,
        "fields": "paperId,title,abstract,authors,year,citationCount,url"
    }

    try:
        # The API key goes in the request HEADERS — metadata sent alongside the request.
        # Semantic Scholar reads the "x-api-key" header to identify who is calling.
        # If _API_KEY is empty (key not set), we send no header → unauthenticated (1 req/sec).
        # If _API_KEY has a value, we send the header → authenticated (100 req/sec).
        headers = {"x-api-key": _API_KEY} if _API_KEY else {}

        # params=params adds the query string to the URL automatically, like:
        #   .../paper/search?query=transformers&limit=8&fields=...
        # headers=headers sends the API key alongside the request.
        # timeout=10 means: if the server doesn't respond in 10 seconds, stop waiting.
        response = requests.get(endpoint, params=params, headers=headers, timeout=10)

        # raise_for_status() checks the HTTP status code.
        # If the server returned 4xx (client error) or 5xx (server error),
        # it raises an HTTPError exception immediately.
        # If everything is fine (200 OK), it does nothing.
        response.raise_for_status()

        # response.json() parses the raw JSON text into a Python dictionary.
        # The Semantic Scholar API wraps results in a "data" key, like:
        # { "total": 123, "data": [ {paper1}, {paper2}, ... ] }
        data = response.json()

        papers = []

        # data.get("data", []) safely gets the list of papers.
        # If the "data" key is missing for some reason, we use [] as a fallback.
        for item in data.get("data", []):

            # item.get("abstract") returns None if the field is missing.
            # `or ""` converts None → "" (empty string), which is your spec's requirement.
            # A paper without an abstract still gets into the list — it just has "".
            abstract = item.get("abstract") or ""

            # item.get("authors", []) returns a list of author objects like:
            # [{"authorId": "123", "name": "Ashish Vaswani"}, ...]
            # We only want the name strings, so we loop and grab just ["name"].
            authors = [a["name"] for a in item.get("authors", [])]

            paper = {
                "paper_id":       item.get("paperId", ""),
                "title":          item.get("title", ""),
                "abstract":       abstract,
                "authors":        authors,
                "year":           item.get("year"),           # None if unknown
                "citation_count": item.get("citationCount", 0),
                "url":            item.get("url", ""),
            }

            papers.append(paper)

        return papers

    except requests.exceptions.HTTPError as e:
        # This catches errors from raise_for_status() — e.g. 404, 429, 500.
        # We print the error and return [] so the calling code doesn't crash.
        print(f"[Semantic Scholar] HTTP error: {e}")
        return []

    except requests.exceptions.ConnectionError as e:
        # This catches network failures: no internet, DNS failure, etc.
        print(f"[Semantic Scholar] Connection error: {e}")
        return []

    except requests.exceptions.Timeout:
        # This catches the case where the server took longer than 10 seconds.
        print("[Semantic Scholar] Request timed out after 10 seconds.")
        return []

    except Exception as e:
        # Catch-all for anything unexpected (malformed JSON, etc.)
        print(f"[Semantic Scholar] Unexpected error: {e}")
        return []
