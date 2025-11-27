#!/usr/bin/env python3
"""
People-Only Wikipedia Degrees-of-Separation Finder
with Link Verification

Requirements:
    - Python 3.8+
    - requests

Install dependency (inside your virtual environment):
    python -m pip install requests

Overview:
    This script finds the shortest chain of hyperlinks between two people on
    English Wikipedia, with the constraint that *every* node in the chain
    must be a person (human) according to Wikidata (instance of Q5).

    It then verifies each hop in the chain by:
        - Fetching the HTML for the source article.
        - Locating the actual hyperlink (<a href="/wiki/Target_Title">).
        - Printing the exact anchor tag and a small HTML snippet around it.
"""

import sys
import time
from typing import Dict, List, Optional, Set, Tuple

import requests

# Suppress the noisy NotOpenSSLWarning from urllib3 v2 on macOS LibreSSL builds.
try:
    from urllib3.exceptions import NotOpenSSLWarning
    import warnings

    warnings.filterwarnings("ignore", category=NotOpenSSLWarning)
except Exception:
    # If urllib3 isn't structured as expected, just ignore; it's not fatal.
    pass


WIKIPEDIA_API_URL = "https://en.wikipedia.org/w/api.php"
WIKIDATA_API_URL = "https://www.wikidata.org/w/api.php"


# ---------------------------------------------------------------------------
# Custom exception types
# ---------------------------------------------------------------------------


class WikipediaAPIError(Exception):
    """Raised when an unrecoverable error occurs while talking to Wikipedia/Wikidata APIs."""


class PageNotFoundError(Exception):
    """Raised when a Wikipedia page cannot be found for the given query or title."""


class UserAbortError(Exception):
    """Raised when the user cancels a selection (e.g., chooses 'None of the above')."""


class SearchTimeoutError(Exception):
    """Raised when the graph search exceeds the configured timeout or expansion limits."""


# ---------------------------------------------------------------------------
# Wikipedia + Wikidata client
# ---------------------------------------------------------------------------


class WikipediaClient:
    """
    Wrapper around the Wikipedia and Wikidata APIs.

    Responsibilities:
        - Search for pages by name.
        - Resolve titles, including redirects and disambiguation pages.
        - Retrieve outgoing links and incoming links (backlinks).
        - Determine if a page is a human via Wikidata (instance of Q5).
        - Cache all of the above aggressively to reduce network calls.
        - Fetch raw HTML for verification.
    """

    def __init__(self, language: str = "en", timeout: int = 10) -> None:
        self.api_url = WIKIPEDIA_API_URL.replace("//en.", f"//{language}.")
        self.timeout = timeout

        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": (
                    "PeopleOnlyWikipediaDegrees/1.2 "
                    "(Python script for shortest path computation with link verification)"
                )
            }
        )

        self.links_from_cache: Dict[str, Set[str]] = {}
        self.links_to_cache: Dict[str, Set[str]] = {}
        self.human_cache: Dict[str, bool] = {}
        self.wikidata_id_cache: Dict[str, Optional[str]] = {}

    # ------------------------ Low-level API helpers ------------------------

    def _api_get(self, params: Dict, max_retries: int = 3) -> Dict:
        params = dict(params)
        params.setdefault("format", "json")
        params.setdefault("formatversion", "2")

        last_exc: Optional[Exception] = None

        for _ in range(max_retries):
            try:
                resp = self.session.get(self.api_url, params=params, timeout=self.timeout)
                resp.raise_for_status()
                data = resp.json()
                if "error" in data:
                    raise WikipediaAPIError(f"Wikipedia API error: {data['error']}")
                return data
            except (requests.RequestException, ValueError) as exc:
                last_exc = exc
                time.sleep(0.5)

        raise WikipediaAPIError(
            f"Failed to contact Wikipedia API after {max_retries} attempts: {last_exc}"
        )

    def _wikidata_get(self, params: Dict, max_retries: int = 3) -> Dict:
        params = dict(params)
        params.setdefault("format", "json")

        last_exc: Optional[Exception] = None

        for _ in range(max_retries):
            try:
                resp = self.session.get(WIKIDATA_API_URL, params=params, timeout=self.timeout)
                resp.raise_for_status()
                data = resp.json()
                if "error" in data:
                    raise WikipediaAPIError(f"Wikidata API error: {data['error']}")
                return data
            except (requests.RequestException, ValueError) as exc:
                last_exc = exc
                time.sleep(0.5)

        raise WikipediaAPIError(
            f"Failed to contact Wikidata API after {max_retries} attempts: {last_exc}"
        )

    # ------------------------ Wikipedia search & page helpers ------------------------

    def search(self, query: str, limit: int = 5) -> List[str]:
        params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": limit,
            "srnamespace": 0,
        }
        data = self._api_get(params)
        results = data.get("query", {}).get("search", [])
        if not results:
            raise PageNotFoundError(f"No Wikipedia pages found for: {query!r}")
        return [r["title"] for r in results]

    def _get_page_info(self, title: str) -> Dict:
        params = {
            "action": "query",
            "prop": "pageprops",
            "titles": title,
            "redirects": 1,
        }
        data = self._api_get(params)
        pages = data.get("query", {}).get("pages", [])
        if not pages:
            raise PageNotFoundError(f"No page found for title: {title!r}")
        page = pages[0]
        if "missing" in page:
            raise PageNotFoundError(f"No page found for title: {title!r}")
        return page

    @staticmethod
    def _is_disambiguation(page: Dict) -> bool:
        props = page.get("pageprops", {})
        return "disambiguation" in props

    def _prompt_for_choice(
        self,
        header: str,
        options: List[str],
        default_index: int = 1,
        allow_none_option: bool = True,
    ) -> str:
        print(header)
        for i, opt in enumerate(options, start=1):
            print(f"  [{i}] {opt}")
        if allow_none_option:
            print("  [0] None of the above")

        while True:
            raw = input(f"Select an option (default {default_index}): ").strip()
            if raw == "":
                choice = default_index
            else:
                if not raw.isdigit():
                    print("Please enter a valid number.")
                    continue
                choice = int(raw)

            if allow_none_option and choice == 0:
                raise UserAbortError("User selected 'None of the above'.")
            if 1 <= choice <= len(options):
                return options[choice - 1]
            print(f"Please enter a number between 0 and {len(options)}.")

    def resolve_title(self, query: str) -> str:
        candidates = self.search(query, limit=5)

        lowered = query.strip().lower()
        exact_matches = [c for c in candidates if c.lower() == lowered]

        if exact_matches:
            chosen = exact_matches[0]
        else:
            chosen = self._prompt_for_choice(
                header=f"\nMultiple matches found for '{query}':",
                options=candidates,
                default_index=1,
                allow_none_option=True,
            )

        page = self._get_page_info(chosen)
        resolved_title = page["title"]

        if self._is_disambiguation(page):
            links = sorted(self.get_links_from(resolved_title))
            if not links:
                raise PageNotFoundError(
                    f"Disambiguation page '{resolved_title}' has no linked articles."
                )
            limited_links = links[:50]
            chosen = self._prompt_for_choice(
                header=(
                    f"\n'{resolved_title}' is a disambiguation page. "
                    "Please choose a more specific article:"
                ),
                options=limited_links,
                default_index=1,
                allow_none_option=True,
            )
            page = self._get_page_info(chosen)
            resolved_title = page["title"]

        return resolved_title

    # ------------------------ Link graph helpers ------------------------

    def get_links_from(self, title: str) -> Set[str]:
        if title in self.links_from_cache:
            return self.links_from_cache[title]

        params = {
            "action": "query",
            "prop": "links",
            "titles": title,
            "plnamespace": 0,
            "pllimit": "max",
        }

        linked_titles: Set[str] = set()

        while True:
            data = self._api_get(params)
            pages = data.get("query", {}).get("pages", [])
            for page in pages:
                for link in page.get("links", []):
                    linked_titles.add(link["title"])
            cont = data.get("continue")
            if not cont:
                break
            params.update(cont)

        self.links_from_cache[title] = linked_titles
        return linked_titles

    def get_links_to(self, title: str) -> Set[str]:
        if title in self.links_to_cache:
            return self.links_to_cache[title]

        params = {
            "action": "query",
            "list": "backlinks",
            "bltitle": title,
            "blnamespace": 0,
            "bllimit": "max",
        }

        backlinks: Set[str] = set()

        while True:
            data = self._api_get(params)
            bl = data.get("query", {}).get("backlinks", [])
            for item in bl:
                backlinks.add(item["title"])
            cont = data.get("continue")
            if not cont:
                break
            params.update(cont)

        self.links_to_cache[title] = backlinks
        return backlinks

    # ------------------------ Wikidata human detection ------------------------

    def _get_wikidata_id(self, title: str) -> Optional[str]:
        if title in self.wikidata_id_cache:
            return self.wikidata_id_cache[title]

        try:
            page = self._get_page_info(title)
        except PageNotFoundError:
            self.wikidata_id_cache[title] = None
            return None

        props = page.get("pageprops", {})
        wd_id = props.get("wikibase_item")
        self.wikidata_id_cache[title] = wd_id
        return wd_id

    def is_human(self, title: str) -> bool:
        if title in self.human_cache:
            return self.human_cache[title]

        wd_id = self._get_wikidata_id(title)
        if not wd_id:
            self.human_cache[title] = False
            return False

        params = {
            "action": "wbgetentities",
            "ids": wd_id,
            "props": "claims",
        }

        try:
            data = self._wikidata_get(params)
        except WikipediaAPIError:
            self.human_cache[title] = False
            return False

        entities = data.get("entities", {})
        entity = entities.get(wd_id, {})
        claims = entity.get("claims", {})
        p31 = claims.get("P31", [])

        is_human_flag = False
        for claim in p31:
            mainsnak = claim.get("mainsnak", {})
            datavalue = mainsnak.get("datavalue", {})
            value = datavalue.get("value", {})
            if isinstance(value, dict) and value.get("id") == "Q5":
                is_human_flag = True
                break

        self.human_cache[title] = is_human_flag
        return is_human_flag

    # ------------------------ HTML fetch for verification ------------------------

    def fetch_html(self, title: str) -> Optional[str]:
        """
        Fetch the raw HTML for an article by title, for verification purposes.
        Returns the HTML as a string, or None on failure.
        """
        url = f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"
        try:
            resp = self.session.get(url, timeout=self.timeout)
            resp.raise_for_status()
            return resp.text
        except requests.RequestException:
            return None


# ---------------------------------------------------------------------------
# Bidirectional BFS (people-only)
# ---------------------------------------------------------------------------


def bidirectional_shortest_path(
    client: WikipediaClient,
    start: str,
    goal: str,
    max_expanded: int = 200_000,
    timeout_seconds: int = 600,
) -> Tuple[List[str], int]:
    """
    Bidirectional BFS over the people-only Wikipedia graph.

    Nodes:
        - Wikipedia article titles that are humans via Wikidata (Q5).

    Edges:
        - Directed hyperlinks between those articles.

    Returns:
        (path, expanded_count):
            - path: start → ... → goal, or [] if not found.
            - expanded_count: number of nodes expanded during search.
    """
    if start == goal:
        return [start], 0

    if not client.is_human(start):
        raise ValueError(f"Start page '{start}' is not classified as a human in Wikidata.")
    if not client.is_human(goal):
        raise ValueError(f"Goal page '{goal}' is not classified as a human in Wikidata.")

    visited_forward: Dict[str, Optional[str]] = {start: None}
    visited_backward: Dict[str, Optional[str]] = {goal: None}

    frontier_forward: Set[str] = {start}
    frontier_backward: Set[str] = {goal}

    start_time = time.time()
    expanded_count = 0

    while frontier_forward and frontier_backward:
        now = time.time()
        if (now - start_time) > timeout_seconds or expanded_count > max_expanded:
            raise SearchTimeoutError(
                "Search exceeded allowed time or expansion limit without finding a path."
            )

        if len(frontier_forward) <= len(frontier_backward):
            current_frontier = frontier_forward
            next_frontier: Set[str] = set()

            for node in current_frontier:
                expanded_count += 1
                neighbors = client.get_links_from(node)

                for nbr in neighbors:
                    if not client.is_human(nbr):
                        continue
                    if nbr not in visited_forward:
                        visited_forward[nbr] = node
                        next_frontier.add(nbr)
                    if nbr in visited_backward:
                        path = _reconstruct_path(nbr, visited_forward, visited_backward)
                        return path, expanded_count

            frontier_forward = next_frontier

        else:
            current_frontier = frontier_backward
            next_frontier = set()

            for node in current_frontier:
                expanded_count += 1
                neighbors = client.get_links_to(node)

                for nbr in neighbors:
                    if not client.is_human(nbr):
                        continue
                    if nbr not in visited_backward:
                        visited_backward[nbr] = node
                        next_frontier.add(nbr)
                    if nbr in visited_forward:
                        path = _reconstruct_path(nbr, visited_forward, visited_backward)
                        return path, expanded_count

            frontier_backward = next_frontier

    return [], expanded_count


def _reconstruct_path(
    meeting: str,
    visited_forward: Dict[str, Optional[str]],
    visited_backward: Dict[str, Optional[str]],
) -> List[str]:
    forward_path: List[str] = []
    cur: Optional[str] = meeting
    while cur is not None:
        forward_path.append(cur)
        cur = visited_forward.get(cur)
    forward_path.reverse()

    backward_path: List[str] = []
    cur = meeting
    while visited_backward.get(cur) is not None:
        cur = visited_backward[cur]
        backward_path.append(cur)

    if forward_path and backward_path and forward_path[-1] == backward_path[0]:
        return forward_path + backward_path[1:]
    return forward_path + backward_path


# ---------------------------------------------------------------------------
# Verification of links: show URLs + exact anchor and context
# ---------------------------------------------------------------------------


def _extract_anchor_and_context(html: str, dst_slug: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Given the HTML of a page and a destination slug (e.g. 'Neil_Diamond'),
    try to extract:
        - The full <a ...>...</a> tag that links to that slug.
        - A short context snippet around it.

    Returns:
        (anchor_tag, context_snippet) or (None, None) if not found.
    """
    needle = f'/wiki/{dst_slug}'
    idx = html.find(needle)
    if idx == -1:
        return None, None

    # Try to find the start of the enclosing <a ...> tag.
    a_start = html.rfind("<a", 0, idx)
    a_end = html.find("</a>", idx)
    if a_start == -1 or a_end == -1:
        return None, None
    a_end += len("</a>")
    anchor = html[a_start:a_end]

    # Build a short, cleaned context snippet around the anchor.
    context_radius = 140
    context_start = max(0, a_start - context_radius)
    context_end = min(len(html), a_end + context_radius)
    snippet = html[context_start:context_end]
    snippet = snippet.replace("\n", " ")
    snippet = " ".join(snippet.split())
    if len(snippet) > 280:
        snippet = snippet[:280] + "..."

    return anchor, snippet


def verify_path_links(client: WikipediaClient, path: List[str]) -> None:
    """
    For each consecutive pair (A -> B) in the path:
        - Print the source and target URLs.
        - Fetch the HTML of A.
        - Locate an <a href="/wiki/B_Title"> anchor.
        - Print the anchor tag and a short context snippet.
    """
    if len(path) < 2:
        print("\nNo edges to verify (path length < 2).")
        return

    print("\n========================================")
    print("        LINK VERIFICATION SNIPPETS      ")
    print("========================================")

    for i in range(len(path) - 1):
        src = path[i]
        dst = path[i + 1]
        src_slug = src.replace(" ", "_")
        dst_slug = dst.replace(" ", "_")
        src_url = f"https://en.wikipedia.org/wiki/{src_slug}"
        dst_url = f"https://en.wikipedia.org/wiki/{dst_slug}"

        print(f"\nStep {i + 1}: {src} → {dst}")
        print(f"  Source URL: {src_url}")
        print(f"  Target URL: {dst_url}")

        html = client.fetch_html(src)
        if html is None:
            print("  [!] Could not fetch HTML for source page; skipping snippet.")
            continue

        anchor, context = _extract_anchor_and_context(html, dst_slug)
        if anchor is None:
            print(
                "  [!] No direct '/wiki/Target' anchor found in the current HTML. "
                "Link may be indirect, template-generated, or changed."
            )
            continue

        print("  Anchor tag found:")
        print(f"    {anchor}")
        print("  Context snippet:")
        print(f"    ...{context}...")

    print("========================================\n")


# ---------------------------------------------------------------------------
# Main user-facing script
# ---------------------------------------------------------------------------


def main() -> None:
    print("======================================================")
    print("      PEOPLE-ONLY WIKIPEDIA DEGREES OF SEPARATION     ")
    print("======================================================")
    print("Finds the shortest path between two people.")
    print("Constraint: Every step must be a Human (Wikidata Q5).")
    print("------------------------------------------------------")

    try:
        raw_a = input("Enter Person A: ").strip()
        raw_b = input("Enter Person B: ").strip()

        if not raw_a or not raw_b:
            print("Both names must be non-empty.")
            sys.exit(1)

        overall_start = time.time()
        client = WikipediaClient(language="en", timeout=10)

        print(f"\nResolving '{raw_a}'...")
        title_a = client.resolve_title(raw_a)
        print(f"Resolved Person A to: {title_a}")

        print(f"\nResolving '{raw_b}'...")
        title_b = client.resolve_title(raw_b)
        print(f"Resolved Person B to: {title_b}")

        print("\nVerifying endpoints are people...")
        if not client.is_human(title_a):
            print(f"'{title_a}' is not classified as a human in Wikidata. Exiting.")
            sys.exit(1)
        if not client.is_human(title_b):
            print(f"'{title_b}' is not classified as a human in Wikidata. Exiting.")
            sys.exit(1)
        print(f"Endpoints verified. Starting search: {title_a} <-> {title_b}")
        print("Note: Expansion speed is limited by Wikidata 'is-human' checks.\n")

        if title_a == title_b:
            total_time = time.time() - overall_start
            print("Both names resolve to the same person. No path needed.")
            print(f"Total Time: {total_time:.2f} sec")
            sys.exit(0)

        search_start = time.time()
        path, expanded = bidirectional_shortest_path(
            client,
            start=title_a,
            goal=title_b,
            max_expanded=200_000,
            timeout_seconds=600,
        )
        search_time = time.time() - search_start
        total_time = time.time() - overall_start

        if not path:
            print("No people-only hyperlink path found within the search constraints.")
            print(f"Search Time:    {search_time:.2f} sec")
            print(f"Total Time:     {total_time:.2f} sec")
            print(f"Nodes Expanded: {expanded}")
            sys.exit(0)

        if path[0] != title_a or path[-1] != title_b:
            print("\n[WARNING] Internal path does not start/end with requested endpoints.")
            print("Computed path:", " -> ".join(path))

        degrees = len(path) - 1

        print("\n========================================")
        print("              PATH FOUND!              ")
        print("========================================")
        print(f"Degrees of Separation: {degrees}")
        print("----------------------------------------")
        for idx, title in enumerate(path, start=1):
            print(f"{idx}. {title}")
            if idx < len(path):
                print("   |")
                print("   v")
        print("----------------------------------------")
        print(f"Search Time:    {search_time:.2f} sec")
        print(f"Total Time:     {total_time:.2f} sec")
        print(f"Nodes Expanded: {expanded}")
        print("========================================")

        # Verify hyperlinks with parsed anchors + context.
        verify_path_links(client, path)

    except PageNotFoundError as e:
        print(f"\nError: {e}")
        sys.exit(1)
    except UserAbortError as e:
        print(f"\nAborted: {e}")
        sys.exit(1)
    except SearchTimeoutError as e:
        print(f"\nSearch stopped: {e}")
        sys.exit(1)
    except WikipediaAPIError as e:
        print(f"\nWikipedia / Wikidata API error: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"\nError: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        sys.exit(1)


if __name__ == "__main__":
    main()
