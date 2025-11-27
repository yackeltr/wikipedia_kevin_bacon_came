#!/usr/bin/env python3
"""
Wikipedia Degrees-of-Separation (People-Only)

This command-line tool finds the shortest chain of hyperlinks between two English Wikipedia biography pages,
restricted strictly to people-only nodes—akin to a “Kevin Bacon number” for arbitrary people.

Key features:
- Resolves user-entered names to canonical Wikipedia article titles via search, with exact-match preference.
- Handles redirects and disambiguation pages with interactive selection.
- Enforces a people-only constraint using Wikidata: articles must be "instance of: human" (Q5).
- Computes a shortest hyperlink path via bidirectional BFS, expanding only people pages.
- Provides clear, readable output with path and lengths, and timing information for search and overall steps.
- Robust error handling for search failures, invalid selections, non-people pages, API errors, limits, and Ctrl+C.

Dependencies:
- Standard Python 3 library
- requests (HTTP client)

Algorithm overview:
- Name Resolution: Use MediaWiki API search to get candidate articles. Prefer exact (case-insensitive) title
  matches. Follow redirects to canonical titles. If the resolved page is a disambiguation page, present its
  linked articles for user choice. Resolve until a concrete article title is selected.
- People-Only Verification: For each title, fetch its Wikidata item (via pageprops 'wikibase_item') and query
  its "instance of" (P31) claims. Only Q5 (human) qualifies. This check is cached aggressively.
- Graph Expansion: Bidirectional BFS from start and goal titles. At each expansion, fetch outgoing links from
  the current page (namespace 0, main article space). For each linked title, verify person via Wikidata and
  expand only if human. Maintain parent pointers for forward and backward searches and stop when frontiers meet.
- Path Reconstruction: When a meeting node is found, reconstruct the path from start to meeting (forward parents)
  and from meeting to goal (backward parents), combining into a full people-only path.

Notes:
- The search space in Wikipedia is large; bidirectional BFS reduces expansions significantly compared to single
  BFS. Still, people-only filtering requires Wikidata checks; caching and batch queries mitigate overhead.
- Limits: Practical default limits are set for maximum expansions and depth to keep runtime reasonable. These
  can be tweaked via constants if desired.

Author: (Your Name)
License: MIT
"""

import sys
import time
import traceback
import requests
from collections import deque, defaultdict
from typing import List, Dict, Optional, Tuple, Set

# ----------------------------- Configuration Constants -----------------------------

WIKI_API = "https://en.wikipedia.org/w/api.php"
WIKIDATA_API = "https://www.wikidata.org/w/api.php"

# HTTP behavior
HTTP_TIMEOUT = 15  # seconds per request
MAX_RETRIES = 3    # retries per request
BACKOFF_SECONDS = 1.2

# Search behavior
SEARCH_LIMIT = 10  # max candidates shown to user
DISAMBIGUATION_OPTIONS_LIMIT = 30  # max links presented from disambiguation page

# BFS behavior
MAX_TOTAL_EXPANSIONS = 20000   # hard cap across both directions
MAX_DEPTH = 6                  # max edges allowed in path
LINKS_PAGE_LIMIT = 500         # page size for listing links (MediaWiki limit for 'pllimit' with 'max')

# -----------------------------------------------------------------------------------

class APIError(Exception):
    """Raised when the Wikipedia or Wikidata API returns an error or unexpected response."""
    pass

class UserCancel(Exception):
    """Raised when the user cancels a selection (e.g., pressing Enter without choosing)."""
    pass

# -----------------------------------------------------------------------------------
# HTTP helpers with retries
# -----------------------------------------------------------------------------------

def http_get(url: str, params: Dict) -> Dict:
    """Perform a GET request with retries and return JSON."""
    last_exc = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, params=params, timeout=HTTP_TIMEOUT, headers={"User-Agent": "PeopleDegrees/1.0"})
            resp.raise_for_status()
            return resp.json()
        except (requests.RequestException, ValueError) as e:
            last_exc = e
            if attempt < MAX_RETRIES:
                time.sleep(BACKOFF_SECONDS * attempt)
            else:
                raise APIError(f"HTTP GET failed after {MAX_RETRIES} attempts: {e}") from e
    # Should not reach here
    raise APIError(f"HTTP GET failed: {last_exc}")

# -----------------------------------------------------------------------------------
# Wikipedia client: search, resolve titles, redirects, disambiguation, links
# -----------------------------------------------------------------------------------

class WikipediaClient:
    """
    Lightweight client for Wikipedia's MediaWiki API supporting:
    - search for titles
    - resolve canonical titles, redirects, and detect disambiguations
    - list outgoing links (namespace 0)
    """

    def search_titles(self, query: str, limit: int = SEARCH_LIMIT) -> List[str]:
        """
        Search Wikipedia for a query string and return up to `limit` page titles.
        Prefers main namespace pages.
        """
        params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srnamespace": 0,  # main namespace
            "srlimit": limit,
            "format": "json",
        }
        data = http_get(WIKI_API, params)
        results = data.get("query", {}).get("search", [])
        return [r["title"] for r in results]

    def resolve_title(self, title: str) -> Tuple[str, Dict]:
        """
        Resolve a title to its canonical form, following redirects.
        Returns (resolved_title, page_record).
        """
        params = {
            "action": "query",
            "titles": title,
            "redirects": 1,
            "prop": "info|pageprops|categories",
            "inprop": "url",
            "format": "json",
            "cllimit": 10,
        }
        data = http_get(WIKI_API, params)
        pages = data.get("query", {}).get("pages", {})
        if not pages:
            raise APIError(f"No page found for title '{title}'.")
        # pages is a dict keyed by pageid
        page = next(iter(pages.values()))
        if "missing" in page:
            raise APIError(f"Page missing for title '{title}'.")
        resolved_title = page.get("title") or title
        return resolved_title, page

    def is_disambiguation(self, page: Dict) -> bool:
        """
        Detect if a page is a disambiguation via pageprops or categories.
        """
        props = page.get("pageprops", {})
        if "disambiguation" in props:
            return True
        # Some disambiguations are categorized; check categories if present
        cats = page.get("categories", []) or []
        for c in cats:
            name = c.get("title", "").lower()
            if "disambiguation pages" in name:
                return True
        return False

    def disambiguation_links(self, title: str, limit: int = DISAMBIGUATION_OPTIONS_LIMIT) -> List[str]:
        """
        Fetch outgoing links from a disambiguation page.
        """
        links = []
        plcontinue = None
        while len(links) < limit:
            params = {
                "action": "query",
                "titles": title,
                "prop": "links",
                "plnamespace": 0,
                "pllimit": min(LINKS_PAGE_LIMIT, limit - len(links)),
                "format": "json",
            }
            if plcontinue:
                params["plcontinue"] = plcontinue
            data = http_get(WIKI_API, params)
            pages = data.get("query", {}).get("pages", {})
            if not pages:
                break
            page = next(iter(pages.values()))
            page_links = [l["title"] for l in page.get("links", [])]
            links.extend(page_links)
            plcontinue = data.get("continue", {}).get("plcontinue")
            if not plcontinue:
                break
        return links

    def outgoing_links(self, title: str, max_links: int = 2000) -> List[str]:
        """
        Fetch outgoing links (main namespace) from a page, up to max_links.
        """
        links = []
        plcontinue = None
        while len(links) < max_links:
            params = {
                "action": "query",
                "titles": title,
                "prop": "links",
                "plnamespace": 0,
                "pllimit": min(LINKS_PAGE_LIMIT, max_links - len(links)),
                "format": "json",
            }
            if plcontinue:
                params["plcontinue"] = plcontinue
            data = http_get(WIKI_API, params)
            pages = data.get("query", {}).get("pages", {})
            if not pages:
                break
            page = next(iter(pages.values()))
            page_links = [l["title"] for l in page.get("links", [])]
            links.extend(page_links)
            plcontinue = data.get("continue", {}).get("plcontinue")
            if not plcontinue:
                break
        return links

    def get_wikibase_item(self, title: str) -> Optional[str]:
        """
        Retrieve the Wikidata item ID (Qxxx) for a Wikipedia title via pageprops.
        """
        params = {
            "action": "query",
            "titles": title,
            "prop": "pageprops",
            "format": "json",
        }
        data = http_get(WIKI_API, params)
        pages = data.get("query", {}).get("pages", {})
        if not pages:
            return None
        page = next(iter(pages.values()))
        props = page.get("pageprops", {})
        return props.get("wikibase_item")

# -----------------------------------------------------------------------------------
# Wikidata client: instance-of checks for human (Q5) with caching
# -----------------------------------------------------------------------------------

class WikidataClient:
    """
    Client for querying Wikidata for "instance of" P31 claims to verify if an item is a human (Q5).
    Implements caching and batching for efficiency.
    """

    def __init__(self):
        self.item_human_cache: Dict[str, bool] = {}
        self.title_to_item_cache: Dict[str, Optional[str]] = {}

    def is_human_item(self, item_id: str) -> bool:
        """
        Check (and cache) whether a Wikidata item has P31 includes Q5 (human).
        """
        if item_id in self.item_human_cache:
            return self.item_human_cache[item_id]

        params = {
            "action": "wbgetentities",
            "ids": item_id,
            "props": "claims",
            "format": "json",
        }
        data = http_get(WIKIDATA_API, params)
        entities = data.get("entities", {})
        ent = entities.get(item_id, {})
        claims = ent.get("claims", {})
        p31 = claims.get("P31", [])  # instance of
        is_human = False
        for claim in p31:
            mainsnak = claim.get("mainsnak", {})
            datavalue = mainsnak.get("datavalue", {})
            value = datavalue.get("value", {})
            if value.get("id") == "Q5":  # human
                is_human = True
                break
        self.item_human_cache[item_id] = is_human
        return is_human

    def is_human_title(self, title: str, wiki: WikipediaClient) -> bool:
        """
        Determine if a Wikipedia title corresponds to a human via its Wikidata item.
        Caches title->item and item->is_human.
        """
        if title in self.title_to_item_cache:
            item = self.title_to_item_cache[title]
        else:
            item = wiki.get_wikibase_item(title)
            self.title_to_item_cache[title] = item

        if not item:
            # If no Wikidata item, we conservatively treat as not human to maintain constraints.
            return False

        return self.is_human_item(item)

# -----------------------------------------------------------------------------------
# Name resolution workflow with interactive disambiguation
# -----------------------------------------------------------------------------------

def prompt_choice(options: List[str], prompt: str) -> str:
    """
    Prompt the user to choose an option by number.
    Pressing Enter without a selection raises UserCancel.
    """
    if not options:
        raise APIError("No options available to choose from.")
    print(prompt)
    for i, opt in enumerate(options, 1):
        print(f"  {i}. {opt}")
    print("Press Enter to cancel.")
    choice = input("> ").strip()
    if choice == "":
        raise UserCancel("Selection cancelled by user.")
    try:
        idx = int(choice)
        if idx < 1 or idx > len(options):
            raise ValueError()
    except ValueError:
        raise APIError("Invalid selection. Please enter a valid number.")
    return options[idx - 1]

def resolve_person_title(name: str, wiki: WikipediaClient, wd: WikidataClient) -> str:
    """
    Resolve a user-entered name to a canonical Wikipedia title for a human.
    Steps:
    - Search for candidates; prefer exact case-insensitive title match.
    - Let the user choose among top results if multiple plausible matches exist.
    - Follow redirects to canonical titles.
    - If resolved to a disambiguation page, present its links and ask user to choose one.
    - Verify the chosen article is a human via Wikidata (Q5).
    """
    candidates = wiki.search_titles(name, limit=SEARCH_LIMIT)
    if not candidates:
        raise APIError(f"No search results found for '{name}'.")

    # Prefer exact match (case-insensitive)
    lower = name.strip().lower()
    exact = next((t for t in candidates if t.lower() == lower), None)
    chosen = exact if exact else prompt_choice(candidates, f"Select the article for '{name}':")

    # Resolve to canonical title and check disambiguation
    resolved_title, page = wiki.resolve_title(chosen)
    if wiki.is_disambiguation(page):
        # Offer options from disambiguation page
        links = wiki.disambiguation_links(resolved_title, limit=DISAMBIGUATION_OPTIONS_LIMIT)
        if not links:
            raise APIError(f"Disambiguation page '{resolved_title}' has no links to select.")
        chosen = prompt_choice(links, f"'{resolved_title}' is a disambiguation page. Choose a specific article:")
        resolved_title, page = wiki.resolve_title(chosen)

    # Verify human via Wikidata
    if not wd.is_human_title(resolved_title, wiki):
        raise APIError(f"'{resolved_title}' is not classified as a person (human) on Wikidata.")
    return resolved_title

# -----------------------------------------------------------------------------------
# Bidirectional BFS (people-only)
# -----------------------------------------------------------------------------------

def bidirectional_people_bfs(
    start: str,
    goal: str,
    wiki: WikipediaClient,
    wd: WikidataClient,
    max_depth: int = MAX_DEPTH,
    max_expansions: int = MAX_TOTAL_EXPANSIONS,
) -> Optional[List[str]]:
    """
    Compute the shortest hyperlink path between start and goal titles using bidirectional BFS.
    Constraints:
    - Only expand through nodes verified as humans via Wikidata.
    - Stop when frontiers meet or limits are exceeded.

    Returns:
        List of titles from start to goal inclusive if found, else None.
    """

    if start == goal:
        return [start]

    # Parent pointers
    parents_fwd: Dict[str, Optional[str]] = {start: None}
    parents_bwd: Dict[str, Optional[str]] = {goal: None}

    # Frontiers
    frontier_fwd: Set[str] = {start}
    frontier_bwd: Set[str] = {goal}

    visited_fwd: Set[str] = {start}
    visited_bwd: Set[str] = {goal}

    expansions = 0
    depth = 0

    # Pre-cache human checks for start/goal (already validated), but ensure cache
    wd.title_to_item_cache.setdefault(start, wiki.get_wikibase_item(start))
    wd.item_human_cache[wd.title_to_item_cache[start]] = True if wd.title_to_item_cache[start] else False
    wd.title_to_item_cache.setdefault(goal, wiki.get_wikibase_item(goal))
    wd.item_human_cache[wd.title_to_item_cache[goal]] = True if wd.title_to_item_cache[goal] else False

    # Alternate expansions between forward and backward
    while frontier_fwd and frontier_bwd and expansions < max_expansions and depth < max_depth:
        # Expand the smaller frontier for efficiency
        expand_forward = len(frontier_fwd) <= len(frontier_bwd)

        current_frontier = frontier_fwd if expand_forward else frontier_bwd
        other_frontier = frontier_bwd if expand_forward else frontier_fwd
        parents = parents_fwd if expand_forward else parents_bwd
        visited = visited_fwd if expand_forward else visited_bwd
        visited_other = visited_bwd if expand_forward else visited_fwd

        next_frontier: Set[str] = set()

        for node in current_frontier:
            # List outgoing links
            try:
                neighbors = wiki.outgoing_links(node, max_links=2000)
            except APIError:
                # Skip this node on error but continue search
                continue

            for nbr in neighbors:
                # People-only filter
                try:
                    if not wd.is_human_title(nbr, wiki):
                        continue
                except APIError:
                    # Treat unknown errors as non-people to be safe
                    continue

                if nbr in visited:
                    continue

                parents[nbr] = node
                visited.add(nbr)
                next_frontier.add(nbr)
                expansions += 1

                # Meeting check
                if nbr in visited_other:
                    # Found connection node: reconstruct path
                    meet = nbr
                    path_fwd = reconstruct_path(parents_fwd, start, meet)
                    path_bwd = reconstruct_path(parents_bwd, goal, meet)
                    return path_fwd + path_bwd[::-1][1:]  # merge without duplicating meet

                if expansions >= max_expansions:
                    break
            if expansions >= max_expansions:
                break

        if expand_forward:
            frontier_fwd = next_frontier
        else:
            frontier_bwd = next_frontier

        depth += 1

    return None

def reconstruct_path(parents: Dict[str, Optional[str]], source: str, target: str) -> List[str]:
    """
    Reconstruct a path from source to target using parent pointers.
    """
    path = [target]
    while path[-1] != source:
        parent = parents.get(path[-1])
        if parent is None:
            # Should not happen if target is reachable from source in this tree
            break
        path.append(parent)
    path.reverse()
    return path

# -----------------------------------------------------------------------------------
# CLI orchestration
# -----------------------------------------------------------------------------------

def main() -> int:
    """
    Program entry point. Orchestrates input, resolution, verification, search, and output.
    Returns process exit code (0 on success, non-zero on error).
    """
    print("Wikipedia People Degrees-of-Separation")
    print("Find the shortest chain of people-only hyperlinks between two Wikipedia biographies.")
    print("Note: Disambiguation and candidate selection may prompt for choices.")
    print()

    wiki = WikipediaClient()
    wd = WikidataClient()

    try:
        # Collect user input
        name1 = input("Enter the first person's name: ").strip()
        name2 = input("Enter the second person's name: ").strip()
        if not name1 or not name2:
            print("Both names are required.")
            return 2

        # Start overall timer (post-input)
        overall_t0 = time.perf_counter()

        # Resolve and verify people-only endpoints
        person1 = resolve_person_title(name1, wiki, wd)
        person2 = resolve_person_title(name2, wiki, wd)

        # Search timer
        search_t0 = time.perf_counter()
        path = bidirectional_people_bfs(person1, person2, wiki, wd)
        search_t1 = time.perf_counter()

        if path is None:
            print("\nNo people-only hyperlink path found within the search limits.")
            print(f"Search time: {search_t1 - search_t0:.2f} s")
            print(f"Total elapsed time: {time.perf_counter() - overall_t0:.2f} s")
            return 3

        # Output path
        print("\nShortest people-only hyperlink chain:")
        print(" -> ".join(path))
        print(f"Path length (edges): {max(0, len(path) - 1)}")
        print(f"Search time: {search_t1 - search_t0:.2f} s")
        print(f"Total elapsed time: {time.perf_counter() - overall_t0:.2f} s")
        return 0

    except UserCancel:
        print("\nSelection cancelled by user. Exiting.")
        return 4
    except APIError as e:
        print(f"\nError: {e}")
        return 5
    except KeyboardInterrupt:
        print("\nInterrupted by user (Ctrl+C). Exiting.")
        return 6
    except Exception as e:
        print("\nUnexpected error occurred:")
        print(str(e))
        # Optionally uncomment for debugging:
        # traceback.print_exc()
        return 7

if __name__ == "__main__":
    sys.exit(main())
