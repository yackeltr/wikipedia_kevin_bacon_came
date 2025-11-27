#!/usr/bin/env python3
"""
Wikipedia Degrees of Separation Tool

This command-line script finds the shortest directed chain of hyperlinks between two English Wikipedia
articles about individual humans, restricted to paths through other human biography articles. It uses
bidirectional BFS: forward BFS from the start article using outgoing hyperlinks (via prop=links API),
backward BFS from the goal article using incoming hyperlinks (via list=backlinks API). Each potential
node is resolved to its canonical title (following redirects recursively), linked to its Wikidata item,
and verified as a "human" via Wikidata's P31 claim containing Q5 ("instance of: human").

Name resolution:
- Searches Wikipedia for candidates, preferring exact case-insensitive title matches.
- If multiple candidates, prompts user to choose.
- Follows redirects to canonical title.
- If resolves to a disambiguation page (detected via pageprops.disambiguation=='yes'), fetches outgoing
  links (main namespace) and prompts user to choose one, then canonicalizes.

People-only constraint:
- For every article (start, goal, intermediates), fetches Wikidata QID via pageprops.wikibase_item.
- Queries Wikidata wbgetentities for claims.P31; qualifies as person iff Q5 is present in mainsnak values.

Pathfinding:
- Caches canonical titles, QIDs, and person status to minimize API calls.
- During expansion, batches Wikidata queries for new QIDs (up to 50 per call).
- Limits search to 10,000 nodes per direction to prevent excessive computation/time.
- Reconstructs path using parent pointers once overlap (meeting node) found.

Output:
- Chain as "Title1 -> Title2 -> ... -> TitleN"
- Number of steps (edges).
- Timing: search (BFS only), total (post-inputs).

Error handling:
- Graceful exit (code 0) for no results, invalid choices, non-person pages, no path.
- Retries API calls (3x), exits (1) on persistent failures.
- Handles Ctrl+C with message, exit 1.

Dependencies: requests (for HTTP/API), standard library otherwise.
No knowledge cutoff; uses live Wikipedia/Wikidata APIs.
"""

import sys
import time
import requests
from collections import deque

# Global caches for efficiency
caches = {
    'canonical': {},
    'title_to_qid': {},
    'qid_to_person': {}
}

WIKIPEDIA_API = 'https://en.wikipedia.org/w/api.php'
WIKIDATA_API = 'https://www.wikidata.org/w/api.php'

def wikipedia_query(params, retries=3):
    """Query Wikipedia API with retries."""
    for attempt in range(retries):
        try:
            r = requests.get(WIKIPEDIA_API, {'format': 'json', **params}, timeout=30)
            r.raise_for_status()
            data = r.json()
            if 'error' in data:
                raise ValueError(f"API error: {data['error']}")
            return data
        except Exception as e:
            if attempt == retries - 1:
                raise
            time.sleep(1)
    raise Exception("Failed to query Wikipedia API after retries")

def wikidata_query(params, retries=3):
    """Query Wikidata API with retries."""
    for attempt in range(retries):
        try:
            r = requests.get(WIKIDATA_API, {'format': 'json', **params}, timeout=30)
            r.raise_for_status()
            data = r