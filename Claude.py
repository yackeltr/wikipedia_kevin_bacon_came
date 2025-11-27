#!/usr/bin/env python3
"""
Wikipedia Degrees of Separation (People-Only Edition)

This script finds the shortest chain of hyperlinks between two people's
Wikipedia biography pages, constrained to people-only nodes (similar to
the "Six Degrees of Kevin Bacon" game).

Algorithm:
- Uses bidirectional BFS to explore from both endpoints simultaneously
- Each node must be verified as a person via Wikidata (instance of Q5: human)
- Only follows links to other person articles
- Stops when the two search frontiers meet at a common person

The people-only constraint is enforced by:
1. Checking start and goal are people (via Wikidata P31 = Q5)
2. Filtering all outbound links to only include person articles
3. Building a person-to-person graph dynamically during search
"""

import sys
import time
import urllib.request
import urllib.parse
import urllib.error
import json
from collections import deque
from typing import List, Dict, Set, Tuple, Optional


# ============================================================================
# Configuration Constants
# ============================================================================

USER_AGENT = "WikipediaDegreesOfSeparation/1.0 (Educational Tool)"
WIKIPEDIA_API = "https://en.wikipedia.org/w/api.php"
WIKIDATA_API = "https://www.wikidata.org/w/api.php"

# Search parameters
MAX_SEARCH_RESULTS = 5
MAX_DISAMBIGUATION_OPTIONS = 15

# BFS limits to prevent infinite search
MAX_BFS_DEPTH = 6  # Maximum path length (edges)
MAX_NODES_EXPLORED = 50000  # Maximum total nodes to explore

# Performance optimization: cache person checks and links
_person_cache = {}  # Cache for is_person checks
_links_cache = {}   # Cache for person links


# ============================================================================
# HTTP Helper Functions
# ============================================================================

def make_api_request(url: str, params: Dict[str, str], max_retries: int = 3) -> Dict:
    """
    Make an HTTP GET request to a MediaWiki API endpoint.
    
    Args:
        url: API endpoint URL
        params: Query parameters
        max_retries: Maximum number of retry attempts
        
    Returns:
        Parsed JSON response as a dictionary
        
    Raises:
        SystemExit on repeated failures
    """
    params['format'] = 'json'
    query_string = urllib.parse.urlencode(params)
    full_url = f"{url}?{query_string}"
    
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(full_url, headers={'User-Agent': USER_AGENT})
            with urllib.request.urlopen(req, timeout=10) as response:
                return json.loads(response.read().decode('utf-8'))
        except urllib.error.HTTPError as e:
            if attempt == max_retries - 1:
                print(f"Error: HTTP {e.code} - {e.reason}", file=sys.stderr)
                sys.exit(1)
            time.sleep(1)
        except urllib.error.URLError as e:
            if attempt == max_retries - 1:
                print(f"Error: Network error - {e.reason}", file=sys.stderr)
                sys.exit(1)
            time.sleep(1)
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"Error: Unexpected error - {e}", file=sys.stderr)
                sys.exit(1)
            time.sleep(1)
    
    return {}


# ============================================================================
# Wikipedia Article Resolution
# ============================================================================

def search_wikipedia(query: str) -> List[str]:
    """
    Search Wikipedia for articles matching the query.
    
    Args:
        query: Search term (person's name)
        
    Returns:
        List of article titles (up to MAX_SEARCH_RESULTS)
    """
    params = {
        'action': 'opensearch',
        'search': query,
        'limit': str(MAX_SEARCH_RESULTS),
        'namespace': '0',  # Main namespace only
        'redirects': 'resolve'
    }
    
    data = make_api_request(WIKIPEDIA_API, params)
    
    # opensearch returns [query, [titles], [descriptions], [urls]]
    if len(data) >= 2 and isinstance(data[1], list):
        return data[1]
    return []


def get_canonical_title(title: str) -> str:
    """
    Resolve a Wikipedia title to its canonical form (following redirects).
    
    Args:
        title: Wikipedia article title
        
    Returns:
        Canonical article title after following redirects
    """
    params = {
        'action': 'query',
        'titles': title,
        'redirects': '1'
    }
    
    data = make_api_request(WIKIPEDIA_API, params)
    
    pages = data.get('query', {}).get('pages', {})
    for page_id, page_data in pages.items():
        if page_id != '-1':  # -1 means page doesn't exist
            return page_data.get('title', title)
    
    return title


def is_disambiguation_page(title: str) -> bool:
    """
    Check if a Wikipedia page is a disambiguation page.
    
    Args:
        title: Wikipedia article title
        
    Returns:
        True if the page is a disambiguation page
    """
    params = {
        'action': 'query',
        'titles': title,
        'prop': 'pageprops'
    }
    
    data = make_api_request(WIKIPEDIA_API, params)
    
    pages = data.get('query', {}).get('pages', {})
    for page_data in pages.values():
        pageprops = page_data.get('pageprops', {})
        if 'disambiguation' in pageprops:
            return True
    
    return False


def get_disambiguation_links(title: str) -> List[str]:
    """
    Extract article links from a disambiguation page.
    
    Args:
        title: Title of a disambiguation page
        
    Returns:
        List of article titles linked from the disambiguation page
    """
    params = {
        'action': 'parse',
        'page': title,
        'prop': 'links',
        'redirects': '1'
    }
    
    data = make_api_request(WIKIPEDIA_API, params)
    
    links = data.get('parse', {}).get('links', [])
    
    # Filter to main namespace (ns=0) and exclude special pages
    article_links = []
    for link in links:
        if link.get('ns') == 0 and link.get('exists', '') != '':
            article_links.append(link.get('*', ''))
    
    return article_links[:MAX_DISAMBIGUATION_OPTIONS]


def resolve_person_name(name: str) -> Optional[str]:
    """
    Resolve a person's name to a canonical Wikipedia article title.
    Handles search, disambiguation, and user selection.
    
    Args:
        name: Person's name as entered by user
        
    Returns:
        Canonical Wikipedia article title, or None if resolution fails
    """
    print(f"\nResolving '{name}'...")
    
    # Search Wikipedia
    search_results = search_wikipedia(name)
    
    if not search_results:
        print(f"Error: No Wikipedia articles found for '{name}'")
        return None
    
    # Check for exact match (case-insensitive)
    exact_match = None
    for title in search_results:
        if title.lower() == name.lower():
            exact_match = title
            break
    
    # If we have an exact match, use it; otherwise present options
    if exact_match and len(search_results) == 1:
        selected_title = exact_match
    elif exact_match:
        print(f"Found exact match: {exact_match}")
        print(f"Using: {exact_match}")
        selected_title = exact_match
    else:
        # Multiple results, ask user to choose
        print(f"Found {len(search_results)} possible articles:")
        for i, title in enumerate(search_results, 1):
            print(f"  {i}. {title}")
        
        while True:
            choice = input(f"Select article (1-{len(search_results)}): ").strip()
            if not choice:
                print("Selection cancelled.")
                return None
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(search_results):
                    selected_title = search_results[idx]
                    break
                else:
                    print(f"Please enter a number between 1 and {len(search_results)}")
            except ValueError:
                print("Please enter a valid number")
    
    # Follow redirects to get canonical title
    canonical_title = get_canonical_title(selected_title)
    
    # Check if it's a disambiguation page
    if is_disambiguation_page(canonical_title):
        print(f"\n'{canonical_title}' is a disambiguation page.")
        disambig_links = get_disambiguation_links(canonical_title)
        
        if not disambig_links:
            print("Error: Could not extract links from disambiguation page")
            return None
        
        print(f"Found {len(disambig_links)} linked articles:")
        for i, title in enumerate(disambig_links, 1):
            print(f"  {i}. {title}")
        
        while True:
            choice = input(f"Select article (1-{len(disambig_links)}): ").strip()
            if not choice:
                print("Selection cancelled.")
                return None
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(disambig_links):
                    canonical_title = get_canonical_title(disambig_links[idx])
                    break
                else:
                    print(f"Please enter a number between 1 and {len(disambig_links)}")
            except ValueError:
                print("Please enter a valid number")
    
    print(f"Resolved to: {canonical_title}")
    return canonical_title


# ============================================================================
# Wikidata Person Verification
# ============================================================================

def get_wikidata_item_id(wikipedia_title: str) -> Optional[str]:
    """
    Get the Wikidata item ID (Q-number) for a Wikipedia article.
    
    Args:
        wikipedia_title: Wikipedia article title
        
    Returns:
        Wikidata item ID (e.g., "Q5") or None if not found
    """
    params = {
        'action': 'query',
        'titles': wikipedia_title,
        'prop': 'pageprops',
        'ppprop': 'wikibase_item'
    }
    
    data = make_api_request(WIKIPEDIA_API, params)
    
    pages = data.get('query', {}).get('pages', {})
    for page_data in pages.values():
        pageprops = page_data.get('pageprops', {})
        wikibase_item = pageprops.get('wikibase_item')
        if wikibase_item:
            return wikibase_item
    
    return None


def is_person(wikipedia_title: str) -> bool:
    """
    Check if a Wikipedia article is about a human being using Wikidata.
    
    This is the core function that enforces the people-only constraint.
    It queries Wikidata for the article's "instance of" (P31) property
    and checks if it includes Q5 (human).
    
    Args:
        wikipedia_title: Wikipedia article title
        
    Returns:
        True if the article is about a human (has P31=Q5 in Wikidata)
    """
    # Get Wikidata item ID
    item_id = get_wikidata_item_id(wikipedia_title)
    if not item_id:
        return False
    
    # Query Wikidata for P31 (instance of) claims
    params = {
        'action': 'wbgetclaims',
        'entity': item_id,
        'property': 'P31'  # instance of
    }
    
    data = make_api_request(WIKIDATA_API, params)
    
    claims = data.get('claims', {}).get('P31', [])
    
    # Check if any claim has value Q5 (human)
    for claim in claims:
        mainsnak = claim.get('mainsnak', {})
        if mainsnak.get('snaktype') == 'value':
            datavalue = mainsnak.get('datavalue', {})
            if datavalue.get('type') == 'wikibase-entityid':
                entity_id = datavalue.get('value', {}).get('id')
                if entity_id == 'Q5':
                    return True
    
    return False


def batch_check_persons(titles: List[str]) -> Set[str]:
    """
    Check multiple Wikipedia titles to see which are about people.
    Uses batch API requests for efficiency and caching.
    
    Args:
        titles: List of Wikipedia article titles
        
    Returns:
        Set of titles that are about people
    """
    if not titles:
        return set()
    
    # Check cache first
    uncached_titles = []
    people = set()
    
    for title in titles:
        if title in _person_cache:
            if _person_cache[title]:
                people.add(title)
        else:
            uncached_titles.append(title)
    
    if not uncached_titles:
        return people
    
    # Process uncached titles in batches of 50 (API limit)
    batch_size = 50
    for i in range(0, len(uncached_titles), batch_size):
        batch = uncached_titles[i:i + batch_size]
        
        # Get Wikidata item IDs for batch
        params = {
            'action': 'query',
            'titles': '|'.join(batch),
            'prop': 'pageprops',
            'ppprop': 'wikibase_item'
        }
        
        data = make_api_request(WIKIPEDIA_API, params)
        
        # Map titles to Wikidata IDs
        title_to_id = {}
        pages = data.get('query', {}).get('pages', {})
        for page_data in pages.values():
            title = page_data.get('title')
            pageprops = page_data.get('pageprops', {})
            wikibase_item = pageprops.get('wikibase_item')
            if title and wikibase_item:
                title_to_id[title] = wikibase_item
            elif title:
                # No wikibase item = not a person
                _person_cache[title] = False
        
        # Batch query Wikidata for P31 claims
        if title_to_id:
            item_ids = list(title_to_id.values())
            params = {
                'action': 'wbgetentities',
                'ids': '|'.join(item_ids),
                'props': 'claims',
                'languages': 'en'
            }
            
            data = make_api_request(WIKIDATA_API, params)
            
            entities = data.get('entities', {})
            for title, item_id in title_to_id.items():
                entity = entities.get(item_id, {})
                claims = entity.get('claims', {}).get('P31', [])
                
                is_person = False
                for claim in claims:
                    mainsnak = claim.get('mainsnak', {})
                    if mainsnak.get('snaktype') == 'value':
                        datavalue = mainsnak.get('datavalue', {})
                        if datavalue.get('type') == 'wikibase-entityid':
                            entity_id = datavalue.get('value', {}).get('id')
                            if entity_id == 'Q5':
                                is_person = True
                                people.add(title)
                                break
                
                # Cache the result
                _person_cache[title] = is_person
    
    return people


# ============================================================================
# Link Extraction
# ============================================================================

def get_person_links(title: str) -> List[str]:
    """
    Get all outbound links from a Wikipedia article, filtered to people only.
    Uses caching to avoid redundant API calls.
    
    This function enforces the people-only graph constraint by:
    1. Extracting all links from the article
    2. Batch-checking which links are about people via Wikidata
    3. Returning only the person links
    
    Args:
        title: Wikipedia article title
        
    Returns:
        List of article titles (people only) linked from this article
    """
    # Check cache first
    if title in _links_cache:
        return _links_cache[title]
    
    all_links = []
    plcontinue = None
    
    # Fetch all links (may require multiple API calls due to pagination)
    while True:
        params = {
            'action': 'query',
            'titles': title,
            'prop': 'links',
            'pllimit': '500',
            'plnamespace': '0'  # Main namespace only
        }
        
        if plcontinue:
            params['plcontinue'] = plcontinue
        
        data = make_api_request(WIKIPEDIA_API, params)
        
        pages = data.get('query', {}).get('pages', {})
        for page_data in pages.values():
            links = page_data.get('links', [])
            all_links.extend([link['title'] for link in links])
        
        # Check if there are more results
        if 'continue' in data and 'plcontinue' in data['continue']:
            plcontinue = data['continue']['plcontinue']
        else:
            break
    
    # Filter to people only using Wikidata (with caching)
    person_links = list(batch_check_persons(all_links))
    
    # Cache the result
    _links_cache[title] = person_links
    
    return person_links


def get_article_snippet(source_title: str, target_title: str) -> Optional[str]:
    """
    Extract a snippet from the source article that mentions the target.
    
    This provides "proof" that the connection exists by showing where
    the target person is mentioned in the source article.
    
    Args:
        source_title: The article we're reading from
        target_title: The person being mentioned/linked
        
    Returns:
        A text snippet showing the mention, or None if not found
    """
    try:
        import re
        
        # First, try to get plain text extract (full article, not just intro)
        params = {
            'action': 'query',
            'titles': source_title,
            'prop': 'extracts',
            'explaintext': '1',
            'exlimit': '1',
            'exintro': '0'  # Get full article
        }
        
        data = make_api_request(WIKIPEDIA_API, params)
        
        pages = data.get('query', {}).get('pages', {})
        extract_text = ''
        
        for page_data in pages.values():
            extract_text = page_data.get('extract', '')
            break
        
        if extract_text:
            # Split into sentences carefully
            sentences = re.split(r'(?<=[.!?])\s+', extract_text)
            
            # Create name variations
            name_parts = target_title.split()
            search_patterns = []
            
            # Full name (highest priority)
            search_patterns.append((target_title, 100))
            
            # Both first and last name
            if len(name_parts) >= 2:
                first_last_pattern = f"{name_parts[0]}.*{name_parts[-1]}"
                search_patterns.append((first_last_pattern, 80))
                # Last name only
                search_patterns.append((name_parts[-1], 50))
            
            best_sentence = None
            best_score = 0
            
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) < 20:  # Skip very short sentences
                    continue
                
                for pattern, score in search_patterns:
                    if re.search(re.escape(pattern) if '.*' not in pattern else pattern, 
                                sentence, re.IGNORECASE):
                        # Prefer sentences that are not too long
                        if len(sentence) < 500:
                            score += 10
                        if score > best_score:
                            best_score = score
                            best_sentence = sentence
                        break  # Found a match, move to next sentence
            
            if best_sentence:
                # Format the snippet nicely
                snippet = best_sentence.strip()
                if len(snippet) > 350:
                    # Find the name and center around it
                    name_pos = snippet.lower().find(target_title.lower())
                    if name_pos == -1 and len(name_parts) >= 2:
                        name_pos = snippet.lower().find(name_parts[-1].lower())
                    
                    if name_pos != -1:
                        start = max(0, name_pos - 150)
                        end = min(len(snippet), name_pos + 200)
                        snippet = snippet[start:end]
                        if start > 0:
                            snippet = "..." + snippet
                        if end < len(best_sentence):
                            snippet = snippet + "..."
                    else:
                        snippet = snippet[:347] + "..."
                
                return snippet
        
        # Fallback: Try to get wikitext and extract context
        params = {
            'action': 'query',
            'titles': source_title,
            'prop': 'revisions',
            'rvprop': 'content',
            'rvslots': 'main',
            'formatversion': '2',
            'rvlimit': '1'
        }
        
        data = make_api_request(WIKIPEDIA_API, params)
        pages = data.get('query', {}).get('pages', [])
        
        if pages:
            revisions = pages[0].get('revisions', [])
            if revisions:
                wikitext = revisions[0].get('slots', {}).get('main', {}).get('content', '')
                
                if wikitext:
                    # Remove wiki markup for cleaner text
                    # Remove templates {{ }}
                    wikitext = re.sub(r'\{\{[^}]*\}\}', '', wikitext)
                    # Remove references <ref>...</ref>
                    wikitext = re.sub(r'<ref[^>]*>.*?</ref>', '', wikitext, flags=re.DOTALL)
                    wikitext = re.sub(r'<ref[^>]*/', '', wikitext)
                    # Remove file/image links
                    wikitext = re.sub(r'\[\[File:.*?\]\]', '', wikitext, flags=re.IGNORECASE)
                    wikitext = re.sub(r'\[\[Image:.*?\]\]', '', wikitext, flags=re.IGNORECASE)
                    
                    # Look for the target name in wiki links [[Target Name]] or [[Target Name|Display]]
                    target_escaped = re.escape(target_title)
                    patterns = [
                        rf'\[\[{target_escaped}\]\]',
                        rf'\[\[{target_escaped}\|[^\]]+\]\]',
                    ]
                    
                    for pattern in patterns:
                        matches = list(re.finditer(pattern, wikitext, re.IGNORECASE))
                        if matches:
                            # Get context around first match
                            match = matches[0]
                            start = max(0, match.start() - 200)
                            end = min(len(wikitext), match.end() + 200)
                            context = wikitext[start:end]
                            
                            # Clean up wiki markup
                            context = re.sub(r'\[\[([^|\]]+)\]\]', r'\1', context)  # [[Link]] -> Link
                            context = re.sub(r'\[\[[^|]+\|([^\]]+)\]\]', r'\1', context)  # [[Link|Text]] -> Text
                            context = re.sub(r"'{2,}", '', context)  # Remove bold/italic marks
                            context = re.sub(r'<[^>]+>', '', context)  # Remove HTML tags
                            context = re.sub(r'\s+', ' ', context)  # Normalize whitespace
                            context = context.strip()
                            
                            # Extract a sentence
                            sentences = re.split(r'[.!?]+', context)
                            for sent in sentences:
                                sent = sent.strip()
                                if len(name_parts) >= 2:
                                    last_name = name_parts[-1]
                                    if last_name.lower() in sent.lower() and len(sent) > 30:
                                        if len(sent) > 300:
                                            sent = sent[:297] + "..."
                                        return sent
                            
                            # Return cleaned context if no good sentence found
                            if len(context) > 50:
                                if len(context) > 300:
                                    context = context[:297] + "..."
                                return context
        
        return None
        
    except Exception as e:
        return None


# ============================================================================
# Bidirectional BFS Search
# ============================================================================

def find_shortest_path(start: str, goal: str) -> Optional[List[str]]:
    """
    Find the shortest path between two people using bidirectional BFS.
    
    Algorithm:
    - Maintains two search frontiers (forward from start, backward from goal)
    - Alternates expanding the smaller frontier
    - Only explores edges between people (verified via Wikidata)
    - Stops when frontiers meet at a common person
    - Reconstructs path using parent pointers from both directions
    
    Optimizations:
    - Caches person checks and link lists to avoid redundant API calls
    - Uses sets for O(1) membership checking
    - Processes nodes in batches when possible
    - Early termination when paths are found
    
    Args:
        start: Starting person's Wikipedia title
        goal: Goal person's Wikipedia title
        
    Returns:
        List of titles forming the shortest path, or None if no path found
    """
    if start == goal:
        return [start]
    
    # Forward search state (from start)
    forward_queue = deque([start])
    forward_visited = {start}
    forward_parent = {start: None}
    forward_depth = {start: 0}
    
    # Backward search state (from goal)
    backward_queue = deque([goal])
    backward_visited = {goal}
    backward_parent = {goal: None}
    backward_depth = {goal: 0}
    
    nodes_explored = 0
    
    print("\nSearching for path (this may take a minute)...")
    
    while forward_queue or backward_queue:
        # Check limits
        if nodes_explored > MAX_NODES_EXPLORED:
            print(f"Reached exploration limit ({MAX_NODES_EXPLORED} nodes)")
            return None
        
        # Expand the smaller frontier (more efficient)
        if forward_queue and (not backward_queue or len(forward_queue) <= len(backward_queue)):
            # Forward expansion
            current = forward_queue.popleft()
            current_depth = forward_depth[current]
            
            if current_depth >= MAX_BFS_DEPTH:
                continue
            
            # Get links to other people (cached)
            try:
                neighbors = get_person_links(current)
            except:
                continue
            
            nodes_explored += 1
            
            for neighbor in neighbors:
                # Check if we've reached the goal
                if neighbor == goal:
                    # Build path: start -> ... -> current -> goal
                    path = []
                    node = current
                    while node is not None:
                        path.append(node)
                        node = forward_parent.get(node)
                    path.reverse()
                    path.append(goal)
                    return path
                
                # Check if we've reached the other frontier
                if neighbor in backward_visited:
                    # Found a meeting point! Reconstruct path
                    path = reconstruct_path(
                        start, goal, neighbor,
                        forward_parent, backward_parent
                    )
                    # Verify and return immediately (BFS guarantees shortest)
                    if path and path[0] == start and path[-1] == goal:
                        return path
                
                # Add to forward frontier
                if neighbor not in forward_visited:
                    forward_visited.add(neighbor)
                    forward_parent[neighbor] = current
                    forward_depth[neighbor] = current_depth + 1
                    forward_queue.append(neighbor)
            
            if nodes_explored % 50 == 0:
                print(f"  Explored {nodes_explored} nodes, "
                      f"forward: {len(forward_queue)}, "
                      f"backward: {len(backward_queue)}, "
                      f"cache: {len(_links_cache)} links")
        
        elif backward_queue:
            # Backward expansion
            current = backward_queue.popleft()
            current_depth = backward_depth[current]
            
            if current_depth >= MAX_BFS_DEPTH:
                continue
            
            # Get links to other people (cached)
            try:
                neighbors = get_person_links(current)
            except:
                continue
            
            nodes_explored += 1
            
            for neighbor in neighbors:
                # Check if we've reached the start
                if neighbor == start:
                    # Build path: start -> current -> ... -> goal
                    path = [start]
                    node = current
                    while node is not None:
                        path.append(node)
                        node = backward_parent.get(node)
                    return path
                
                # Check if we've reached the other frontier
                if neighbor in forward_visited:
                    # Found a meeting point! Reconstruct path
                    path = reconstruct_path(
                        start, goal, neighbor,
                        forward_parent, backward_parent
                    )
                    # Verify and return immediately (BFS guarantees shortest)
                    if path and path[0] == start and path[-1] == goal:
                        return path
                
                # Add to backward frontier
                if neighbor not in backward_visited:
                    backward_visited.add(neighbor)
                    backward_parent[neighbor] = current
                    backward_depth[neighbor] = current_depth + 1
                    backward_queue.append(neighbor)
    
    # No path found
    return None


def reconstruct_path(
    start: str,
    goal: str,
    meeting_point: str,
    forward_parent: Dict[str, Optional[str]],
    backward_parent: Dict[str, Optional[str]]
) -> List[str]:
    """
    Reconstruct the full path from start to goal through a meeting point.
    
    Args:
        start: Starting person
        goal: Goal person
        meeting_point: Person where both searches met
        forward_parent: Parent pointers from forward search
        backward_parent: Parent pointers from backward search
        
    Returns:
        Complete path as a list of titles
    """
    # Build path from start to meeting point (going backwards from meeting point)
    forward_path = []
    current = meeting_point
    while current is not None:
        forward_path.append(current)
        current = forward_parent.get(current)
    forward_path.reverse()
    
    # Build path from meeting point to goal (going backwards from goal)
    # First, build the path from goal to meeting point, then reverse it
    backward_path = []
    current = goal
    while current is not None and current != meeting_point:
        backward_path.append(current)
        current = backward_parent.get(current)
    
    # backward_path now goes from goal to meeting point, so reverse it
    backward_path.reverse()
    
    # Skip the meeting point in backward path since it's already in forward path
    if backward_path and backward_path[0] == meeting_point:
        backward_path = backward_path[1:]
    
    # Combine paths: start -> ... -> meeting_point + meeting_point -> ... -> goal
    return forward_path + backward_path


# ============================================================================
# Main Program
# ============================================================================

def main():
    """
    Main entry point for the Wikipedia Degrees of Separation tool.
    """
    print("=" * 70)
    print("Wikipedia Degrees of Separation (People-Only Edition)")
    print("=" * 70)
    print("\nThis tool finds the shortest chain of Wikipedia links between")
    print("two people's biography pages, using only people as intermediate nodes.")
    print("\nExample: Find the path from 'Albert Einstein' to 'Taylor Swift'")
    print("=" * 70)
    
    try:
        # Get user input
        print("\nEnter two people's names:")
        first_name = input("First person: ").strip()
        if not first_name:
            print("Error: Name cannot be empty")
            sys.exit(1)
        
        second_name = input("Second person: ").strip()
        if not second_name:
            print("Error: Name cannot be empty")
            sys.exit(1)
        
        start_time = time.time()
        
        # Resolve names to Wikipedia articles
        first_title = resolve_person_name(first_name)
        if not first_title:
            sys.exit(1)
        
        second_title = resolve_person_name(second_name)
        if not second_title:
            sys.exit(1)
        
        # Verify both are people using Wikidata
        print("\nVerifying articles are about people...")
        first_is_person = is_person(first_title)
        second_is_person = is_person(second_title)
        
        if not first_is_person:
            print(f"Error: '{first_title}' is not classified as a person in Wikidata")
            print("This tool only works with human biography pages.")
            sys.exit(1)
        
        if not second_is_person:
            print(f"Error: '{second_title}' is not classified as a person in Wikidata")
            print("This tool only works with human biography pages.")
            sys.exit(1)
        
        print(f"✓ Both articles verified as people")
        
        # Find shortest path
        search_start = time.time()
        path = find_shortest_path(first_title, second_title)
        search_time = time.time() - search_start
        
        # Display results
        print("\n" + "=" * 70)
        if path:
            print("PATH FOUND!")
            print("=" * 70)
            print(f"\nShortest path: {' → '.join(path)}")
            print(f"Path length: {len(path) - 1} step{'s' if len(path) - 1 != 1 else ''}")
            
            # Show detailed connections
            print("\n" + "=" * 70)
            print("CONNECTION DETAILS:")
            print("=" * 70)
            
            for i in range(len(path) - 1):
                source = path[i]
                target = path[i + 1]
                
                print(f"\n[Step {i + 1}] {source} → {target}")
                print(f"  📄 {source}: https://en.wikipedia.org/wiki/{urllib.parse.quote(source.replace(' ', '_'))}")
                print(f"     (This article links to {target})")
            
            # Add link to final article
            print(f"\n[Destination] {path[-1]}")
            print(f"  📄 {path[-1]}: https://en.wikipedia.org/wiki/{urllib.parse.quote(path[-1].replace(' ', '_'))}")
            
        else:
            print("NO PATH FOUND")
            print("=" * 70)
            print(f"\nCould not find a people-only path between:")
            print(f"  • {first_title}")
            print(f"  • {second_title}")
            print(f"\nThis could mean:")
            print(f"  • The path is longer than {MAX_BFS_DEPTH} steps")
            print(f"  • They are in disconnected components of the people graph")
            print(f"  • Search limits were reached")
        
        # Timing information
        total_time = time.time() - start_time
        print("\n" + "=" * 70)
        print(f"Search time: {search_time:.2f} seconds")
        print(f"Total time:  {total_time:.2f} seconds")
        print("=" * 70)
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Exiting...")
        sys.exit(130)
    except Exception as e:
        print(f"\nUnexpected error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()