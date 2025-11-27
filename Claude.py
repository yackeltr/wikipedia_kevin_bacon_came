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
    Uses batch API requests for efficiency.
    
    Args:
        titles: List of Wikipedia article titles
        
    Returns:
        Set of titles that are about people
    """
    if not titles:
        return set()
    
    people = set()
    
    # Process in batches of 50 (API limit)
    batch_size = 50
    for i in range(0, len(titles), batch_size):
        batch = titles[i:i + batch_size]
        
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
                
                for claim in claims:
                    mainsnak = claim.get('mainsnak', {})
                    if mainsnak.get('snaktype') == 'value':
                        datavalue = mainsnak.get('datavalue', {})
                        if datavalue.get('type') == 'wikibase-entityid':
                            entity_id = datavalue.get('value', {}).get('id')
                            if entity_id == 'Q5':
                                people.add(title)
                                break
    
    return people


# ============================================================================
# Link Extraction
# ============================================================================

def get_person_links(title: str) -> List[str]:
    """
    Get all outbound links from a Wikipedia article, filtered to people only.
    
    This function enforces the people-only graph constraint by:
    1. Extracting all links from the article
    2. Batch-checking which links are about people via Wikidata
    3. Returning only the person links
    
    Args:
        title: Wikipedia article title
        
    Returns:
        List of article titles (people only) linked from this article
    """
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
    
    # Filter to people only using Wikidata
    person_links = batch_check_persons(all_links)
    
    return list(person_links)


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
    
    while forward_queue and backward_queue:
        # Check limits
        nodes_explored += 1
        if nodes_explored > MAX_NODES_EXPLORED:
            print(f"Reached exploration limit ({MAX_NODES_EXPLORED} nodes)")
            return None
        
        # Expand the smaller frontier (more efficient)
        if len(forward_queue) <= len(backward_queue):
            # Forward expansion
            current = forward_queue.popleft()
            current_depth = forward_depth[current]
            
            if current_depth >= MAX_BFS_DEPTH:
                continue
            
            # Get links to other people
            try:
                neighbors = get_person_links(current)
            except:
                continue
            
            for neighbor in neighbors:
                # Check if we've reached the other frontier
                if neighbor in backward_visited:
                    # Found a meeting point! Reconstruct path
                    return reconstruct_path(
                        start, goal, neighbor,
                        forward_parent, backward_parent
                    )
                
                # Add to forward frontier
                if neighbor not in forward_visited:
                    forward_visited.add(neighbor)
                    forward_parent[neighbor] = current
                    forward_depth[neighbor] = current_depth + 1
                    forward_queue.append(neighbor)
            
            if nodes_explored % 100 == 0:
                print(f"  Explored {nodes_explored} nodes, "
                      f"forward frontier: {len(forward_queue)}, "
                      f"backward frontier: {len(backward_queue)}")
        
        else:
            # Backward expansion
            current = backward_queue.popleft()
            current_depth = backward_depth[current]
            
            if current_depth >= MAX_BFS_DEPTH:
                continue
            
            # Get links to other people
            try:
                neighbors = get_person_links(current)
            except:
                continue
            
            for neighbor in neighbors:
                # Check if we've reached the other frontier
                if neighbor in forward_visited:
                    # Found a meeting point! Reconstruct path
                    return reconstruct_path(
                        start, goal, neighbor,
                        forward_parent, backward_parent
                    )
                
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
    # Build path from start to meeting point
    forward_path = []
    current = meeting_point
    while current is not None:
        forward_path.append(current)
        current = forward_parent[current]
    forward_path.reverse()
    
    # Build path from meeting point to goal
    backward_path = []
    current = backward_parent[meeting_point]  # Skip meeting point (already in forward)
    while current is not None:
        backward_path.append(current)
        current = backward_parent[current]
    
    # Combine paths
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
            print(f"\n{' → '.join(path)}")
            print(f"\nPath length: {len(path) - 1} steps")
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