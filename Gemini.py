import requests
import time
from collections import deque
import sys

class WikiPathFinder:
    """
    A class to find the shortest path between two Wikipedia articles using 
    Bidirectional Breadth-First Search (BFS).
    """

    def __init__(self, lang="en"):
        self.api_url = f"https://{lang}.wikipedia.org/w/api.php"
        self.session = requests.Session()
        # Wikipedia requires a User-Agent to identify the bot/script
        self.session.headers.update({
            "User-Agent": "WikiShortestPathBot/1.0 (Educational Script)"
        })

    def get_canonical_title(self, title):
        """
        Verifies if a page exists and returns its canonical title 
        (resolving redirects and capitalization).
        """
        params = {
            "action": "query",
            "format": "json",
            "titles": title,
            "redirects": 1
        }
        
        try:
            response = self.session.get(self.api_url, params=params).json()
            pages = response["query"]["pages"]
            
            # The 'pages' dictionary keys are page IDs. "-1" implies missing.
            page_id = next(iter(pages))
            if page_id == "-1":
                return None
                
            return pages[page_id]["title"]
        except Exception as e:
            print(f"Error checking title '{title}': {e}")
            return None

    def get_links(self, title):
        """
        Fetches all internal Wikipedia links (namespace 0) for a given page title.
        Handles pagination (continue tokens) to ensure all links are retrieved.
        """
        links = set()
        params = {
            "action": "query",
            "format": "json",
            "titles": title,
            "prop": "links",
            "pllimit": "max",  # Request max allowed links (500 for standard users)
            "plnamespace": 0   # Only filter for Main Article namespace (ignore Talk, User, etc.)
        }

        while True:
            try:
                response = self.session.get(self.api_url, params=params).json()
                
                pages = response.get("query", {}).get("pages", {})
                page = next(iter(pages.values()))

                if "links" in page:
                    for link in page["links"]:
                        links.add(link["title"])

                # Check if there are more links to fetch (pagination)
                if "continue" in response:
                    params["plcontinue"] = response["continue"]["plcontinue"]
                else:
                    break
                    
            except requests.exceptions.RequestException as e:
                print(f"Network error fetching links for {title}: {e}")
                break
            except Exception as e:
                # Occasional API hiccups or malformed responses
                break
        
        return links

    def reconstruct_path(self, parents_start, parents_end, meeting_point):
        """
        Reconstructs the path when the two BFS searches collide.
        
        Args:
            parents_start: Dict tracking path from source -> middle
            parents_end: Dict tracking path from target -> middle
            meeting_point: The article where the two searches met
        """
        # Backtrack from meeting point to start
        path_start = []
        curr = meeting_point
        while curr:
            path_start.append(curr)
            curr = parents_start.get(curr)
        path_start.reverse()

        # Backtrack from meeting point to end
        path_end = []
        curr = parents_end.get(meeting_point) # Start from parent of meeting point
        while curr:
            path_end.append(curr)
            curr = parents_end.get(curr)
        
        return path_start + path_end

    def find_path(self, start_title, end_title):
        """
        Executes Bidirectional BFS to find the shortest path.
        """
        start_node = self.get_canonical_title(start_title)
        end_node = self.get_canonical_title(end_title)

        if not start_node:
            return None, f"Page '{start_title}' not found."
        if not end_node:
            return None, f"Page '{end_title}' not found."
        
        if start_node == end_node:
            return [start_node], "Start and End are the same."

        print(f"Searching for path between '{start_node}' and '{end_node}'...")
        print("This may take a moment depending on the 'distance'...")

        # Queues for BFS: (current_page)
        queue_start = deque([start_node])
        queue_end = deque([end_node])

        # Parent dictionaries to reconstruct path: child -> parent
        # Also serves as "visited" set
        parents_start = {start_node: None}
        parents_end = {end_node: None}

        while queue_start and queue_end:
            # OPTIMIZATION: Always expand the smaller frontier to reduce search space
            if len(queue_start) <= len(queue_end):
                active_queue = queue_start
                active_parents = parents_start
                other_parents = parents_end
                direction = "forward"
            else:
                active_queue = queue_end
                active_parents = parents_end
                other_parents = parents_start
                direction = "backward"

            current_page = active_queue.popleft()
            
            # Fetch links for the current page
            links = self.get_links(current_page)

            for link in links:
                if link in active_parents:
                    continue # Already visited in this direction

                # Collision check: Have we seen this node in the other direction?
                if link in other_parents:
                    print(f"Connection found at: {link}")
                    if direction == "forward":
                        return self.reconstruct_path(active_parents, other_parents, link), None
                    else:
                        # If we were searching backward, swap args to reconstruct correctly
                        return self.reconstruct_path(other_parents, active_parents, link), None

                # Add to queue and mark parent
                active_parents[link] = current_page
                active_queue.append(link)

        return None, "No path found (graph might be disconnected)."

def main():
    print("--- Wikipedia Shortest Path Finder ---")
    print("Required: 'requests' library (pip install requests)")
    print("----------------------------------------")

    finder = WikiPathFinder()

    try:
        start_input = input("Enter start page name (e.g., Kevin Bacon): ").strip()
        end_input = input("Enter end page name (e.g., Barack Obama): ").strip()

        if not start_input or not end_input:
            print("Error: Please provide valid names.")
            return

        start_time = time.time()
        path, error = finder.find_path(start_input, end_input)
        end_time = time.time()

        if error:
            print(f"\nResult: {error}")
        else:
            print(f"\n--- Path Found ({len(path) - 1} degrees of separation) ---")
            for i, page in enumerate(path):
                if i < len(path) - 1:
                    print(f"{i+1}. {page} ->")
                else:
                    print(f"{i+1}. {page}")
            
            print(f"\nSearch took {end_time - start_time:.2f} seconds.")

    except KeyboardInterrupt:
        print("\nSearch cancelled by user.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")

if __name__ == "__main__":
    main()