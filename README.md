# **Homework Assignment: Wikipedia Degrees-of-Separation Tool (People-Only Edition)**

Your task is to create a complete, runnable Python 3 command-line program that meets the following functional and behavioral requirements.

---

## **Program Objective**

Build a command-line tool that takes the names of **two individuals who each have a Wikipedia page** and determines the **shortest chain of hyperlinks** that connects their articles on English Wikipedia.
All nodes in the chain must be **people only**.

---

## **User Interaction Requirements**

1. When run, the program must:

   * Print a short description of what it does.
   * Prompt the user to enter:

     * A first person’s name.
     * A second person’s name.

2. The program must resolve each entered name to a **specific** English Wikipedia article:

   * Search Wikipedia for matching pages.
   * Prefer an exact case-insensitive title match if one exists.
   * If multiple matches are found, display them and require the user to choose one by number.
   * If the selected page is a **disambiguation page**, display the articles it lists and require another user selection.
   * Final stored titles must be the canonical resolved titles (i.e., after redirects).

---

## **People-Only Requirements**

1. Both chosen Wikipedia pages must represent **individual human beings**.
2. Every page within the final hyperlink chain must also represent an individual human being.
3. Human/non-human classification must be determined through **Wikidata**, checking the page’s “instance of” classification.

If either selected page is not a person, the program must print an informative message and exit.

---

## **Graph Requirements**

Your program must conceptually treat Wikipedia as a graph where:

* **Nodes** = English Wikipedia pages representing **people**.
* **Edges** = Hyperlinks between Wikipedia articles (internal main-namespace links).

The program must search this graph to find the **shortest** hyperlink chain between the two individuals.

The graph is not pre-computed; your program must query Wikipedia as needed.

---

## **Search Behavior Requirements**

1. Your program must compute the **shortest people-only hyperlink path**.
2. You must support:

   * Outgoing links from a page.
   * Incoming links to a page (backlinks).
3. You must impose:

   * A maximum number of nodes expanded before aborting.
   * A wall-clock time limit before aborting.
4. If limits are reached before finding a path, the program must:

   * Abort the search.
   * Print a clear message explaining that no solution was found within the constraints.

---

## **Output Requirements**

If a valid people-only chain exists:

* Print the titles in order, formatted as a readable chain, for example:

  ```
  Person A -> Person B -> Person C -> Person D
  ```
* Print the length of the chain (number of edges).
* Print:

  * Search duration.
  * Total program runtime (including name resolution and human-checking).

If no valid chain exists:

* Print a clear explanation that no people-only hyperlink path was found within the search constraints.

---

## **Error-Handling Requirements**

The program must gracefully handle all of the following:

* No Wikipedia results for an input name.
* User cancels a selection (e.g., presses Enter with no choice).
* Wikipedia page exists but is not about a human.
* Wikipedia/Wikidata API errors or repeated HTTP failures.
* Search timeout or expansion-limit exceeded.
* Keyboard interrupt (Ctrl+C).

In all cases, the program must exit cleanly with an informative message.

---

## **Non-Functional Requirements**

1. The tool must run as a **standalone Python script**.
2. It must use:

   * Standard Python 3.
   * A single external HTTP library (if needed).
3. Code must include:

   * A short header comment explaining the program’s purpose.
   * Clear, well-structured function organization.
   * High-level explanatory comments written for human readers.


