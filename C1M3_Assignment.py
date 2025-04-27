#!/usr/bin/env python
# coding: utf-8

# # Assignment - Implementing graph algorithms with LLMs
# 
# 
# Welcome to the first assignment of Course 1 in the Generative AI for Software Development Specialization! In this assignment you will **work alongside an LLM to take on four different graph-based problems** including finding the shortest path through a graph and solving different variations of the classic Travelling Salesman Problem in which you must find the shortest tour through every vertex in the graph. This is a great opportunity to practice using the LLM skills you've been learning! You can use the LLM of your choosing but [GPT-4o is available here](https://www.coursera.org/learn/introduction-to-generative-ai-for-software-development/ungradedLab/Vuqvf/gpt-3-5-environment), in the ungraded lab that accompanies this assignment.
# 
# **Here's a quick summary of the contents of this notebook:**
# 
# - Section 1: Introduction, setup, and the `Graph` class this assignment is based on
# - Section 2: Recommendations on how to complete the assignment
# - Section 3: A playground to test solutions
# - Section 4: The exercises
# 

# 
# # Table of Contents
# - [ 1 - Introduction](#1)
#   - [ 1.1 - Importing unittests](#1-1)
#   - [ 1.2 - The `Graph` Class](#1-2)
# - [ 2 - Recommentations](#2)
#   - [ 2.1 - Recommended Development Steps](#2-1)
# - [ 3 - Playground](#3)
#   - [ 3.1 - The `generate_graph` function](#3-1)
#   - [ 3.2 Example usage (using the base Graph class)](#3-2)
# - [ 4 - Exercises](#4)
#   - [ Exercise 1](#ex01)
#     - [ Test Exercise 1 (`shortest_path`)](#4-1)
#   - [ Exercise 2](#ex02)
#     - [ Test Exercise 2 (`tsp_small_graph`)](#4-2)
#   - [ Exercise 3](#ex03)
#     - [ Test Exercise 3 (`tsp_large_graph`)](#4-3)
#   - [ Exercise 4](#ex04)
#     - [Test Exercise 4 (`tsp_medium_graph`)](#4-4)
# 

# <a id='1'></a>
# 
# ## 1 - Introduction  
# 
# This assignment revolves around a pre-implemented `Graph` class. You will implement four additional classes, each focused on specific graph-related problems:  
# 
# - **GraphShortestPath** – Implements the `shortest_path` method to find the shortest route between two nodes.  
# - **GraphTSPSmallGraph** – Implements the `tsp_small_graph` method to solve the Travelling Salesman Problem (TSP) efficiently for small graphs.  
# - **GraphTSPMediumGraph** – Implements the `tsp_medium_graph` method to solve TSP for medium-sized graphs.  
# - **GraphTSPLargeGraph** – Implements the `tsp_large_graph` method to tackle TSP for large graphs.  
# 
# Each task comes with constraints and execution time limits, detailed in their respective sections. Work alongside an LLM to develop these algorithms, but avoid copying solutions outright. Instead, ask for explanations, break down the logic, and refine your approach. This will help with debugging and optimizing the code to meet the assignment's requirements.
# 
# **Here's a few important points on how this notebook works:**
# 
# - **Only the graded cells are Graded:** You can add new cells to experiment but these will be ignored by the grader. If you want something graded, include it in the provided cell that contains the respective Graph class.
# - **Some cells are frozen:** Some cells are frozen, e.g. the `Graph` class, to avoid mistakenly editing their code.
# - **Avoid importing new libraries:** Avoid importing new libraries beyond those you'll find in section 1.1, immediately below. **In particular `joblib` or any multiprocessing or parallel computing approach will crash the grader.
# - **Save before submitting:** Do this to ensure you are graded on your most recent work.
# 
# **If you experience any issues with the assignment or have any difficulties, please reach out to our [Discourse Community](https://www.coursera.org/learn/introduction-to-generative-ai-for-software-development/item/hIZen). Mentors, fellow learners, and staff will be available to assist you with any challenges you may encounter!**
# 
# Happy coding!

# <a id='1-1'></a>
# ### 1.1 - Importing unittests
# 
# This library includes unit tests to evaluate your solutions. After every exercise, there will be a cell with a unittest. Run it to get your solution tested.
# 
# There is also a submission checker, which you should run at the end of the assignment, to ensure that your solution will be properly evaluated by the autograder.

# In[1]:


import unittests
import submission_checker


# <a id='1-2'></a>
# ### 1.2 - The `Graph` Class
# 
# The `Graph` class that defines the structure of a graph and the methods that can be performed on them. Take a moment to familiarize yourself with the class's structure and its methods, or better yet, share the code with an LLM and ask it for an explanation!
# 
# The cell below is frozen and you cannot edit the class. Instead, you will work inside of the 4 graph classes, one for each exercise.

# In[2]:


import random
import heapq
import itertools

class Graph:
    def __init__(self, directed=False):
        """
        Initialize the Graph.

        Parameters:
        - directed (bool): Specifies whether the graph is directed. Default is False (undirected).

        Attributes:
        - graph (dict): A dictionary to store vertices and their adjacent vertices (with weights).
        - directed (bool): Indicates whether the graph is directed.
        """
        self.graph = {}
        self.directed = directed
    
    def add_vertex(self, vertex):
        """
        Add a vertex to the graph.

        Parameters:
        - vertex: The vertex to add. It must be hashable.

        Ensures that each vertex is represented in the graph dictionary as a key with an empty dictionary as its value.
        """
        if not isinstance(vertex, (int, str, tuple)):
            raise ValueError("Vertex must be a hashable type.")
        if vertex not in self.graph:
            self.graph[vertex] = {}
    
    def add_edge(self, src, dest, weight):
        """
        Add a weighted edge from src to dest. If the graph is undirected, also add from dest to src.

        Parameters:
        - src: The source vertex.
        - dest: The destination vertex.
        - weight: The weight of the edge.
        
        Prevents adding duplicate edges and ensures both vertices exist.
        """
        if src not in self.graph or dest not in self.graph:
            raise KeyError("Both vertices must exist in the graph.")
        if dest not in self.graph[src]:  # Check to prevent duplicate edges
            self.graph[src][dest] = weight
        if not self.directed and src not in self.graph[dest]:
            self.graph[dest][src] = weight
    
    def remove_edge(self, src, dest):
        """
        Remove an edge from src to dest. If the graph is undirected, also remove from dest to src.

        Parameters:
        - src: The source vertex.
        - dest: The destination vertex.
        """
        if src in self.graph and dest in self.graph[src]:
            del self.graph[src][dest]
        if not self.directed:
            if dest in self.graph and src in self.graph[dest]:
                del self.graph[dest][src]
    
    def remove_vertex(self, vertex):
        """
        Remove a vertex and all edges connected to it.

        Parameters:
        - vertex: The vertex to be removed.
        """
        if vertex in self.graph:
            # Remove any edges from other vertices to this one
            for adj in list(self.graph):
                if vertex in self.graph[adj]:
                    del self.graph[adj][vertex]
            # Remove the vertex entry itself
            del self.graph[vertex]
    
    def get_adjacent_vertices(self, vertex):
        """
        Get a list of vertices adjacent to the specified vertex.

        Parameters:
        - vertex: The vertex whose neighbors are to be retrieved.

        Returns:
        - List of adjacent vertices. Returns an empty list if vertex is not found.
        """
        return list(self.graph.get(vertex, {}).keys())  
    
    def tour_length(self, tour):
        """
        Calculate the length of a tour.  Handles cases where edges might be missing.

        Parameters:
        - tour: A list of vertices representing a tour. A tour ends and starts in the initial vertex. This is assumed, so you should not write the last vertice.

        Returns:
        - The total length of the tour. Returns infinity if any edge in the tour is missing.
        """
        if tour and tour[0] == tour[-1] and len(tour) > 1:
            raise ValueError("Tour should not include the return to the starting vertex.")
        total_length = 0
        for i in range(len(tour)):
            weight = self._get_edge_weight(tour[i], tour[(i + 1) % len(tour)])
            if weight == float('inf'):  # Check for missing edge
                return float('inf')  # Tour is invalid if any edge is missing
            total_length += weight
        return total_length

    def _get_edge_weight(self, src, dest):
        """
        Get the weight of the edge from src to dest.

        Parameters:
        - src: The source vertex.
        - dest: The destination vertex.

        Returns:
        - The weight of the edge. If the edge does not exist, returns infinity.
        """
        return self.graph[src].get(dest, float('inf'))
    
    def __str__(self):
        """
        Provide a string representation of the graph's adjacency list for easy printing and debugging.

        Returns:
        - A string representation of the graph dictionary.
        """
        return str(self.graph)


# <a id='2'></a>
# ## 2 - Recommentations
# 
# <a id='2-1'></a>
# ### 2.1 - Recommended Development Steps
# 
# The goal of this project is to develop your solutions alongside an LLM. Here's some steps to try:
# 
# - **Understand the Problem**: Discuss graph theory problems like the Travelling Salesman Problem (TSP) and shortest path finding with the LLM. Use it to clarify concepts and get examples, enhancing your foundational knowledge.
# 
# - **Analyze the `Graph` Class**: Analyze the provided Graph class functions and structure by consulting the LLM. Its explanations will help you understand how to effectively utilize and extend the class.
# 
# - **Brainstorm Solutions**: For each exercise, there is a class and a method (`shortest_path`, `tsp_small_graph`, `tsp_large_graph`, `tsp_medium_graph`), brainstorm with the LLM potential solutions. Share the time constraints and whether the solutions need to be optimal and see which algorithms the LLM recommends.
# 
# - **Implement Solutions**: Have the LLM generate methods that implement the solutions you brainstorm. Ensure the LLM understands the structure of the `Graph` class and the function you want to implement so that the generated code will run as expected.
# 
# 5. **Debug Errors**: Use the LLM to strategize and review your testing approach, especially for tests that yield unexpected results. It can offer debugging and optimization advice to improve your solutions.
# 
# Of course the LLM Best Practices of "Be specific", "Assign a role", "Request an expert opinion", "Give feedback" should serve you well throughout!

# <a id='3'></a>
# ## 3 - Playground
# 
# Use the space below to experiment with and test your methods. The `generate_graph` function can be used to generate graphs with different properties.
# 
# To measure the execution time of your code, you can use the `%%timeit` magic method, an example of which appears at the bottom of this section. **Remember, `%%timeit` should be placed at the beginning of a code cell**, even before any comments marked by `#`.
# 
# You're encouraged to create as many new cells as needed for testing, but keep in mind that only the code within the graded cells will be considered during grading.

# <a id='3-1'></a>
# ### 3.1 - The `generate_graph` function
# 
# The function below will generate graphs and you may find it useful in experimenting with solutions or testing your code. This function is also the one used to generate the graphs used in the unit tests. If your algorithm fails a test case, you will be given the call to this function with the appropriate arguments needed to replicate the graph that caused your algorithm to fail. Additionally, the reason for the failure will be provided, whether it was due to exceeding the time limit or not achieving an optimal or near-optimal distance.

# In[3]:


def generate_graph(graph, nodes, edges=None, complete=False, weight_bounds=(1,600), seed=None):
    """
    Generates a graph with specified parameters, allowing for both complete and incomplete graphs.
    
    This function creates a graph with a specified number of nodes and edges, with options for creating a complete graph, and for specifying the weight bounds of the edges. It uses the Graphs class to create and manipulate the graph.

    Parameters:
    - graph (one of the graph calsses for each exercise): The graph class to generate the test graph.
    - nodes (int): The number of nodes in the graph. Must be a positive integer.
    - edges (int, optional): The number of edges to add for each node in the graph. This parameter is ignored if `complete` is set to True. Defaults to None.
    - complete (bool, optional): If set to True, generates a complete graph where every pair of distinct vertices is connected by a unique edge. Defaults to False.
    - weight_bounds (tuple, optional): A tuple specifying the lower and upper bounds (inclusive) for the random weights of the edges. Defaults to (1, 600).
    - seed (int, optional): A seed for the random number generator to ensure reproducibility. Defaults to None.

    Raises:
    - ValueError: If `edges` is not None and `complete` is set to True, since a complete graph does not require specifying the number of edges.

    Returns:
    - The graph class: An instance of the Graph class you passed in the graph parameter, representing the generated graph, with vertices labeled as integers starting from 0.

    Examples:
    - Generating a complete graph with 5 nodes:
        generate_graph(5, complete=True)
    
    - Generating an incomplete graph with 5 nodes and 2 edges per node:
        generate_graph(5, edges=2)
    
    Note:
    - The function assumes the existence of a Graph class with methods for adding vertices (`add_vertex`) and edges (`add_edge`), as well as a method for getting adjacent vertices (`get_adjacent_vertices`).
    """
    random.seed(seed)
    graph = graph()
    if edges is not None and complete:
        raise ValueError("edges must be None if complete is set to True")
    if not complete and edges > nodes:
        raise ValueError("number of edges must be less than number of nodes")
    

    for i in range(nodes):
        graph.add_vertex(i)
    if complete:
        for i in range(nodes):
            for j in range(i+1,nodes):
                weight = random.randint(weight_bounds[0], weight_bounds[1])
                graph.add_edge(i,j,weight)
    else:
        for i in range(nodes):
            for _ in range(edges):
                j = random.randint(0, nodes - 1)
                while (j == i or j in graph.get_adjacent_vertices(i)) and len(graph.get_adjacent_vertices(i)) < nodes - 1:  # Ensure the edge is not a loop or a duplicate
                    j = random.randint(0, nodes - 1)
                weight = random.randint(weight_bounds[0], weight_bounds[1])
                if len(graph.graph[i]) < edges and len(graph.graph[j]) < edges:
                    graph.add_edge(i, j, weight)
    return graph


# <a id='3-2'></a>
# ### 3.2 Example usage (using the base Graph class)
# 
# Let's experiment this function using the Graph class

# In[4]:


graph = generate_graph(graph = Graph, nodes = 10, complete = True)


# In[5]:


# Using its methods:

print(graph.tour_length([1, 4, 6, 8, 5]))


# In[6]:


print(graph._get_edge_weight(5, 8))


# In[7]:


## Example function and use of the %%timeit magic method
def foo(n):
    i = 0
    for i in range(10000):
        for j in range(n):
            i += j
    return i


# In[8]:


print(foo(10))


# In[9]:


get_ipython().run_cell_magic('timeit', '', 'foo(10)\n')


# Feel free to add as many cells as you want! You can do so by clicking in the `+` button in the left upper corner of this notebook.

# <a id='4'></a>
# ## 4 - Exercises
# 
# Let's dive into the exercises! There are a total of four exercises, and you must pass at least three of them. To successfully complete the exercises, it is **essential** that you do not blindly copy an LLM solution. Instead, ask it to explain the concepts to you and then work together to complete the exercise. The explanation for each exercise is provided above. Feel free to use the `generate_graph` function to debug your solution if it fails to pass any tests.

# <a id='ex01'></a>
# ### Exercise 1: `shortest_path` (Sparse Graphs, 10,000 nodes, Optimal Solution, 0.5 seconds)
# 
# In this challenge you will be given a large graph with tens of thousands of nodes. The graph is "sparse", however, meaning that each vertex may only have edges leading to a few other vertices. You will need to develop an algorithm to find the shortest path between two vertices in this graph. The solution will be the length of the path, and the list of vertices along that path. This is a classic computer science problem, and there are a few standard algorithms your LLM is likely to suggest. To ensure your code is efficient, it must execute in less than `0.5` seconds. 
# 
# Read **carefully** the function's docstring (the text describing the function), as the parameters and the return must **NOT** be changed.
# 
# **Important Note:** Do **NOT** include any new libraries beyond those imported at the beginning of the notebook. You may need to explicitly state this and ask the LLM **NOT** to use any additional libraries if necessary. Using additional libraries may cause you to fail the autograder, even if you pass the local tests.
# 
# **What you CAN do:** Create new functions **within the `GraphShortestPath` class**. Do **NOT** create functions outside the `GraphShortestPath` class or outside this graded cell. You may do so during development, but in the end, paste your solution into the graded cell following these instructions.

# In[10]:


class GraphShortestPath(Graph):

    def shortest_path(self, start, end):
        """
        Calculate the shortest path from a starting node to an ending node in a sparse graph
        with potentially 10000s of nodes. Must run under 0.5 second and find the shortest distance between two nodes.

        Parameters:
        start: The starting node.
        end: The ending node.

        Returns:
        A tuple containing the total distance of the shortest path and a list of nodes representing that path.
        """
        import heapq

        # Priority queue to hold the nodes to explore
        queue = []
        heapq.heappush(queue, (0, start))  # (distance, node)

        # Dictionary to store the shortest distance to each node
        distances = {start: 0}

        # Dictionary to store the path taken to reach each node
        previous_nodes = {start: None}

        while queue:
            current_distance, current_node = heapq.heappop(queue)

            # If we reach the end node, we can reconstruct the path
            if current_node == end:
                path = []
                while current_node is not None:
                    path.append(current_node)
                    current_node = previous_nodes[current_node]
                path.reverse()
                return current_distance, path

            # Explore neighbors
            for neighbor, weight in self.get_neighbors(current_node):
                distance = current_distance + weight

                # If a shorter path to the neighbor is found
                if neighbor not in distances or distance < distances[neighbor]:
                    distances[neighbor] = distance
                    previous_nodes[neighbor] = current_node
                    heapq.heappush(queue, (distance, neighbor))

        # If the end node is not reachable from the start node
        return float('inf'), []

    def get_neighbors(self, node):
        """
        Get the neighbors of the given node as a list of tuples (neighbor, weight).
        Assumes this is implemented in the base Graph class.
        """
        if node not in self.graph:
            return []
        return [(neighbor, weight) for neighbor, weight in self.graph[node].items()]


# <a id='4-1'></a>
# ### 4.1 - Test Exercise 1 (`shortest_path`)
# 
# Run the code below to test the `shortest_path` method on sparsely connected graphs with 10,000 nodes. The requirements for passing this exercise are:
# 
# - The algorithm must complete its run in under `0.5` second for each graph.
# - It must accurately find the shortest path.

# In[11]:


unittests.test_shortest_path(GraphShortestPath)


# <a id='ex02'></a>
# ### Exercise 2: `tsp_small_graph` (Complete Graphs, 10 nodes, Optimal Solution, 1 second)
# 
# The Traveling Salesman Problem asks you to find the shortest path through a graph that visits all vertices and returns to the start, also called a "tour". This problem famously is computationally intensive for large graphs, making it essentially impossible to find the absolute best solution. For smaller graphs, however, a brute force approach is possible, and that's the first version of the Traveling Salesman Problem you'll tackle.
# 
# Write a method that, given a small graph of about 10 nodes, finds the shortest tour of the graph that starts and ends at node 0. Unlike in the first exercise, the graphs here are "complete", meaning there is an edge from each node to each other node. The method should return the length of the tour and a list of the nodes visited. **The tour found must be the absolute shortest through the graph.** Your solution must also be efficient, completing the task in under `1` second.
# 
# Read **carefully** the function's docstring (the text describing the function), as the parameters and the return must **NOT** be changed.
# 
# **Important Note:** Do **NOT** include any new libraries beyond those imported at the beginning of the notebook. You may need to explicitly state this and ask the LLM **NOT** to use any additional libraries if necessary. Using additional libraries may cause you to fail the autograder, even if you pass the local tests.
# 
# **What you CAN do:** Create new functions **within the `GraphTSPPath` class**. Do **NOT** create functions outside the `GraphTSPPath` class or outside this graded cell. You may do so during development, but in the end, paste your solution into the graded cell following these instructions.

# In[12]:


# GRADED CELL 2 - Do NOT delete it, do NOT place your solution anywhere else. You can create new cells and work from there, but in the end add your solution in this cell.

from itertools import permutations



class Graph:
    def __init__(self, directed=False):
        """Initialize the Graph."""
        self.graph = {}
        self.directed = directed

    def add_vertex(self, vertex):
        """Add a vertex to the graph."""
        if vertex not in self.graph:
            self.graph[vertex] = {}

    def add_edge(self, src, dest, weight):
        """Add a weighted edge from src to dest. If undirected, add both directions."""
        if src not in self.graph or dest not in self.graph:
            raise KeyError("Both vertices must exist in the graph.")
        self.graph[src][dest] = weight
        if not self.directed:
            self.graph[dest][src] = weight

    def tour_length(self, tour):
        """
        Calculate the length of a tour, including returning to the starting vertex.
        Handles cases where edges might be missing.
        """
        total_length = 0
        for i in range(len(tour) - 1):  # No need to check the last-to-first in this loop
            weight = self._get_edge_weight(tour[i], tour[i + 1])
            if weight == float('inf'):  # If an edge is missing
                return float('inf')  # Tour is invalid if any edge is missing
            total_length += weight
        return total_length

    def _get_edge_weight(self, src, dest):
        """Get the weight of the edge from src to dest. If edge does not exist, return infinity."""
        return self.graph[src].get(dest, float('inf'))

class GraphTSPSmallGraph(Graph):
    def tsp_small_graph(self, start_vertex):
        """
        Solve the Travelling Salesman Problem for a small (~10 node) complete graph starting from a specified node.
        Required to find the optimal tour. Expect graphs with at most 10 nodes. Must run under 1 second.
        
        Parameters:
        start_vertex: The starting node.
        
        Returns:
        A tuple containing the total distance of the tour and a list of nodes representing the tour path.
        """
        # Ensure the graph has at most 10 nodes
        if len(self.graph) > 10:
            raise ValueError("Graph exceeds the limit of 10 nodes for this method.")
        
        # Get all nodes
        nodes = list(self.graph.keys())
        n = len(nodes)
        start_index = nodes.index(start_vertex)
        
        # Initialize DP table
        # dp[mask][i] represents the minimum cost to visit all nodes in "mask" ending at node "i"
        dp = [[float('inf')] * n for _ in range(1 << n)]
        dp[1 << start_index][start_index] = 0
        
        # Precompute edge weights for easier access
        edge_weights = [[self._get_edge_weight(nodes[i], nodes[j]) for j in range(n)] for i in range(n)]
        
        # Fill DP table
        for mask in range(1 << n):
            for i in range(n):
                if not (mask & (1 << i)):  # If node i is not in the current mask
                    continue
                for j in range(n):
                    if mask & (1 << j):  # If node j is already in the mask
                        continue
                    next_mask = mask | (1 << j)
                    dp[next_mask][j] = min(dp[next_mask][j], dp[mask][i] + edge_weights[i][j])
        
        # Find the minimum cost to complete the tour
        min_cost = float('inf')
        last_mask = (1 << n) - 1  # All nodes visited
        last_index = start_index
        for i in range(n):
            if i == start_index:
                continue
            min_cost = min(min_cost, dp[last_mask][i] + edge_weights[i][start_index])
        
        # Reconstruct the path
        path = []
        mask = last_mask
        current_node = start_index
        for _ in range(n - 1, -1, -1):
            path.append(nodes[current_node])
            for i in range(n):
                if mask & (1 << i) and dp[mask][current_node] == dp[mask ^ (1 << current_node)][i] + edge_weights[i][current_node]:
                    mask ^= (1 << current_node)
                    current_node = i
                    break
        path.append(start_vertex)
        path.reverse()
        
        return min_cost, path


# <a id='4-2'></a>
# ### 4.2 Test Exercise 2 (`tsp_small_graph`)
# 
# Run the code below to test the `tsp_small_graph` on complete (fully connected) graphs with 10 nodes. The requirements for passing this exercise are:
# 
# - The algorithm must complete its run in under `1` second for each graph.
# - It must fund the optimal solution, starting at node 0.

# In[13]:


unittests.test_tsp_small_graph(GraphTSPSmallGraph)


# <a id='ex03'></a>
# ### Exercise 3: `tsp_large_graph` (Complete Graphs, 1,000 nodes, "Pretty Good" Solution, 0.5 seconds)
# 
# In this exercise you again tackle the Traveling Salesman Problem, but for much larger graphs of about 1,000 nodes. Once again the graph is complete, with an edge between every pair of nodes. In graphs this size, a brute force approach is now computationally infeasible, so the tour length requirement on this method has been loosened substantially. You now must simply find a "pretty good" tour through the graph using some kind of heuristic approach. There are several commonly-used heuristics, and almost all of them (with the exception perhaps of randomly generating a tour) will produce tours that are short enough to pass the tests on this method. While your solution no longer needs to be optimal, your code should run quickly, in less than `0.5` seconds. Have your LLM focus on speed rather than tour length, and you should find an algorithm that works!
# 
# Read **carefully** the function's docstring (the text describing the function), as the parameters and the return must **NOT** be changed.
# 
# **Important Note:** Do **NOT** include any new libraries beyond those imported at the beginning of the notebook. You may need to explicitly state this and ask the LLM **NOT** to use any additional libraries if necessary. Using additional libraries may cause you to fail the autograder, even if you pass the local tests.
# 
# **What you CAN do:** Create new functions **within the `GraphTSPLargeGraph` class**. Do **NOT** create functions outside the `GraphTSPLargeGraph` class or outside this graded cell. You may do so during development, but in the end, paste your solution into the graded cell following these instructions.

# In[14]:


class GraphTSPLargeGraph(Graph):
    def tsp_large_graph(self, start):
        """
        Solve the Travelling Salesman Problem for a large (~1000 node) complete graph starting from a specified node.
        No requirement to find the optimal tour. Must run under 0.5 seconds and find a solution.
        
        Parameters:
        start: The starting node.
        
        Returns:
        A tuple containing the total distance of the tour and a list of nodes representing the tour path.
        """
        # Initialize the tour
        current_node = start
        visited = {current_node}
        tour = [current_node]
        total_distance = 0

        # Use a greedy approach to construct the tour
        while len(visited) < len(self.graph):
            # Find the nearest unvisited neighbor
            nearest_neighbor = None
            shortest_distance = float('inf')
            for neighbor, weight in self.graph[current_node].items():
                if neighbor not in visited and weight < shortest_distance:
                    nearest_neighbor = neighbor
                    shortest_distance = weight
            
            # Update the tour
            tour.append(nearest_neighbor)
            visited.add(nearest_neighbor)
            total_distance += shortest_distance
            current_node = nearest_neighbor

        # Return to the starting node to complete the tour
        tour.append(start)
        total_distance += self.graph[current_node][start]
        
        return total_distance, tour


# <a id='4-3'></a>
# ### 4.3 Test Exercise 3 (`tsp_large_graph`)
# 
# Run the code below to test the `tsp_large_graph` on complete (fully connected) graphs with 1000 nodes. The requirements for passing this exercise are:
# 
# - The algorithm must complete its run in under `0.5` second for each graph.
# - It must find the good solution (less than a specified value, depending on the graph). 

# In[15]:


unittests.test_tsp_large_graph(GraphTSPLargeGraph)


# <a id='ex04'></a>
# ### Exercise 4: `tsp_medium_graph` (Complete Graphs, 300 nodes, Near-Optimal Solution, 1.5 seconds)
# 
# In this last version of the Traveling Salesman Problem, you will need to work with an LLM to develop a solution that improves upon the algorithm you used in Exercise 3. Now you will be given complete graphs of about 300 nodes. The time requirement has been relaxed to `1.5` seconds, giving your algorithm 3 times as long to run. The tour length requirements, however, have been tightened, meaning you'll need to find tours that are much closer to the theoretical optimum. This likely means that the heuristic you used in Exercise 3 will no longer produce short enough tours. Brainstorm with your LLM new heuristics you could implement that take advantage of the longer runtime you're allowed and generate relatively shorter tours.
# 
# Read **carefully** the function's docstring (the text describing the function), as the parameters and the return must **NOT** be changed.
# 
# **Important Note:** Do **NOT** include any new libraries beyond those imported at the beginning of the notebook. You may need to explicitly state this and ask the LLM **NOT** to use any additional libraries if necessary. Using additional libraries may cause you to fail the autograder, even if you pass the local tests.
# 
# **What you CAN do:** Create new functions **within the `GraphTSPMediumGraph` class**. Do **NOT** create functions outside the `GraphTSPMediumGraph` class or outside this graded cell. You may do so during development, but in the end, paste your solution into the graded cell following these instructions.

# In[16]:


# GRADED CELL 4 - Do NOT delete it, do NOT place your solution anywhere else. You can create new cells and work from there, but in the end add your solution in this cell.

class GraphTSPMediumGraph(Graph):
    def tsp_medium_graph(self, start):
        """
        Solve the Travelling Salesman Problem for a medium (~300 node) complete graph starting from a specified node.
        Expected to perform better than tsp_large_graph. Must run under 1.5 seconds.
        
        Parameters:
        start: The starting node.
        
        Returns:
        A tuple containing the total distance of the tour and a list of nodes representing the tour path.
        """
        # Your code here
        dist = None
        path = None
        return dist, path


# <a id='4-4'></a>
# ### 4.4 Test Exercise 4 (`tsp_medium_graph`)
# 
# Run the code below to test the `tsp_medium_graph` on complete (fully connected) graphs with 300 nodes. The requirements for passing this exercise are:
# 
# - The algorithm must complete its run in under `1.5` seconds for each graph.
# - It must find the good solution (less than a specified value, depending on the graph). 

# In[17]:


unittests.test_tsp_medium_graph(GraphTSPMediumGraph)


# ## Preparing Your Submission for Grading
# 
# Your submission will be evaluated by an automated grading system, known as an autograder. This system automatically reviews your notebook and assigns a grade based on specific criteria. It's important to note that the autograder will only evaluate the cells marked for grading and will not consider the content of the entire notebook. Therefore, if you include any additional content (such as print statements) outside the functions in the graded cells, it might disrupt the autograder's process. This discrepancy could be why you might pass all the unit tests but still encounter issues with the autograder.
# 
# To avoid such problems, please execute the following cell before submitting. This step will check for consistency within the graded cells but will not evaluate the correctness of your solutions—that aspect is determined by the unit tests. If the consistency check uncovers any issues, you'll have the opportunity to review and adjust your code accordingly.
# 
# **Remember, this check is focused on ensuring the graded cells are properly formatted and does not assess the accuracy of your answers.**

# In[ ]:


submission_checker.check_notebook()


# Once you feel good about your submission, **save your work and submit!** 
# 
# **Congratulations on completing the first major assignment in this specialization!**
