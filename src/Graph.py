class Graph:
    def __init__(self, edges):
        """
        Initializes a graph with a given number of vertices.
        The graph is represented by an adjacency list.
        """
        self.edges = edges
        self.vertices = self.get_number_of_vertices()
        self.graph = [
            [] for _ in range(self.vertices)
        ]  # List of lists to store adjacency list
        self.add_edges()  # Adding all the edges to the graph

    def get_number_of_vertices(self):
        """
        Determine the number of vertices by finding the maximum vertex index in the edge list.
        """
        max_vertex = 0
        for u, v, _ in self.edges:
            max_vertex = max(max_vertex, u, v)
        return max_vertex + 1  # Vertices are 0-indexed, so add 1 to get the total count

    def add_edges(self):
        """
        Adds a directed edge from vertex u to vertex v with the given capacity.
        Adds a reverse edge with 0 capacity for the residual graph.
        """
        for u, v, capacity in self.edges:
            self.graph[u].append([v, capacity])  # Add the forward edge
            self.graph[v].append(
                [u, 0]
            )  # Add the reverse edge (residual capacity initially 0)

    def BFS(self, source, sink, parent):
        """
        Breadth-First Search to find an augmenting path in the residual graph.
        """
        visited = [False] * self.vertices
        queue = [source]
        visited[source] = True

        while queue:
            u = queue.pop(0)

            for edge in self.graph[u]:
                v, capacity = edge
                if not visited[v] and capacity > 0:
                    parent[v] = u
                    if v == sink:
                        return True
                    queue.append(v)
                    visited[v] = True

        return False

    def FordFulkerson(self, source, sink):
        """
        Ford-Fulkerson method to calculate the maximum flow from source to sink.
        """
        parent = [-1] * self.vertices
        max_flow = 0

        while self.BFS(source, sink, parent):
            path_flow = float("Inf")
            s = sink

            while s != source:
                for edge in self.graph[parent[s]]:
                    if edge[0] == s:
                        path_flow = min(path_flow, edge[1])
                s = parent[s]

            max_flow += path_flow

            v = sink
            while v != source:
                u = parent[v]

                # Decrease the forward edge capacity
                for edge in self.graph[u]:
                    if edge[0] == v:
                        edge[1] -= path_flow

                # Increase the reverse edge capacity
                for edge in self.graph[v]:
                    if edge[0] == u:
                        edge[1] += path_flow

                v = parent[v]

        return max_flow

    def __str__(self):
        """
        Returns a string representation of the adjacency list of the graph.
        """
        result = []
        for i in range(self.vertices):
            edges = ", ".join(f"({v}, {capacity})" for v, capacity in self.graph[i])
            result.append(f"Vertex {i}: [{edges}]")
        return "\n".join(result)

    def reconstruct_graph(self):
        """
        Reconstructs and returns a graph showing only the forward edges
        used in the maximum flow after running the Ford-Fulkerson algorithm.
        """
        reconstructed_graph = [[] for _ in range(self.vertices)]

        for u in range(self.vertices):
            for edge in self.graph[u]:
                v, capacity = edge
                # Check if the edge had flow, i.e., if the capacity decreased
                if capacity == 0:  # Forward edge was fully used in flow
                    reconstructed_graph[u].append(v)  # Add only forward edge

        return reconstructed_graph

    def str_reconstructed_graph(self):
        """
        Returns a string representation of the reconstructed graph
        that only includes the forward edges used in the flow.
        """
        reconstructed_graph = self.reconstruct_graph()

        result = []
        result.append("Reconstructed Flow Graph (only forward edges used):")

        for u in range(self.vertices):
            edges = ", ".join(str(v) for v in reconstructed_graph[u])
            result.append(f"Vertex {u}: [{edges}]")

        return "\n".join(result)


def create_graph():
    preferences = [[2, 1], [2, 2], [1, 1], [2, 1], [0, 2]]
    places = [2, 3]


def num_verts(preferences, places):
    verts = 2
    for participant in preferences:
        for preference in participant:
            if preference == 2:
                verts += 1
                break  # Move to the next participant

    for _ in places:
        verts += 1

    return verts


if __name__ == "__main__":
    preferences = [[2, 1], [2, 2], [1, 1], [2, 1], [0, 2]]
    places = [2, 3]

    print(num_verts(preferences, places))

    # # Example usage:
    # edges = [
    #     # Super node to participants
    #     (0, 1, 1),  # P1
    #     (0, 2, 1),  # P2
    #     (0, 3, 1),  # P3
    #     (0, 4, 1),  # P4
    #     (1, 5, 1),
    #     (2, 5, 1),
    #     (2, 6, 1),
    #     (3, 5, 1),
    #     (4, 6, 1),
    #     (5, 7, 2),
    #     (6, 7, 2),
    # ]

    # edges = [
    #     (0, 1, 1),
    #     (1, 2, 1),
    #     (1, 3, 1),
    #     (2, 4, 0),
    #     (3, 4, 1),
    # ]

    # graph = Graph(edges)

    # source = 0
    # sink = 4

    # max_flow = graph.FordFulkerson(source, sink)
    # print(graph)

    # print("The maximum possible flow is:", max_flow)
