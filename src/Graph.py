class Graph:
    def __init__(self, edges):
        """
        Initializes a graph with a given number of vertices.
        The graph is represented by an adjacency list.
        """
        self.edges = edges
        self.vertices, self.source, self.sink = self.get_number_of_vertices_sink()
        self.graph = [[] for _ in range(self.vertices)]  # List of lists to store adjacency list
        self.add_edges()  # Adding all the edges to the graph
        self.FordFulkerson()
        self.reconstructed_graph = self.reconstruct_graph()

    def get_number_of_vertices_sink(self):
        """
        Determine the number of vertices by finding the maximum vertex index in the edge list.
        """
        max_vertex = 0
        for u, v, _ in self.edges:
            max_vertex = max(max_vertex, u, v)
        return max_vertex + 1, 0, max_vertex  # Vertices are 0-indexed, so add 1 to get the total count

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

    def BFS(self, parent):
        """
        Breadth-First Search to find an augmenting path in the residual graph.
        """
        visited = [False] * self.vertices
        queue = [self.source]
        visited[self.source] = True

        while queue:
            u = queue.pop(0)

            for edge in self.graph[u]:
                v, capacity = edge
                if not visited[v] and capacity > 0:
                    parent[v] = u
                    if v == self.sink:
                        return True
                    queue.append(v)
                    visited[v] = True

        return False

    def FordFulkerson(self):
        """
        Ford-Fulkerson method to calculate the maximum flow from self.source to sink.
        """
        parent = [-1] * self.vertices
        max_flow = 0

        while self.BFS(parent):
            path_flow = float("Inf")
            s = self.sink

            while s != self.source:
                for edge in self.graph[parent[s]]:
                    if edge[0] == s:
                        path_flow = min(path_flow, edge[1])
                s = parent[s]

            max_flow += path_flow

            v = self.sink
            while v != self.source:
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

    def reconstruct_graph(self):
        """
        Reconstructs and returns a graph showing only the forward edges
        used in the maximum flow after running the Ford-Fulkerson algorithm.
        """
        reconstructed_graph = [[] for _ in range(self.vertices)]

        for u in range(self.vertices):
            for edge in self.graph[u]:
                v, capacity = edge
                # Check if the edge had flow (if the capacity decreased)
                if capacity == 0:  # Forward edge was fully used in flow
                    reconstructed_graph[u].append(v)  # Add only forward edge

        return reconstructed_graph

    def __str__(self):
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


class Network:
    def __init__(self, preferences, places) -> None:
        self.preferences = preferences
        self.places = places
        self.activity_nodes = []
        self.edges = self.create_flow_network()

    def create_flow_network(self):
        edges = []
        num_participants = len(self.preferences)
        num_activities = len(self.places)

        # Create edges from source to participants
        edges += self.add_participant_edges(num_participants)

        # Track leader and non-leader nodes for each activity (printed)
        leader_offset, non_leader_offset, activity_offset, sink = self.calculate_offsets(num_participants, num_activities)
        self.track_leader_non_leader_nodes(num_activities, leader_offset, non_leader_offset)

        # Create edges from participants to leader or non-leader nodes based on preferences
        edges += self.add_preference_edges(leader_offset, non_leader_offset)

        # Create edges from leader nodes to activity nodes
        edges += self.add_leader_activity_edges(num_activities, leader_offset, activity_offset)

        # Create edges from non-leader nodes to activity nodes with remaining capacity
        edges += self.add_non_leader_activity_edges(num_activities, non_leader_offset, activity_offset, leader_offset)

        # Create edges from activity nodes to the sink
        edges += self.add_activity_sink_edges(num_activities, activity_offset, sink)

        return edges

    def add_participant_edges(self, num_participants):
        """Creates edges from the source to each participant."""
        edges = []
        for participant in range(num_participants):
            edges.append((0, participant + 1, 1))  # Capacity of 1 for each participant
        return edges

    def calculate_offsets(self, num_participants, num_activities):
        """Calculates the offset values for leader, non-leader, and activity nodes, as well as the sink."""
        leader_offset = num_participants + 1
        non_leader_offset = leader_offset + num_activities
        activity_offset = non_leader_offset + num_activities
        sink = activity_offset + num_activities
        
        return leader_offset, non_leader_offset, activity_offset, sink

    def track_leader_non_leader_nodes(self, num_activities, leader_offset, non_leader_offset):
        """Prints and stores the leader and non-leader nodes for each activity."""

        for activity in range(num_activities):
            leader_node = leader_offset + activity
            non_leader_node = non_leader_offset + activity
            
            # Store the nodes directly in the list
            self.activity_nodes.append((leader_node, non_leader_node))


    def add_preference_edges(self, leader_offset, non_leader_offset):
        """Creates edges from participants to leader or non-leader nodes based on preferences."""
        edges = []
        for participant, preference in enumerate(self.preferences):
            for activity, interest_level in enumerate(preference):
                if interest_level == 2:
                    # Participant is interested and experienced
                    edges.append((participant + 1, leader_offset + activity, 1))
                elif interest_level == 1:
                    # Participant is interested but not experienced
                    edges.append((participant + 1, non_leader_offset + activity, 1))
        return edges

    def add_leader_activity_edges(self, num_activities, leader_offset, activity_offset):
        """Creates edges from leader nodes to activity nodes with capacity 2."""
        edges = []
        for activity in range(num_activities):
            leader_node = leader_offset + activity
            activity_node = activity_offset + activity
            edges.append((leader_node, activity_node, 2))  # Two leaders needed for each activity
        return edges

    def add_non_leader_activity_edges(self, num_activities, non_leader_offset, activity_offset, leader_offset):
        """Creates edges from non-leader nodes to activity nodes with remaining capacity."""
        edges = []
        for activity in range(num_activities):
            leader_node = leader_offset + activity
            non_leader_node = non_leader_offset + activity
            activity_node = activity_offset + activity
            edges.append((non_leader_node, activity_node, self.places[activity] - 2))  # Remaining capacity for non-leaders
            edges.append((leader_node, non_leader_node, self.places[activity] - 2)) # Leader to non leader nodes 
        return edges

    def add_activity_sink_edges(self, num_activities, activity_offset, sink):
        """Creates edges from activity nodes to the sink."""
        edges = []
        for activity in range(num_activities):
            activity_node = activity_offset + activity
            edges.append((activity_node, sink, self.places[activity]))  # Total capacity of each activity
        return edges

def assign(preferences, places):
    # Initialize the result list with empty sublists for each activity
    len_participants = len(preferences)
    len_places = len(places)

    result = [[] for _ in range(len_places)]

    network = Network(preferences, places)
    graph = Graph(network.edges)
    
    reconstructed_graph = graph.reconstructed_graph
    activity_nodes = network.activity_nodes

    participant_assign = reconstructed_graph[1:len_participants + 1]  # [[6], [7], [9], [6], [7]]


    count = 0
    # Iterate over each activity and its corresponding nodes (leader and non-leader)
    for activity_index, activity_node in enumerate(activity_nodes):  # (6, 8)
        # Iterate over each participant and their assigned node
        for participant_index, participant in enumerate(participant_assign):
            # Check both the leader (index 0) and non-leader (index 1) nodes
            for i in range(2):
                # If the participant's node matches the current activity node
                if participant[0] == activity_node[i]:
                    # Add the participant's index to the result list for this activity
                    result[activity_index].append(participant_index)
                    count += 1

    if count != len_participants:
        return None

    return result  # Return the list of participant assignments for each activity




if __name__ == "__main__":
    # preferences = [[2, 1], [2, 2], [1, 1], [2, 1], [0, 2]]
    # places = [2, 3]
    preferences = [[2, 1], [2, 1], [2, 0], [1, 2], [1, 2]]
    places = [3, 2]

    print(assign(preferences, places))

    network = Network(preferences, places)


    graph = Graph(network.edges)

    # print(graph.reconstructed_graph)

    edges = [
        (0, 1, 1),
        (0, 2, 1),
        (0, 3, 1),
        (0, 4, 1),
        (0, 5, 1),
        (0, 5, 1),
        (1, 6, 1),
        (1, 9, 1),
        (2, 6, 1),
        (2, 8, 1),
        (3, 7, 1),
        (3, 9, 1),
        (4, 6, 1),
        (4, 9, 1),
        (5, 8, 1),
        (6, 10, 2),
        (8, 11, 2),
        (8, 9, 1),
        (9, 11, 1),
        (10, 12, 2),
        (11, 12, 3),
    ]

    # edges = [
    #     (0, 1, 1),
    #     (0, 2, 1),
    #     (0, 3, 1),
    #     (0, 4, 1),
    #     (0, 5, 1),
    #     (0, 5, 1),
    #     (1, 6, 1),
    #     (1, 8, 1),
    #     (2, 6, 1),
    #     (2, 8, 1),
    #     (3, 6, 1),
    #     (3, 8, 1),
    #     (4, 6, 1),
    #     (4, 8, 1),
    #     (5, 6, 1),
    #     (5, 8, 1),
    #     (6, 10, 2),
    #     (8, 11, 2),
    #     (8, 9, 1),
    #     (9, 11, 1),
    #     (10, 12, 2),
    #     (11, 12, 3),
    # ]
