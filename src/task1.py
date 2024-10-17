class Graph:
    def __init__(self, edges):
        # Initialize the graph with edges and set up vertices, source, and sink.
        self.edges = edges
        self.vertices, self.source, self.sink = self.get_number_of_vertices_sink()
        # Create an adjacency list for storing the graph.
        self.graph = [[] for _ in range(self.vertices)]
        self.add_edges()  # Populate the graph with edges.
        self.FordFulkerson()  # Run Ford-Fulkerson to find the max flow.
        self.reconstructed_graph = self.reconstruct_graph()  # Build the residual graph.

    def get_number_of_vertices_sink(self):
        # Find the number of vertices by determining the maximum vertex index.
        max_vertex = 0
        for u, v, _ in self.edges:
            max_vertex = max(max_vertex, u, v)  # Track the largest vertex index.
        source = 0  # The source vertex is always 0.
        sink = max_vertex  # Sink is the highest indexed vertex.
        return max_vertex + 1, source, sink  # Total vertices are max index + 1.

    def add_edges(self):
        # Add forward and reverse edges to the graph to represent capacity and residual capacity.
        for u, v, capacity in self.edges:
            self.graph[u].append([v, capacity])  # Add the forward edge with capacity.
            self.graph[v].append([u, 0])  # Add the reverse edge with 0 capacity.

    def BFS(self, parent):
        # Perform a Breadth-First Search to find an augmenting path.
        visited = [False] * self.vertices  # Track visited nodes.
        queue = [self.source]  # Start BFS from the source node.
        visited[self.source] = True  # Mark source as visited.

        while queue:
            u = queue.pop(0)  # Dequeue the front node.

            # Traverse all adjacent nodes to find augmenting paths.
            for edge in self.graph[u]:
                v, capacity = edge
                if not visited[v] and capacity > 0:  # Check if edge can carry more flow.
                    parent[v] = u  # Record the path.
                    if v == self.sink:  # Stop if sink is reached.
                        return True
                    queue.append(v)  # Enqueue the adjacent vertex.
                    visited[v] = True  # Mark as visited.

        return False  # No augmenting path found.

    def FordFulkerson(self):
        # Implement the Ford-Fulkerson algorithm to compute the max flow.
        parent = [-1] * self.vertices  # Store the augmenting path.
        max_flow = 0  # Initialize the max flow.

        while self.BFS(parent):  # Continue while there is an augmenting path.
            path_flow = float("Inf")  # Set path flow to infinity initially.
            s = self.sink

            # Find the minimum capacity along the augmenting path.
            while s != self.source:
                for edge in self.graph[parent[s]]:
                    if edge[0] == s:
                        path_flow = min(path_flow, edge[1])
                s = parent[s]

            max_flow += path_flow  # Accumulate the flow.

            v = self.sink
            while v != self.source:
                u = parent[v]

                # Decrease the capacity of the forward edge by path flow.
                for edge in self.graph[u]:
                    if edge[0] == v:
                        edge[1] -= path_flow

                # Increase the capacity of the reverse edge by path flow.
                for edge in self.graph[v]:
                    if edge[0] == u:
                        edge[1] += path_flow

                v = parent[v]

        return max_flow  # Return the computed max flow.

    def reconstruct_graph(self):
        # Reconstruct the graph to show only used forward edges in the flow.
        reconstructed_graph = [[] for _ in range(self.vertices)]

        for u in range(self.vertices):
            for edge in self.graph[u]:
                v, capacity = edge
                if capacity == 0:  # Check if the edge was fully used.
                    reconstructed_graph[u].append(v)  # Add forward edge to reconstructed graph.

        return reconstructed_graph

    def __str__(self):
        # Create a string representation of the reconstructed graph.
        reconstructed_graph = self.reconstruct_graph()

        result = []
        result.append("Reconstructed Flow Graph (only forward edges used):")

        for u in range(self.vertices):
            edges = ", ".join(str(v) for v in reconstructed_graph[u])
            result.append(f"Vertex {u}: [{edges}]")

        return "\n".join(result)


class Network:
    def __init__(self, preferences, places) -> None:
        # Initialize with participants' preferences and activity places.
        self.preferences = preferences
        self.places = places
        self.activity_nodes = []  # Store leader and non-leader nodes.
        self.edges = self.create_flow_network()  # Create the flow network.

    def create_flow_network(self):
        # Build the flow network by adding all necessary edges.
        edges = []
        num_participants = len(self.preferences)  # Total number of participants.
        num_activities = len(self.places)  # Total number of activities.

        edges += self.add_participant_edges(num_participants)  # Add source-participant edges.
        leader_offset, non_leader_offset, activity_offset, sink = self.calculate_offsets(num_participants, num_activities)
        self.track_leader_non_leader_nodes(num_activities, leader_offset, non_leader_offset)
        edges += self.add_preference_edges(leader_offset, non_leader_offset)  # Add preference-based edges.
        edges += self.add_leader_activity_edges(num_activities, leader_offset, activity_offset)  # Add leader-activity edges.
        edges += self.add_non_leader_activity_edges(num_activities, non_leader_offset, activity_offset, leader_offset)  # Add non-leader-activity edges.
        edges += self.add_activity_sink_edges(num_activities, activity_offset, sink)  # Add activity-sink edges.

        return edges

    def add_participant_edges(self, num_participants):
        # Create edges from the source to each participant.
        return [(0, participant + 1, 1) for participant in range(num_participants)]

    def calculate_offsets(self, num_participants, num_activities):
        # Calculate the offsets for different types of nodes and the sink.
        leader_offset = num_participants + 1
        non_leader_offset = leader_offset + num_activities
        activity_offset = non_leader_offset + num_activities
        sink = activity_offset + num_activities

        return leader_offset, non_leader_offset, activity_offset, sink

    def track_leader_non_leader_nodes(self, num_activities, leader_offset, non_leader_offset):
        # Store the leader and non-leader nodes for each activity.
        self.activity_nodes = [(leader_offset + i, non_leader_offset + i) for i in range(num_activities)]

    def add_preference_edges(self, leader_offset, non_leader_offset):
        # Create edges from participants to leader or non-leader nodes based on preferences.
        edges = []
        for participant, preference in enumerate(self.preferences):
            for activity, interest_level in enumerate(preference):
                if interest_level == 2:  # Experienced participant interested in the activity.
                    edges.append((participant + 1, leader_offset + activity, 1))
                elif interest_level == 1:  # Interested but not experienced.
                    edges.append((participant + 1, non_leader_offset + activity, 1))
        return edges

    def add_leader_activity_edges(self, num_activities, leader_offset, activity_offset):
        # Create edges from leader nodes to activity nodes with capacity 2.
        return [(leader_offset + i, activity_offset + i, 2) for i in range(num_activities)]

    def add_non_leader_activity_edges(self, num_activities, non_leader_offset, activity_offset, leader_offset):
        # Create edges from non-leader nodes to activity nodes with remaining capacity.
        edges = []
        for activity in range(num_activities):
            leader_node = leader_offset + activity
            non_leader_node = non_leader_offset + activity
            activity_node = activity_offset + activity
            edges.append((non_leader_node, activity_node, self.places[activity] - 2))  # Capacity for non-leaders.
            edges.append((leader_node, non_leader_node, self.places[activity] - 2))  # Leader to non-leader connection.
        return edges

    def add_activity_sink_edges(self, num_activities, activity_offset, sink):
        # Create edges from activity nodes to the sink.
        return [(activity_offset + i, sink, self.places[i]) for i in range(num_activities)]


def assign(preferences, places):
    network = Network(preferences, places)  # Build the flow network.
    graph = Graph(network.edges)  # Run max flow on the network.

    activity_nodes = network.activity_nodes  # Get leader/non-leader nodes for activities.
    participant_assign = graph.reconstructed_graph[1:len(preferences) + 1]  # Extract assignments.

    result = [[] for _ in range(len(places))]  # Prepare result list for activities.
    count = 0  # Track the number of assigned participants.

    for i, (leader, non_leader) in enumerate(activity_nodes):  # Iterate over activities.
        for j, assigned_node in enumerate(participant_assign):  # Check each participant's assignment.
            if assigned_node[0] in (leader, non_leader):  # Match leader/non-leader nodes.
                result[i].append(j)  # Add participant to the activity.
                count += 1  # Increment assigned participant count.

    return result if count == len(preferences) else None  # Return result if all are assigned.


if __name__ == "__main__":
    preferences = [[2, 1], [2, 2], [1, 1], [2, 1], [0, 2]]
    places = [2, 3]

    print(assign(preferences, places))
