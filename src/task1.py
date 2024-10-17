class Graph:
    """
    Class description:  
    The Graph class implements the construction of a flow network and computes the maximum flow 
    using the Ford-Fulkerson algorithm. It also provides functionality to reconstruct the graph 
    to display only forward edges that were fully utilized in the flow.
    """
    def __init__(self, edges: list):
        """
        Function description:  
        Initializes the Graph with given edges, constructs an adjacency list, and computes 
        the max flow using Ford-Fulkerson.

        Input:  
        - edges: A list of tuples representing edges in the form (u, v, capacity).  

        Output: None

        Time complexity: O(V * E^2)
        - V is the number of vertices represented by self.vertices.
        - E is the number of edges in self.edges

        Time complexity analysis:  
        - The init first uses the get_number_of_vertices_sink method which has complexity of O(E)
        - Then self.graph is initialised in O(V) time
        - add_edges method is called which has complexity O(E)
        - FordFulkerson method is then called with a complexity of O(V * E^2) .
        - Finally, the reconstruct_graph method is called in O(V + E).
        - Thus, the sum of all these complexities is: O(3*E + 2*V + V * E^2) 
        - This can then be simplified to O(V * E^2)

        Space complexity: O(V + E)

        Space complexity analysis:
        - O(V): Storing the adjacency list with V empty lists, one for each vertex.
        - O(E): Storing all edges in the graph. Each edge contributes an entry in the adjacency list.
        - O(V): For the parent list used during BFS in Ford-Fulkerson.
        - O(E): Additional space for storing the reverse edges in the residual graph.
        """
        # Initialize the graph with edges and set up vertices, source, and sink.
        self.edges = edges
        self.vertices, self.source, self.sink = self.get_number_of_vertices_sink()
        # Create an adjacency list for storing the graph.
        self.graph = [[] for _ in range(self.vertices)]
        self.add_edges()  # Populate the graph with edges.
        self.FordFulkerson()  # Run Ford-Fulkerson to find the max flow.
        self.reconstructed_graph = self.reconstruct_graph()  # Build the residual graph.

    def get_number_of_vertices_sink(self):
        """
        Function description:  
        Calculates the total number of vertices by finding what the highest vertex value is in the edges list
        and also determines the source and sink vertices.

        Input: None  

        Output:  
        - (vertices, source, sink): Tuple containing the number of vertices, source node (0), 
          and sink node (highest-indexed vertex).

        Time complexity: O(E)  
        - E is the number of edges in self.edges

        Time complexity analysis:  
        - The function iterates over all edges to find the maximum vertex index.

        Space complexity: O(1)

        Space complexity analysis:
        - The space which the max_vertex takes when iterating through is O(1)
        """
        # Find the number of vertices by determining the maximum vertex index.
        max_vertex = 0
        for u, v, _ in self.edges:
            max_vertex = max(max_vertex, u, v)  # Track the largest vertex index.
        source = 0  # The source vertex is always 0.
        sink = max_vertex  # Sink is the highest indexed vertex.
        return max_vertex + 1, source, sink  # Total vertices are max index + 1.

    def add_edges(self):
        """
        Function description:  
        Adds forward and reverse edges to the adjacency list to represent capacity and 
        residual capacity for the flow network.

        Input: None  

        Output: None  

        Time complexity: O(E)  
        - E is the number of edges in self.edges

        Time complexity analysis:  
        - Each edge is processed once to add both forward and reverse (residual) edges.

        Space complexity: O(E)  
        - E is the number of edges in self.edges

        Space complexity analysis:  
        - The adjacency list represented by self.graph requires 2*E of space for every edges, therefore the space
        complexity simplifies to O(E).
        """
        # Add forward and reverse edges to the graph to represent capacity and residual capacity.
        for u, v, capacity in self.edges:
            self.graph[u].append([v, capacity])  # Add the forward edge with capacity.
            self.graph[v].append([u, 0])  # Add the reverse edge with 0 capacity.

    def BFS(self, parent: list):
        """
        Function description:  
        Performs a Breadth-First Search (BFS) to find an augmenting path from the source to the sink in the residual graph. 
        The graph is represented as an adjacency list in self.graph. An augmenting path is a path with available capacity 
        where additional flow can be pushed.

        Input:  
        - parent: List to store the path from the source to the sink.  

        Output:  
        - Boolean indicating if an augmenting path exists.

        Time complexity: O(V + E)  
        - V is the number of vertices represented by self.vertices.
        - E is the number of edges in self.edges

        Time complexity analysis:  
        - In BFS, each vertex is visited at most once. For every visited vertex, all its adjacent edges are processed once. 
        Thus, the total time complexity is O(V + E).

        Space complexity: O(V)  
        - V is the number of vertices represented by self.vertices.

        Space complexity analysis:  
        - The space complexity comes from storing the visited list, the queue, and the parent list.
        - The visited list requires O(V) space to keep track of visited nodes.
        - The queue can contain up to O(V) elements in the worst case.
        - The parent list also takes O(V) space to store the path information.
        """
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
        """
        Function description:  
        Implements the Ford-Fulkerson algorithm to compute the maximum flow from the source 
        to the sink in a flow network. This algorithm repeatedly finds augmenting paths using 
        the Breadth-First Search (BFS) and pushes flow along these paths until no more augmenting paths exist.
        Algorithm explanation:  
        1. Use BFS to find an augmenting path from the source to the sink in the residual graph.
        2. Compute the minimum capacity (bottleneck) along the path found.
        3. Update the flow along the path:
        - Subtract the bottleneck capacity from forward edges.
        - Add the bottleneck capacity to reverse edges to reflect flow adjustments.
        4. Repeat the process until no more augmenting paths can be found.

        Input: None  

        Output:  
        - The computed maximum flow value.

        Time complexity: O(V * E^2)  
        - V is the number of vertices, and E is the number of edges.

        Time complexity analysis:
        - Each BFS call takes O(V + E). In the worst case, the algorithm performs E augmenting 
        path searches for every unit of flow added, leading to a time complexity of O(V * E^2).

        Space complexity: O(V + E)  
        - V is the number of vertices represented by self.vertices.
        - E is the number of edges in self.edges

        Space complexity analysis:  
        - The parent list takes O(V) space to store the path.
        - The graph uses O(E) space to represent edges and their capacities.
        """
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
        """
        Function description:  
        Reconstructs the residual graph to display only the fully utilised forward edges from the flow computation.  
        This reconstructed graph provides a clearer representation of the paths where all available capacity was exhausted.

        Input: None  

        Output:  
        - A reconstructed adjacency list containing only the used forward edges.

        Time complexity: O(V + E) 
        - V is the number of vertices represented by self.vertices.
        - E is the number of edges in self.edges 

        Time complexity analysis:
        - We visit each vertex once O(V) to initialize and traverse through its outgoing edges.  
        - For each vertex, we iterate over its adjacency list, which sums to E operations across all vertices.
        - Thus, the total time spent is proportional to the number of vertices and edges combined: O(V + E).  
        - This is an optimal traversal because every edge is processed exactly once.

        Space complexity: O(V + E)
        - V is the number of vertices represented by self.vertices.
        - E is the number of edges in self.edges

        Space complexity analysis:
        - Even though not all edges from the original graph may be included (only those with 0 capacity are added), 
        in the worst case, every edge could be fully utilised and added to the reconstructed graph.  
        - Therefore, the space required will still be proportional to the sum of vertices and edges: O(V + E).
        """
        # Reconstruct the graph to show only used forward edges in the flow.
        reconstructed_graph = [[] for _ in range(self.vertices)]

        for u in range(self.vertices):
            for edge in self.graph[u]:
                v, capacity = edge
                if capacity == 0:  # Check if the edge was fully used.
                    reconstructed_graph[u].append(v)  # Add forward edge to reconstructed graph.

        return reconstructed_graph

class Network:
    """
    Class description:  
    Represents a flow network for assigning participants to activities based on preferences.
    """
    def __init__(self, preferences: list, places: list) -> None:
        """
        Function description:  
        Initializes the Network class with preferences, activity capacities, and constructs the flow network.

        Input:  
        - preferences: List of participant preferences for activities.  
        - places: List of activity capacities.  

        Output: None  

        Time complexity: O(1)  
        Time complexity analysis:  
        - The function initializes attributes without complex operations.

        Space complexity: O(N + M)  
        Space complexity analysis:  
        - Memory is required for storing preferences, places, and network data.
        """
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

    def add_participant_edges(self, num_participants: int):
        # Create edges from the source to each participant.
        return [(0, participant + 1, 1) for participant in range(num_participants)]

    def calculate_offsets(self, num_participants: int, num_activities: int):
        # Calculate the offsets for different types of nodes and the sink.
        leader_offset = num_participants + 1
        non_leader_offset = leader_offset + num_activities
        activity_offset = non_leader_offset + num_activities
        sink = activity_offset + num_activities

        return leader_offset, non_leader_offset, activity_offset, sink

    def track_leader_non_leader_nodes(self, num_activities: int, leader_offset: int, non_leader_offset: int):
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

    def add_leader_activity_edges(self, num_activities: int, leader_offset: int, activity_offset: int):
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

    def add_activity_sink_edges(self, num_activities: int, activity_offset: int, sink: int):
        # Create edges from activity nodes to the sink.
        return [(activity_offset + i, sink, self.places[i]) for i in range(num_activities)]


def assign(preferences: list, places: list):
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
