class Graph:
    """
    Class description:  
    The Graph class implements the construction of a flow network and computes the maximum flow 
    using the Ford-Fulkerson algorithm. It also provides functionality to reconstruct the graph 
    to display only forward edges that were fully utilised in the flow.
    """
    def __init__(self, edges: list):
        """
        Function description:  
        Initialises the Graph with given edges, constructs an adjacency list, and computes 
        the max flow using Ford-Fulkerson.

        Input:  
        - edges: A list of tuples representing edges in the form (u, v, capacity).  

        Output: None

        Time complexity: O(E * F)
        - E is the number of edges in self.edges
        - F is the max flow of the network


        Time complexity analysis:  
        - The init first uses the get_number_of_vertices_sink method which has complexity of O(E)
        - Then self.graph is initialised in O(V) time
        - add_edges method is called which has complexity O(E)
        - FordFulkerson method is then called with a complexity of O(E * F).
        - Finally, the reconstruct_graph method is called in O(V + E).
        - Thus, the sum of all these complexities is: O(3*E + 2*V + E * F) 
        - This can then be simplified to O(E * F)

        Space complexity: O(V + E)

        Space complexity analysis:
        - O(V): Storing the adjacency list with V empty lists, one for each vertex.
        - O(E): Storing all edges in the graph. Each edge contributes an entry in the adjacency list.
        - O(V): For the parent list used during BFS in Ford-Fulkerson.
        - O(E): Additional space for storing the reverse edges in the residual graph.
        """
        # Initialise the graph with edges and set up vertices, source, and sink.
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

        Time complexity: O(E * F) 
        - V is the number of vertices, and E is the number of edges.
        - F is the max flow of the network

        Time complexity analysis:
        - The max number of augmenting path searches needed is F and if each search with BFS
        takes O(V + E), then the complexity becomes O(F * (E + V))
        - Since we know that E will always be larger than V, the complexity simplifies to O(F * E)

        Space complexity: O(V + E)  
        - V is the number of vertices represented by self.vertices.
        - E is the number of edges in self.edges

        Space complexity analysis:  
        - The parent list takes O(V) space to store the path.
        - The graph uses O(E) space to represent edges and their capacities.
        """
        # Implement the Ford-Fulkerson algorithm to compute the max flow.
        parent = [-1] * self.vertices  # Store the augmenting path.
        max_flow = 0  # Initialise the max flow.

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
        - We visit each vertex once O(V) to initialise and traverse through its outgoing edges.  
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
    Represents a flow network for assigning participants to activities based on their preferences.  
    This class models the problem of allocating participants to activities while ensuring that:
    - Each participant can express varying levels of interest (experienced or not).
    - Each activity has a defined capacity that dictates how many participants can be assigned.
    - Experienced participants are prioritised over non-experienced ones when first assigning to activities.
    
    The flow network is constructed using a directed graph where:
    - Nodes represent the source, participant nodes, leader and non-leader nodes, activities, and a sink.
    - Edges represent the allowable assignments between these nodes with associated capacities.

    """
    def __init__(self, preferences: list, places: list) -> None:
        """
        Function description:  
        Initialises the Network class with preferences, activity capacities, and constructs the flow network.

        Input:  
        - preferences: List of participant preferences for activities.  
        - places: List of activity capacities.  

        Output: None  

        Time complexity: O(N^2)
        - N is the number of participants

        Time complexity analysis:  
        - The init calls the create_flow_network method which has a complexity of O(N^2)

        Space complexity: O(N^2)
        - N is the number of participants

        Space complexity analysis:  
        - Calls the create_flow_network method which has a complexity of O(N^2)
        """
        # Initialise with participants' preferences and activity places.
        self.preferences = preferences
        self.places = places
        self.activity_nodes = []  # Store leader and non-leader nodes.
        self.edges = self.create_flow_network()  # Create the flow network.

    def create_flow_network(self):
        """
        Function description:  
        Builds the complete flow network by constructing all necessary edges between nodes. This method combines all the other
        smaller methods relating to the construction of the flow network. It constructs the network in the following way:
        Adds edges from source to participant nodes
        Adds edges from participant to leader/non-leader nodes
        Adds edges from leader/non-leader nodes to activity nodes
        Adds edges from activity nodes to sink node
        
        Input: None

        Output:  
        - List of tuples representing the edges in the flow network.

        Time complexity: O(N^2)
        - N is the number of participants

        Time complexity analysis:  
        - Adding preference-based edges involves iterating over participants N and activities M.
        - Since we know that M is at most N/2, the complexity becomes O(N * N/2) which comes from the add_preference_edges method.
        - Therefore the complexity of this method is O(N^2).

        Space complexity: O(N^2)  
        - N is the number of participants

        Space complexity analysis:  
        - Space complexity of the add_preference_edges method is O(N^2) as well
        """
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
        """
        Function description:  
        Creates edges from the source to each participant with capacity 1.

        Input:  
        - num_participants: Integer representing the total number of participants.

        Output:  
        - List of tuples representing source-to-participant edges.

        Time complexity: O(N)
        - N is the number of participants

        Time complexity analysis:  
        - Each participant is processed exactly once.

        Space complexity: O(N)  
        - N is the number of participants

        Space complexity analysis:  
        - O(N) space is used to store the participant edges.
        """ 
        # Create edges from the source to each participant.
        return [(0, participant + 1, 1) for participant in range(num_participants)]

    def calculate_offsets(self, num_participants: int, num_activities: int):
        """
        Function description:  
        Calculates offsets to correctly index different types of nodes in the flow network.

        Input:  
        - num_participants: Number of participants.
        - num_activities: Number of activities.

        Output:  
        - Tuple containing the offsets for leader, non-leader, activity nodes, and the sink.

        Time complexity: O(1)  
        Time complexity analysis:  
        - Simple arithmetic operations are performed.

        Space complexity: O(1)  
        Space complexity analysis:  
        - No additional space is required.
        """
        # Calculate the offsets for different types of nodes and the sink.
        leader_offset = num_participants + 1
        non_leader_offset = leader_offset + num_activities
        activity_offset = non_leader_offset + num_activities
        sink = activity_offset + num_activities

        return leader_offset, non_leader_offset, activity_offset, sink

    def track_leader_non_leader_nodes(self, num_activities: int, leader_offset: int, non_leader_offset: int):
        """
        Function description:  
        Tracks and stores the leader and non-leader nodes for each activity.

        Input:  
        - num_activities: Number of activities.
        - leader_offset: Index offset for leader nodes.
        - non_leader_offset: Index offset for non-leader nodes.

        Output: None  

        Time complexity: O(M)  
        - M is the number of activities

        Time complexity analysis:  
        - Each activity is processed once.

        Space complexity: O(M)  
        - M is the number of activities

        Space complexity analysis:  
        - O(M) space is used to store the leader and non-leader nodes.
        """
        # Store the leader and non-leader nodes for each activity.
        self.activity_nodes = [(leader_offset + i, non_leader_offset + i) for i in range(num_activities)]

    def add_preference_edges(self, leader_offset, non_leader_offset):
        """
        Function description:  
        Creates edges from participants to leader or non-leader nodes based on their preferences.

        Input:  
        - leader_offset: Index offset for leader nodes.
        - non_leader_offset: Index offset for non-leader nodes.

        Output:  
        - List of tuples representing preference-based edges.

        Time complexity: O(N^2)  
        - N is the number of participants

        Time complexity analysis:  
        - For each participant N, their preferences for all activities M are processed.
        - From the spec we are told that M is at most N/2.
        - There is a nested for loop which iterates over the all activities for each participant, therefore the complexity 
        becomes O(N * N/2).
        - We can simplify this to O(N^2)

        Space complexity: O(N^2)  
        - N is the number of participants.

        Space complexity analysis:  
        - For each participant, the max number of edges they can have is if they are interested in all activities.
        - Therefore this takes up O(N * M) space which becomes O(N^2)
        """
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
        """
        Function description:  
        Creates edges from leader nodes to activity nodes with capacity 2.

        Input:  
        - num_activities: Number of activities.
        - leader_offset: Index offset for leader nodes.
        - activity_offset: Index offset for activity nodes.

        Output:  
        - List of tuples representing leader-to-activity edges.

        Time complexity: O(M)  
        - M is the number of activities

        Time complexity analysis:  
        - Each activity is processed once.

        Space complexity: O(M)  
        - M is the number of activities

        Space complexity analysis:  
        - O(M) space is used to store the edges.
        """
        # Create edges from leader nodes to activity nodes with capacity 2.
        return [(leader_offset + i, activity_offset + i, 2) for i in range(num_activities)]

    def add_non_leader_activity_edges(self, num_activities, non_leader_offset, activity_offset, leader_offset):
        """
        Function description:  
        Creates edges from non-leader nodes to activity nodes with the remaining capacity.

        Input:  
        - num_activities: Number of activities.
        - non_leader_offset: Index offset for non-leader nodes.
        - activity_offset: Index offset for activity nodes.
        - leader_offset: Index offset for leader nodes.

        Output:  
        - List of tuples representing non-leader-to-activity and leader-to-non-leader edges.

        Time complexity: O(M)  
        - M is the number of activities

        Time complexity analysis:  
        - Each activity is processed once.

        Space complexity: O(M)  
        - M is the number of activities

        Space complexity analysis:  
        - O(M) space is used to store the edges.
        """
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
        """
        Function description:  
        Creates edges from activity nodes to the sink with capacities equal to the activity's capacity in the list of places.

        Input:  
        - num_activities: Number of activities.
        - activity_offset: Index offset for activity nodes.
        - sink: Index of the sink node.

        Output:  
        - List of tuples representing activity-to-sink edges.

        Time complexity: O(M)  
        - M is the number of activities

        Time complexity analysis:  
        - Each activity is processed once.

        Space complexity: O(M)  
        - M is the number of activities

        Space complexity analysis:  
        - O(M) space is used to store the edges.
        """
        # Create edges from activity nodes to the sink.
        return [(activity_offset + i, sink, self.places[i]) for i in range(num_activities)]


def assign(preferences: list, places: list):
    """
    Function description:
    Assigns participants to activities based on their preferences and available places in a flow network model.
    This function builds a flow network from the participants' preferences and the activity capacities by first retrieving all the
    network edges through the instatiation of a network object with the preferences and places as inputs. One the edges are constructed,
    a graph object is instatiated with these network edges. As part of the instatiation of the graph object with the network's edges,
    the fordfulkerson method is run on this network and the optimal assignment of participants to activities is calculated.

    Input:
    - preferences: List of lists, where each inner list represents the preferences of a participant 
      for different activities.
    - places: List of integers representing the maximum capacity of participants that each activity can accommodate.

    Output:
    - A list of lists where each inner list contains the indices of participants assigned to the corresponding activity. 
      If all participants are assigned to activities according to their preferences and the available places, 
      the result will include all assigned participants. If not all participants can be assigned, the function returns None.

    Time complexity: O(N^3)
    - N is the number of participants.
    
    Time complexity analysis:
    - The construction of the flow network and the execution of the max flow algorithm dominate the time complexity. 
    - The Network class's initialisation and edge construction methods collectively contribute O(N^2) complexity.
    - The complexity of running the max flow algorithm which is run in the instatiation of the Graph class is O(F * E).
    - In our case, since there are N participants, and each one must be allocated to an activity, we know that the flow
    will always be N.
    - We also know that the number of edges in the network in the worst case will be O(N * 3M) which can be simplified
    to: O(N^2)
    - Therefore, the compelxity of running the FordFulkerson method in the graph class will be O(N^3).
    - If we sum the complexity of th edge construction and the max flow traversal, the complexity is O(N^2 + N^3).
    - Since N^3 is the dominating term here, the total time complexity becomes O(N^3)
      
    Space complexity: O(N^2)
    - N is the number of participants.

    Space complexity analysis:
    - The method utilises space to store the network's edges and the results of participant assignments. The maximum 
    space required is the space for the network edges which is N^2.
    """
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

import string

class TrieNode:
    """  
    Class description:  
    Represents a node in the Trie data structure. Each node stores references to its children, metadata about 
    words that end at the node, and maintains ranked lists of the top words passing through it.
    """

    def __init__(self):
        """
        Function description:  
        Initialises a new TrieNode with children slots, word metadata, and ranking information.

        Input: None  
        Output: None  

        Time complexity: O(1)  
        Time complexity analysis:  
        - The function initialises a fixed number of attributes and does not depend on input size.

        Space complexity: O(1)  
        Space complexity analysis:  
        - Each node requires a fixed amount of memory space, regardless of the input.
        """
        # Each node has 62 children slots (26 lowercase + 26 uppercase + 10 digits)
        self.children = [None] * 62
        self.is_end_of_word = False  # Marks the end of a valid word
        self.frequency = 0  # Tracks how often the word appears
        self.ascii_sum = 0  # Stores the sum of ASCII values of the word
        self.word = None  # Holds the Word object at the node
        self.word_ranks = []  # Stores top 3 words ranked by frequency and ASCII sum

class Word:
    """  
    Class description:  
    Stores information about a word, including its frequency and ASCII sum value.
    """
    def __init__(self, word: str, frequency: int, ascii_val: int):
        """
        Function description:  
        Initialises a new Word object with the provided word, frequency, and ASCII value.

        Input:  
            word (str): The word being stored.  
            frequency (int): Frequency of the word.  
            ascii_val (int): Sum of the ASCII values of the word's characters.

        Output: None  

        Time complexity: O(1)  
        Space complexity: O(1)  
        """
        # Stores the word, its frequency, and its ASCII sum value
        self.word = word
        self.frequency = frequency
        self.ascii_val = ascii_val

class Trie:
    """  
    Class description:  
    Implements a Trie data structure to efficiently store and search words with ranking by frequency and ASCII values.
    """
    def __init__(self):
        """
        Function description:  
        Initialises the Trie with a root node and a valid set of characters (alphanumeric).

        Input: None  
        Output: None  

        Time complexity: O(1)  
        Space complexity: O(1)  
        """
        # Initialise root node and valid character set
        self.root = TrieNode()
        self.chars = string.ascii_lowercase + string.ascii_uppercase + '0123456789'

    def insert(self, word: str):
        """
        Function description:  
        Inserts a word into the Trie, updating the frequency, ASCII sum, and word rankings for nodes traversed. After inserting a word
        into the trie, it checks whether the word's final letter node has its own word object. It then creates one if it doesn't exist 
        and if it does, it updates the frequency value of the existing word object. After this, the method traverses all the nodes in
        the nodes_explored list (all letter nodes of inserted word), and as it traverses this list, it updates the word_ranks attribute
        of each letter TrieNode so that the top 3 words at each node of the inserted word can be updated dynamically.

        Input:  
            word (str): The word to insert into the Trie.

        Output: None  

        Time complexity: O(N)  
        - N is the length of the word.

        Time complexity analysis:  
        - The function first iterates over each character in the word and adds a new TrieNode in the Trie for each new character. O(N) 
        - When the end of the word is reached, either a Word object is created for a new word or the frequency of the current word
        object at that node is created. O(1)
        - After the word is inserted into the Trie, the nodes_explored list is traversed N times and for each node, its word_ranks
        attribute is updated by appending the current nodes word to the list and then re-sorting this list based on the word's freq
        and ascii value. 
        - The nodes_explored traversal takes O(N) time and the sorting of the word_ranks list takes in the worst case O(4log(4)) time.
        O(4log4) can be simplified to constant time O(1) because 4 is constant.
        - Therefore, the total worst case complexity of this function is O(N + N). which simplifies to O(N)

        Space complexity: O(N)  
        - Each new node created for the word increases the space used.

        Space complexity analysis:
        - For each new node, a max of N new TrieNodes are created. O(N)
        - For each end of word node, a new Word object is created. O(1)
        - Therefore, space complexity is o(N)

        """
        current_node = self.root  # Start from the root
        ascii_sum = 0  # Track ASCII sum of the word
        nodes_explored = []  # Keep track of nodes visited along the way

        # Traverse the trie, inserting characters
        for char in word:
            ascii_sum += ord(char)  # Accumulate ASCII sum
            index = self.chars.index(char)  # Find character index in valid chars

            # Create a new node if the current character path doesn't exist
            if current_node.children[index] is None:
                current_node.children[index] = TrieNode()

            current_node = current_node.children[index]  # Move to the next node
            nodes_explored.append(current_node)  # Store the visited node

        # Mark the word as complete and update frequency and ASCII sum
        if not current_node.is_end_of_word:
            current_node.is_end_of_word = True
            current_node.frequency = 1  # Initialise frequency for a new word
            current_node.ascii_sum = ascii_sum
            current_node.word = Word(word, 1, ascii_sum)  # Create Word object
        else:
            current_node.frequency += 1  # Increment frequency if word exists
            current_node.word.frequency += 1

        # Update word ranks for all nodes explored
        for node in nodes_explored:
            if node.word_ranks:
                if current_node.word not in node.word_ranks:
                    node.word_ranks.append(current_node.word)  # Add word to ranks
                    node.word_ranks.sort(key=lambda w: (-w.frequency, w.ascii_val))  # Sort by frequency, then ASCII
                    node.word_ranks = node.word_ranks[:3]  # Keep top 3 words
            else:
                node.word_ranks.append(current_node.word)  # Add first word to ranks

    def search(self, input_word: int):
        """
        Function description:  
        Searches for words matching a given input_word and returns top-ranked candidates. Traverses each valid node starting from the
        deepest one (most chars in common with input_word) and returns a list of words in ranked order. This method leverages the 
        already organised insertion of the words in the trie. Because each TrieNode has a list of the top ranked words, all this method
        must do is get the top ranked words form the deepest node/nodes until it finds 3 words that contain the same prefix as the one 
        in the query.

        Input:  
            input_word (str): The input_word to search for.

        Output:  
            List of top 3 words (list[str]) matching the input_word.

        Time complexity: O(M)  
        - M is the length of the input word.

        Time complexity analysis:
        - First, this function traverses all the chars in the input input_word O(M) and puts all these letter's nodes in the 
        nodes_of_input_word list. 
        - If a letter node in the sequence of nodes in the input_word is invlaid, the traversal breaks out and the nodes_of_input_word
        list only contains the nodes up to the point of the input char.
        - These operations are done in O(M) time where M is the length of the input word.
        - If there aren't any letter nodes in the nodes_of_input_word list, the function doesn't traverse this list and therefore
        simply returns an empty list in O(M) time.
        - If there are letter nodes in the nodes_of_input_word list, and the candidates list has less than 3 candidates then the 
        end node of this list is popped off.
        - At this node, the word_ranks list is traversed in max O(3) time and the top words from this list are appended to the 
        canditates list if they are not already in there in max O(3) time.
        - If the list reaches capacity after appending candidates, the loop is broken and the list is returned.
        - In the worst case, there are only traverses 3 of the deepest nodes appending 1 word to the candidate list at a time O(3).
        - Therefore, the worst case time complexity is O(2M + 6) which simplifies to O(M)


        Space complexity: O(M)  
        - Stores references to nodes and candidate words during the search.

        Space complexity analysis: 
        - Space required for storing nodes in nodes_of_input_word: O(M)  
        - Space required for storing candidate words: O(1)  
        - Overall: O(M)
        """
        node = self.root  # Start from the root
        nodes_of_input_word = []  # Keep track of nodes matching the input_word
        candidates = []  # Store candidate words

        # Traverse the trie along the input_word path
        for char in input_word:
            index = self.chars.index(char)  # Find index of the character
            if node.children[index] is None:  # If path breaks, stop
                break
            node = node.children[index]  # Move to the next node
            nodes_of_input_word.append(node)  # Store matching node

        # Collect top words from nodes in reverse order (closer to input_word)
        while nodes_of_input_word and len(candidates) < 3:
            node = nodes_of_input_word.pop()
            for word in node.word_ranks:
                if word.word not in candidates:
                    candidates.append(word.word)  # Add unique candidate words
                if len(candidates) >= 3:
                    break  # Exit the inner loop


        # If the input_word already exits in the trie, return an empty list
        if input_word in candidates:
            return []

        return candidates  # Return top candidates

class SpellChecker:
    """  
    Class description:  
    Implements a spell checker using a Trie to store words from a file and provide suggestions based on an input word. The primary purpose
    of this class is to extract the valid words from the file and add these to its trie. Once all valid words from the input file have
    been added to the classes trie, the check method then calls on the search method of the Trie class which gives the top 3
    suggestions based on the following criteria:
    - Common prefix length
    - Frequency of word
    - Ascii value
    """
    def __init__(self, filename: str):
        """
        Function description:  
        Initialises the SpellChecker and loads words from the given file using the _preprocess_messages method.

        Input:  
            filename (str): Path to the file containing words to load into the Trie.

        Output: None  

        Time complexity: O(T)  
        - T is the total number of chars in the input file.

        Time complexity analysis:
        - _preprocess_messages method takes O(T) time

        Space complexity: O(T)  
        - T is the total number of chars in the input file.

        Space complexity analysis:
        - _preprocess_messages method has O(T) space complexity
        """
        self.trie = Trie()  # Initialise the trie
        self._preprocess_messages(filename)  # Load words from the file

    def _preprocess_messages(self, filename: str):
        """
        Function description:  
        Reads and extracts the words the file line by line using the _extract_words method and inserts words into the Trie using 
        the insert method of the trie class.

        Input:  
            filename (str): Path to the file.

        Output: None  

        Time complexity: O(T)
        - T is the total number of chars in the input file.
        
        Time complexity analysis:
        - For each line in the file, all chars from the line are being looped over by the _extract_words method.
        - For each line's valid words that have been extracted, these words are looped over and they are inserted into the classes trie
        with the trie's insert method.
        - This insert method has a complexity of O(N) where N is the number of chars of its input word.
        - Therefore, for all the valid extracted words in the input file, each char gets inserted into the trie which takes O(T) time.

        Space complexity: O(T)
        - T is the total number of chars in the input file.

        Space complexity analysis:
        - Words list stores M chars for all words being added into the trie.
        """
        # Read file line by line and insert words into the trie
        with open(filename, "r") as file:
            for line in file:
                words = self._extract_words(line)  # Extract words from the line
                for word in words:
                    self.trie.insert(word)  # Insert each word into the trie

    def _extract_words(self, line: str):
        """
        Function description:  
        Extracts words from a line of text, ignoring non-alphanumeric characters.

        Input:  
            line (str): A line of text.

        Output:  
            List of words (list[str]).

        Time complexity: O(N)  
        - N is the number of chars in the line.

        Time complexity analysis:
        - Iterating through each char in the line and assessing whether it is alphanumeric or not.
        - If it is it appends this char to a temporary word string.
        - When the loop comes across a char that is not alphanumeric in the line, it then appends this built word to the words list and
        resets the temporary word string.

        Space complexity: O(N)  
        - N is the number of chars in the line.

        Space complexity analysis:
        - The storage of N chars in the line has a space complexity of O(N)
        """
        word = ""  # Buffer to build words
        words = []  # Store extracted words
        for char in line:
            if char.isalnum():  # If character is alphanumeric, add to the word
                word += char
            elif word:  # If non-alphanumeric, store the completed word
                words.append(word)
                word = ""  # Reset word buffer
        if word:  # Add the last word if any
            words.append(word)
        return words  # Return list of words

    def check(self, input_word):
        """
        Function description:  
        Searches for suggestions based on the input word prefix using the search method of the trie class.

        Input:  
            input_word (str): The word to check for suggestions.

        Output:  
            List of suggestions (list[str]).

        Time complexity: O(M)  
        - M is the length of the input word.

        Time complexity analysis:
        - The time complexity of the seach method is O(M)

        Space complexity: O(M)  

        Space complexity analysis:
        - The space complexity of the search method is O(M)
        """
        # Search for suggestions based on the input word input_word
        return self.trie.search(input_word)

