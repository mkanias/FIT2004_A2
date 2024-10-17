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

# Example usage
if __name__ == "__main__":
    filename = "src/text_files_q2/test_case_2.txt"  # Input file with words

    spellchecker = SpellChecker(filename)  # Initialise the spell checker
    print(spellchecker.check('INDKNJJNINJDNAK'))  # Check for suggestions based on input
