import string

class TrieNode:
    def __init__(self):
        # Each node has 62 children slots (26 lowercase + 26 uppercase + 10 digits)
        self.children = [None] * 62
        self.is_end_of_word = False  # Marks the end of a valid word
        self.frequency = 0  # Tracks how often the word appears
        self.ascii_sum = 0  # Stores the sum of ASCII values of the word
        self.word = None  # Holds the Word object at the node
        self.word_ranks = []  # Stores top 3 words ranked by frequency and ASCII sum

class Word:
    def __init__(self, word, frequency, ascii_val):
        # Stores the word, its frequency, and its ASCII sum value
        self.word = word
        self.frequency = frequency
        self.ascii_val = ascii_val

class Trie:
    def __init__(self):
        # Initialize root node and valid character set
        self.root = TrieNode()
        self.chars = string.ascii_lowercase + string.ascii_uppercase + '0123456789'

    def insert(self, word):
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
            current_node.frequency = 1  # Initialize frequency for a new word
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

    def search_with_prefix(self, prefix):
        node = self.root  # Start from the root
        nodes_of_prefix = []  # Keep track of nodes matching the prefix
        candidates = []  # Store candidate words

        # Traverse the trie along the prefix path
        for char in prefix:
            index = self.chars.index(char)  # Find index of the character
            if node.children[index] is None:  # If path breaks, stop
                break
            node = node.children[index]  # Move to the next node
            nodes_of_prefix.append(node)  # Store matching node

        # Collect top words from nodes in reverse order (closer to prefix)
        while nodes_of_prefix:
            node = nodes_of_prefix.pop()
            for word in node.word_ranks:
                if len(candidates) < 3 and word.word not in candidates:
                    candidates.append(word.word)  # Add unique candidate words

        # If the prefix is a valid word, return an empty list
        if prefix in candidates:
            return []

        return candidates  # Return top candidates

class SpellChecker:
    def __init__(self, filename):
        self.trie = Trie()  # Initialize the trie
        self._preprocess_messages(filename)  # Load words from the file

    def _preprocess_messages(self, filename):
        # Read file line by line and insert words into the trie
        with open(filename, "r") as file:
            for line in file:
                words = self._extract_words(line)  # Extract words from the line
                for word in words:
                    self.trie.insert(word)  # Insert each word into the trie

    def _extract_words(self, line):
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
        # Search for suggestions based on the input word prefix
        return self.trie.search_with_prefix(input_word)

# Example usage
if __name__ == "__main__":
    filename = "src/text_files_q2/test_case_2.txt"  # Input file with words

    spellchecker = SpellChecker(filename)  # Initialize the spell checker
    print(spellchecker.check('IDJM'))  # Check for suggestions based on input
