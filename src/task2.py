import string

class TrieNode:
    def __init__(self):
        # Initialize an array for 62 characters (a-z, A-Z, 0-9)
        self.children = [None] * 62  # 26 lowercase + 26 uppercase + 10 digits
        self.is_end_of_word = False
        self.frequency = 0  # To keep track of word frequency
        self.depth = 0
        self.ascii_sum = 0
        self.suggested = False  # Flag to track if the word has been suggested
        self.sorted_children = None
        self.word = None
        self.word_ranks = []


class Word:
    def __init__(self, word, frequency, ascii_val) -> None:
        self.word = word
        self.frequency = frequency
        self.ascii_val = ascii_val

class Trie:
    def __init__(self):
        self.root = TrieNode()
        self.chars = string.ascii_lowercase + string.ascii_uppercase + '0123456789' # String with all valid chars and ints
        self.common_prefix_len = 0


    def insert(self, word):
        current_node = self.root
        ascii_sum = 0
        nodes_explored = []

        for char in word:
            ascii_sum += ord(char)
            index = self.chars.index(char)

            if current_node.children[index] is None:
                new_node = TrieNode()
                new_node.depth = current_node.depth + 1
                current_node.children[index] = new_node  # Create new node if it doesn't exist
                nodes_explored.append(new_node)

            current_node = current_node.children[index]
            
        # Increment frequency and mark the end of the word
        if not current_node.is_end_of_word:
            current_node.is_end_of_word = True
            current_node.frequency = 1  # Initialize frequency to 1 for a new word
            current_node.ascii_sum = ascii_sum
        else:
            current_node.frequency += 1  # Increment frequency if the word already exists
        
        if not current_node.word:
            current_node.word = Word(word, current_node.frequency, current_node.ascii_sum)
        
        for node in nodes_explored:
            if node.word_ranks: # [Words]
                if current_node.word not in node.word_ranks:
                    if len(node.word_ranks) < 3:
                        node.word_ranks.append(current_node.word)
                        node.word_ranks.sort(key=lambda x: (-x.frequency, x.ascii_val))

                    

            else:
                node.word_ranks.append(current_node.word)
            
        # Sort children by frequency and ASCII value after inserting
        # self.sort_children(current_node)

    def sort_children(self, node: TrieNode):
        node.sorted_children = sorted(
            [child for child in node.children if child is not None],
            key=lambda x: (-x.frequency, x.ascii_sum)
        )

    def search(self, input_word):
        candidates = []
        node = self.root

        # Step 1: Traversing to the node matching the input prefix
        for char in input_word:
            index = self.chars.index(char)
            if node.children[index] is None:
                break # Prefix doesnt exist in the trie
            node = node.children[index]

        # Step 2: If prefix matches, collect words from this node
        if node.is_end_of_word:
            return [] # The input word matches an existing word
        
        # Step 3: Collect candidates using DFS with dynamic ranking
        self.collect_top_words(node, input_word, candidates)

        # Step 4: Return the top 3 candidates
        return [word for word, _, _ in candidates]

    def collect_top_words(self, node: TrieNode, prefix: str, candidates: list, max_results=3):
        if node.is_end_of_word:
            # Add the word to candidates with the freq and ascii val
            candidates.append((prefix, node.frequency, node.ascii_sum))
            if len(candidates) >= max_results:
                return # Coz we already got top 3 candidates

        # Recursively explore the children in prioriy ranking order 
        for i, child in enumerate(node.children):
            if child is not None and len(candidates) < max_results:
                new_prefix = prefix + self.index_to_char(i)
                self.collect_top_words(child, new_prefix, candidates)

    def index_to_char(self, index):
        """Converts an index to its corresponding character (a-z, A-Z, 0-9)."""
        if 0 <= index < 26:  # a-z
            return chr(index + ord('a'))
        elif 26 <= index < 52:  # A-Z
            return chr(index - 26 + ord('A'))
        elif 52 <= index < 62:  # 0-9
            return chr(index - 52 + ord('0'))





    # def search_with_prefix(self, prefix, max_results=3):
    #     candidates = []
    #     iterations = 0
    #     for i in range(len(prefix), 0, -1):
    #         self.common_prefix_len = i
    #         iterations += 1
    #         shorter_prefix = prefix[:i]
    #         print(shorter_prefix)
    #         self._collect_words_with_prefix(self.root, shorter_prefix, candidates, max_results, iterations)
    #         print(candidates)
    #         if len(candidates) >= max_results:
    #             break  # Stop if we have enough results

    #     # Sort candidates by common prefix, frequency, and ASCII value
    #     return sorted(
    #         candidates,
    #         key=lambda x: (-x[2], -x[1], x[3])
    #     )

    # def _collect_words_with_prefix(self, node, prefix, candidates, max_results, iterations):
    #     # Traverse to the node matching the prefix
    #     for char in prefix:
    #         index = self.chars.index(char)
    #         if node.children[index] is None:
    #             return  # No words with the given prefix
    #         node = node.children[index]

    #     if iterations == 1 and node.is_end_of_word:
    #         return []
    #     # Perform DFS to collect words from the current node
    #     self._dfs(node, prefix, candidates, max_results)

    # def _dfs(self, node, current_prefix, candidates, max_results):
    #     if node.is_end_of_word and not node.suggested:
    #         print(current_prefix)
    #         node.num_chars_in_common_prefix = self.common_prefix_len
    #         word_info = [
    #             current_prefix,  # Word
    #             node.frequency,  # Frequency
    #             node.num_chars_in_common_prefix,  # Common prefix length
    #             node.ascii,  # ASCII value
    #         ]
    #         candidates.append(word_info)
    #         node.suggested = True  # Mark as suggested

    #     # Recur for all children
    #     for i, child in enumerate(node.children):
    #         if child and len(candidates) < max_results:
    #             char = self.chars[i]
    #             self._dfs(child, current_prefix + char, candidates, max_results)

    # def _longest_common_prefix_length(self, word, prefix):
    #     common_length = 0
    #     for w_char, p_char in zip(word, prefix):
    #         if w_char == p_char:
    #             common_length += 1
    #         else:
    #             break  # Stop counting once there's a mismatch
    #     return common_length
    
    # def _find_ascii(self, word):
    #     ascii_total = 0
    #     for char in word:
    #         ascii_total += ord(char)
    #     return ascii_total


class SpellChecker:
    def __init__(self, filename):
        self.trie = Trie()
        self._preprocess_messages(filename)

    def _preprocess_messages(self, filename):
        with open(filename, "r") as file:
            for line in file:
                words = self._extract_words(line)
                for word in words:
                    self.trie.insert(word)

    def _extract_words(self, line):
        # Extract words from the line by filtering out non-alphanumeric characters
        word = ""
        words = []
        for char in line:
            if char.isalnum():  # Check if character is alphanumeric
                word += char
            else:
                if word:  # If we have a word, add it to the list
                    words.append(word)
                    word = ""  # Reset for the next word
        if word:  # Add the last word if it exists
            words.append(word)
        return words

    def check(self, input_word):
        all_candidates = []  # Accumulate candidates here
        
        # # Try progressively shorter prefixes
        # for i in range(len(input_word), 0, -1):
        #     shorter_prefix = input_word[:i]
        #     candidates = self.trie.search_with_prefix(shorter_prefix)

        #     # Add unique candidates to the list
        #     for word_info in candidates:
        #         word, frequency, _, ascii_val, node = word_info  # Assuming word_info includes the TrieNode

        #         # Check if the word has been suggested already
        #         # print(word, node.suggested, all_candidates)
        #         if not node.suggested:  # Use the TrieNode to check if suggested
        #             all_candidates.append(word)
        #             node.suggested = True  # Mark this word as suggested

        # # If fewer than 3 candidates are found, return all available candidates
        # result = all_candidates[:3]
        # if input_word in result:
        #     return []
        result = self.trie.search(input_word)
        return result



# Example usage:
if __name__ == "__main__":
    filename = "src/text_files_q2/test_case_2.txt"

    
    spellchecker = SpellChecker(filename)
    print(spellchecker.check('IDc'))



    