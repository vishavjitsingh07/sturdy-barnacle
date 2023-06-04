from google.cloud import storage
from collections import defaultdict
from nltk.corpus import stopwords as sw
from firebase_admin import credentials, db
import random, string, pickle, firebase_admin
from config import gcd_credentials, BUCKET_NAME


def gcdBucket():
    cred = credentials.credentials(gcd_credentials)
    firebase_admin = firebase_admin.initialize_app(cred)
    storage_client = storage.Client.from_service_account_json(gcd_credentials)
    bucket = storage_client.get_bucket(BUCKET_NAME)
    return bucket


def generate_random_string():
    characters = string.digits
    random_string = (
        "".join(random.choice(characters) for _ in range(10))
        + "_"
        + "".join(random.choice(characters) for _ in range(10))
        + ".jpg"
    )
    return random_string


class TrieNode:
    def __init__(self):
        self.children = defaultdict(TrieNode)
        self.is_word = False
        self.count = 0


class TrieNode:
    def __init__(self):
        self.children = defaultdict(TrieNode)
        self.is_word = False
        self.count = 0


class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        node = self.root
        for char in word:
            node = node.children[char]
        node.is_word = True

    def search(self, word):
        node = self.root
        for char in word:
            if char in node.children:
                node = node.children[char]
            else:
                return None
        return node


def get_matching_strings(
    main_string,
    min_keyword_matches,
    target_strings=pickle.load(open("mapping.pkl", "rb")),
):
    trie = Trie()
    stopwords = set(sw.words("english"))  # List of stopwords to ignore
    stopwords.add("startseq")
    stopwords.add("endseq")
    stopwords.add("start")
    stopwords.add("end")
    stopwords.add("<start>")
    stopwords.add("<end>")
    # Preprocess main string
    main_string_words = [
        word.lower() for word in main_string.split() if word.lower() not in stopwords
    ]

    # Insert target strings into Trie
    for target_id, target_string in target_strings.items():
        target_string_words = [
            word.lower()
            for word in target_string.strip().split()
            if word.lower() not in stopwords
        ]
        for word in target_string_words:
            trie.insert(word)

    # Count keyword matches for target strings
    matches = []
    for target_id, target_string in target_strings.items():
        target_string_words = [
            word.lower()
            for word in target_string.split()
            if word.lower() not in stopwords
        ]
        count = sum(word in main_string_words for word in target_string_words)
        #         print(count, target_string_words)
        #         count = 1000
        if count >= min_keyword_matches:
            node = trie.search(target_string_words[0])
            if node and node.is_word:
                node.count += count
                matches.append((target_id, node.count))

    matches.sort(key=lambda x: x[1], reverse=True)
    sorted_ids = [match[0] for match in matches]

    return sorted_ids if len(sorted_ids) < 10 else sorted_ids[10:]
