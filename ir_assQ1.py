import os
import zipfile
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import defaultdict
from typing import List, Dict, Set

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Preprocessing function
def preprocess(text: str) -> List[str]:
    # Case folding
    text = text.lower()

    # Tokenization
    words = nltk.word_tokenize(text)

    # Remove non-alphabetic tokens and normalize
    words = [word for word in words if word.isalpha()]

    # Stopword removal
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]

    return words

# Function to build the inverted index
def build_inverted_index(docs: Dict[int, str]) -> Dict[str, Set[int]]:
    inverted_index = defaultdict(set)

    for doc_id, content in docs.items():
        words = preprocess(content)
        for word in words:
            inverted_index[word].add(doc_id)

    return inverted_index

# Function to handle boolean search queries
def boolean_search(query: str, inverted_index: Dict[str, Set[int]]) -> Set[int]:
    # Tokenize the query
    tokens = query.split()

    # Stack to manage the boolean operations
    stack = []

    i = 0
    while i < len(tokens):
        token = tokens[i].upper()

        if token == "AND":
            operand1 = stack.pop()
            i += 1
            operand2 = get_docs(tokens[i], inverted_index)
            stack.append(operand1.intersection(operand2))

        elif token == "OR":
            operand1 = stack.pop()
            i += 1
            operand2 = get_docs(tokens[i], inverted_index)
            stack.append(operand1.union(operand2))

        elif token == "NOT":
            i += 1
            operand = get_docs(tokens[i], inverted_index)
            if stack:
                stack[-1] = stack[-1].difference(operand)
            else:
                # In case NOT is the first operation, use all document IDs minus the NOT set
                all_docs = set(range(1, len(inverted_index) + 1))
                stack.append(all_docs.difference(operand))

        else:
            # First operand (no operator before it)
            stack.append(get_docs(token, inverted_index))

        i += 1

    return stack.pop() if stack else set()

# Helper function to get documents for a token
def get_docs(token: str, inverted_index: Dict[str, Set[int]]) -> Set[int]:
    # Preprocess the token (case folding, stop word removal, lemmatization)
    processed_token = preprocess(token)[0] if preprocess(token) else ""
    return inverted_index.get(processed_token, set())

# Function to display document preview and link
def display_document(doc_id: int, docs: Dict[int, str], file_to_doc_id: Dict[int, str], file_id_mapping: Dict[int, str]):
    doc_content = docs[doc_id]
    lines = doc_content.split('\n')
    first_two_lines = '\n'.join(lines[:2])

    # Get the file ID and document name for the document
    file_name = file_to_doc_id.get(doc_id, "Unknown Document")
    file_id = file_id_mapping.get(doc_id, None)
    if file_id:
        # Construct the Google Drive link for the individual file
        link = f"https://drive.google.com/file/d/{file_id}/view?usp=sharing"
    else:
        link = "Link not available"

    print(f"  Document Name: {file_name}")
    print(f"  Content Preview:\n{first_two_lines}")
    print(f"  Link to Document: {link}")
    print("-" * 50)

# Main function to run the program
def main():
    # Extract documents from Corpus.zip
    corpus_zip_path = 'Corpus.zip'
    corpus_dir = 'Corpus'

    if not os.path.exists(corpus_dir):
        os.makedirs(corpus_dir)

    with zipfile.ZipFile(corpus_zip_path, 'r') as zip_ref:
        zip_ref.extractall(corpus_dir)

    # Iterate inside the 'Corpus' directory
    docs_dir = os.path.join(corpus_dir, 'Corpus')
    docs = {}
    file_to_doc_id = {}  # Dictionary to map document IDs to filenames

    if os.path.exists(docs_dir):
        for i, filename in enumerate(os.listdir(docs_dir)):
            if filename.endswith('.txt'):
                file_path = os.path.join(docs_dir, filename)
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                    docs[i + 1] = content  # Using i+1 as document ID to start from 1
                    file_to_doc_id[i + 1] = filename  # Map document ID to filename
    else:
        print(f"Directory {docs_dir} does not exist.")
        return

    if not docs:
        print("No documents loaded. Please check the files and their content.")
        return

    # Build the inverted index
    inverted_index = build_inverted_index(docs)

    # Example mapping for file IDs
    file_id_mapping = {
        1: '17gtGk9isJIjMOwJUhrvScWIkBeIivQvk',
        2: '1Bi7UM3yJfro4Tu3-g3fm5Y2XZ8YeZHg5',
        3: '1RDdQqOcbVrmzPYIDQ6R1t7QXWrvzI68P',
        4: '1PRdJlT5wixJd_FTe3moFqTlHbanAo_5_',
        5: '1EEjfP9vyh8uzq3-N-TF95i_T91ZOF1On',
        6: '1-zWgMQAwSOCsmQGYzkyAINNZdGZaG_9N',
        7: '1iiaWgK1Ic8jxQCFTDoAJwfEPOCpBcHyb',
        8: '1RHmqiyJN3dsr0eLJwZaN_7cqJf44u4mQ',
        9: '10Iz3xS6CMbbkFkwDuENYEKoHrGh47WHZ',
        10: '1i_XUVOQDZOvH21vVDAzPMFe3Db-60jIX',
        11: '1Vyp9dRXbByFNjHUCWjx6IOT5fBSNSPSu',
        12: '1SU3qe7WPduaiOWQiEWKjVgWWNJJG2P8l', #paypal
        13: '1myzc16SQiItwv7Fu34xKPdzhEAg8WMf_', #ig
        14: '1YdVx3j2U1QIUqjApzzu_7Vc0yhwdN4W0', #volkswagen
        15: '1gym8q9fuilqyNOZ_MbdTriQKiF4fs21T', #hp
        16: '1cfhbL9uugppmXFwqCtwpgvpbx7k8iQtC', #spotify
        17: '1uRFepmUklcGoydXm7CNOCbyApWmyJ3zu', #uber
        18: '1u32cxm6JkQXMttqcgpKtOaW9JF_io2HZ', #utube
        19: '1jTtyGhZ5Nt0r5H5P4q9SCMM4EKNocFbq', #motorola
        20: '1L8cqu31UdqfUlSl2BUPZm2WgcA7KWXhW', #yahoo
        21: '1Yf9ypzXtV0GVRzSROA0Rf5lDlzEZdurG', #lenevo
        22: '1q628g8gPoXhQwQophtXpa1EOfLTia6Yr', #dell
        23: '1NXxLhKMWLjFZw-OHi2omCUHneE65MEKy', #levis
        24: '1IKOAeE6R_lYM79KcCg1R69jaZqFESi4T', #whatsapp
        25: '1eJefZuegIKYKtO_Ui6Ft613HjNy4gjNa', #bing
        26: '1UM0HfKDquiYfQUYPM2w8zOhOodJ8mpXB', #hawai
        27: '1W-hLebBkUgVqyvpPxrei8todpzcj2Zsf', #samsung
        28: '1z8tNchwcY5fZpu9X5PTN_brt3TKn8jEA', #swiggy
        29: '1M44nNmnO8tBORZJefBZDpsA4eRsAvK8u', #skype
        30: '1Ow9GrppA1haCplAzpl8sQCwinar6Ynri', #messenger
        31: '1c_wKzP_Kx9Sbu58ewhPSf8MyeX7C0OvK', #tele
        32: '1FcptFYZQ7aLZYXl0kQXp5UluCXzLG-Q9', #steam
        33: '1885pbtntJwW_FkqriJBi_3OP450H-4rA', #reliance
        34: '1RAThPxqywa6QnsfOx7m1Ye12oD-FFb48', #canva
        35: '1vd-v2IEBm3HEtCmQgEzlQTUixZrBFU-g', #puma
        36: '1aeyAcBDBROKq0-oodl8ozOF_lKN0uQlV', #nokia
        37: '1Mg_hDi642we5KCGW6ZL6flSiijWgXxLH', #reddit
        38: '1Lg1DRlNH4JS2SMyysDMN60JyHWsi3Q6T', #sony
        39: '1KWB-AIr2ThSdquUGLfKXf6V-6haWSWwV', #ola
        40: '1gu9NXN9jW4Kr7O8XNaxfdtG7XVt3pG2V', #adobe
        41: '1c9-7OnQuQo9ZuRzPRhSlqaDJUI57Cjbb', #blackberry
    }

    while True:
        # Prompt the user for a query
        query = input("Enter your search query (or 'exit' to quit): ")

        if query.lower() == 'exit':
            break

        result_docs = boolean_search(query, inverted_index)
        print(f"\nQuery: {query}")
        if result_docs:
            print("Matching Documents:")
            for doc_id in result_docs:
                display_document(doc_id, docs, file_to_doc_id, file_id_mapping)
        else:
            print("No matching documents.")
        print("-" * 40)

# Entry point for the program
if __name__ == "__main__":
    main()