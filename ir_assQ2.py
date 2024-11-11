import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from collections import defaultdict, namedtuple
from typing import List, Dict, Set, Tuple
import os
import zipfile

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Define a named tuple to store document metadata (name and content)
Document = namedtuple('Document', ['name', 'content'])

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

# Preprocessing function
def preprocess(text: str) -> List[str]:
    text = text.lower()
    words = nltk.word_tokenize(text)
    words = [word for word in words if word.isalpha()]
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word, pos='v') for word in words]  # Verb lemmatization to handle words like 'searching'
    return words

# Function to build the inverted index and biword index
def build_inverted_index(docs: Dict[int, Document]) -> Tuple[Dict[str, Dict[int, List[int]]], Dict[str, Set[int]]]:
    inverted_index = defaultdict(lambda: defaultdict(list))
    biword_index = defaultdict(set)

    for doc_id, document in docs.items():
        words = preprocess(document.content)
        for pos, word in enumerate(words):
            inverted_index[word][doc_id].append(pos)
            if pos < len(words) - 1:
                biword = f"{word} {words[pos + 1]}"
                biword_index[biword].add(doc_id)

    return inverted_index, biword_index

# Function to handle proximity queries
def proximity_search(word1: str, word2: str, max_distance: int, inverted_index: Dict[str, Dict[int, List[int]]]) -> Set[int]:
    docs_word1 = inverted_index.get(word1, {})
    docs_word2 = inverted_index.get(word2, {})

    result_docs = set()

    for doc_id in docs_word1:
        if doc_id in docs_word2:
            positions_word1 = docs_word1[doc_id]
            positions_word2 = docs_word2[doc_id]

            for pos1 in positions_word1:
                for pos2 in positions_word2:
                    if abs(pos1 - pos2) <= max_distance:
                        result_docs.add(doc_id)
                        break

    return result_docs

# Function to handle Soundex matching
def soundex(name: str) -> str:
    name = name.upper()
    code = {'A': '', 'E': '', 'I': '', 'O': '', 'U': '', 'H': '', 'W': '', 'Y': '',
            'B': '1', 'F': '1', 'P': '1', 'V': '1',
            'C': '2', 'G': '2', 'J': '2', 'K': '2', 'Q': '2', 'S': '2', 'X': '2', 'Z': '2',
            'D': '3', 'T': '3',
            'L': '4',
            'M': '5', 'N': '5',
            'R': '6'}
    result = name[0]
    prev_code = code.get(name[0], '')

    for char in name[1:]:
        new_code = code.get(char, '')
        if new_code and new_code != prev_code:
            result += new_code
        prev_code = new_code

    result = result[:4].ljust(4, '0')
    return result

# Modified soundex_search function to return both document IDs and matching words
def soundex_search_single(name: str, inverted_index: Dict[str, Dict[int, List[int]]]) -> Set[int]:
    name_code = soundex(name)
    result_docs = set()

    for word, docs in inverted_index.items():
        if soundex(word) == name_code:
            result_docs.update(docs.keys())

    return result_docs

# Function to handle Soundex search with Boolean operators (AND, OR, NOT)
def soundex_boolean_search(query: str, inverted_index: Dict[str, Dict[int, List[int]]]) -> Set[int]:
    query = query.lower()
    words = query.split()

    # Separate operators and words
    terms = []
    operators = []
    i = 0
    while i < len(words):
        if words[i] in {"and", "or", "not"}:
            operators.append(words[i])
        else:
            terms.append(words[i])
        i += 1

    # Evaluate the first term
    if terms:
        result_docs = soundex_search_single(terms[0], inverted_index)
    else:
        return set()  # No terms to search

    # Process the rest of the terms and operators
    for idx, operator in enumerate(operators):
        next_term_docs = soundex_search_single(terms[idx + 1], inverted_index)

        if operator == "and":
            result_docs &= next_term_docs  # Intersection of sets for AND
        elif operator == "or":
            result_docs |= next_term_docs  # Union of sets for OR
        elif operator == "not":
            result_docs -= next_term_docs  # Difference of sets for NOT

    return result_docs

# Function to handle biword queries with sequence validation
def biword_search(phrase: str, biword_index: Dict[str, Set[int]], docs: Dict[int, Document]) -> Dict[int, str]:
    # Preprocess the input phrase
    words = preprocess(phrase)

    if len(words) < 2:
        print("Phrase is too short for biword search.")
        return {}

    # Construct biwords from the preprocessed words
    biwords = [f"{words[i]} {words[i + 1]}" for i in range(len(words) - 1)]

    # Find the initial set of documents matching the first biword
    result_docs = set(biword_index.get(biwords[0], set()))

    for biword in biwords[1:]:
        result_docs &= biword_index.get(biword, set())

    # Filter results to ensure all biwords appear in the exact order in the document
    valid_docs = {}
    for doc_id in result_docs:
        doc_content = preprocess(docs[doc_id].content)

        doc_biwords = [f"{doc_content[i]} {doc_content[i + 1]}" for i in range(len(doc_content) - 1)]

        # Check if biwords appear in sequence in the document
        index = 0
        for biword in biwords:
            try:
                index = doc_biwords.index(biword, index)
            except ValueError:
                break
        else:
            valid_docs[doc_id] = docs[doc_id].name

    return valid_docs

# Extract the Corpus.zip file and load the documents
def load_documents(zip_path: str, extract_to: str) -> Dict[int, Document]:
    if not os.path.exists(extract_to):
        os.makedirs(extract_to)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

    # Verify extraction
    extracted_files = []
    for root, dirs, files in os.walk(extract_to):
        for file in files:
            extracted_files.append(os.path.join(root, file))

    # Load the documents
    docs = {}
    for i, file_path in enumerate(extracted_files):
        if file_path.endswith('.txt'):
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read().strip()
                if content:  # Only add non-empty files
                    docs[i + 1] = Document(name=os.path.basename(file_path), content=content)

    return docs

def display_document(doc_id: int, docs: Dict[int, Document], file_id_mapping: Dict[int, str]):
    doc = docs[doc_id]
    lines = doc.content.split('\n')
    first_two_lines = '\n'.join(lines[:2])
    
    # Get the file ID for the document
    file_id = file_id_mapping.get(doc_id, None)
    if file_id:
        # Construct the Google Drive link for the individual file
        link = f"https://drive.google.com/file/d/{file_id}/view?usp=sharing"
    else:
        link = "Link not available"
    
    print(f"  Document ID: {doc_id}")
    print(f"  Document Name: {doc.name}")
    print(f"  Content Preview:\n{first_two_lines}")
    print(f"  Link to Document: {link}")
    print("-" * 50)


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


def main():
    # Load documents from the Corpus.zip
    zip_path = 'Corpus.zip'
    extract_to = 'Corpus'
    docs = load_documents(zip_path, extract_to)

    # Build inverted index and biword index
    inverted_index, biword_index = build_inverted_index(docs)

    


    while True:
        # User input for selecting the search type
        print("Search Options:")
        print("1. Biword Search")
        print("2. Proximity Search")
        print("3. Soundex  Search")
        print("4. Soundex Boolean Search")
        print("5. Exit")
        search_type = input("Enter your choice (1-5): ")

        if search_type == '1':
            print("Biword Search")
            phrase = input("Enter the phrase to search: ")
            result_docs = biword_search(phrase, biword_index, docs)
            print(f"Biword search for phrase '{phrase}':")
            if result_docs:
                for doc_id in result_docs:
                    display_document(doc_id, docs, file_id_mapping)  # Pass file_id_mapping dictionary here
            else:
                print("  No results found.")

        elif search_type == '2':
            print("Proximity Search")
            word1 = input("Enter the first word: ")
            word2 = input("Enter the second word: ")
            max_distance = int(input("Enter the maximum distance: "))
            result_docs = proximity_search(word1, word2, max_distance, inverted_index)
            print(f"Proximity search for words '{word1}' and '{word2}' with max distance {max_distance}:")
            if result_docs:
                for doc_id in result_docs:
                    display_document(doc_id, docs, file_id_mapping)  # Pass file_id_mapping dictionary here
            else:
                print("  No results found.")

        elif search_type == '3':
            print("Soundex Search")
            name = input("Enter the name to search: ")
            result_docs = soundex_search_single(name, inverted_index)
            print(f"Soundex search for name '{name}':")
            if result_docs:
                for doc_id in result_docs:
                    display_document(doc_id, docs, file_id_mapping)  # Pass file_id_mapping dictionary here
            else:
                print("  No results found.")

        elif search_type == '4':
            print("Soundex Boolean Search (AND, OR, NOT)")
            query = input("Enter the Boolean query: ")
            result_docs = soundex_boolean_search(query, inverted_index)
            print(f"Soundex Boolean search for query '{query}':")
            if result_docs:
                for doc_id in result_docs:
                    display_document(doc_id, docs, file_id_mapping)  # Pass file_id_mapping dictionary here
        
        elif search_type == '5':
            print("Exited.")
            break

        else:
            print("Invalid choice. Please enter a number between 1 and 5.")


      

        cont = input("Do you want to perform another search? (y/n): ")
        if cont.lower() != 'y':
            break


if __name__ == "__main__":
    main()