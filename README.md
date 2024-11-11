This project implements a Boolean retrieval model using an inverted index, supporting complex search queries with logical operators (AND, OR, NOT). Additionally, it extends the model to support phrase queries, proximity searches, and Soundex-based spelling matching.

Project Overview
The goal of this project is to develop an efficient document retrieval system that can handle Boolean and extended Boolean queries on a document collection. The system indexes documents after preprocessing, allowing for precise and flexible search capabilities.

Features
1. Boolean Retrieval Model with Inverted Index
Preprocessing:Each document in the collection undergoes preprocessing steps to normalize and clean the text:
Case Folding: Convert all words to lowercase.
Tokenization: Split text into individual searchable units.
Non-alphabetic Token Removal: Remove numbers, punctuation, and symbols.
Stop Word Removal: Remove common stop words to reduce noise.
Lemmatization: Reduce words to their base forms.

Inverted Index Construction
An inverted index maps each word to the list of documents in which it appears, enabling efficient retrieval based on word occurrences.

Boolean Query Handling
Supports Boolean operators to refine search results:

AND: Returns documents containing all specified terms.
OR: Returns documents containing at least one of the specified terms.
NOT: Excludes documents containing a specified term.
Example Queries
Query: "technology AND phone"
Result: Retrieves documents containing both "technology" and "phone."

Query: "deliveries OR foods"
Result: Retrieves documents containing either "deliveries" or "foods."

2. Extended Boolean Retrieval Model
This model enhances the basic Boolean retrieval with additional search capabilities:

a) Phrase Queries with Biword Index
Allows users to search for exact phrases by building an index of consecutive word pairs (biwords) in each document.

Example: For the query "search engine", retrieves documents where "search" and "engine" appear in sequence.
b) Proximity Queries with Positional Index
Supports queries that specify the maximum distance between words in the same document.

Example: For the query "easy and passenger" with a maximum distance of 10, retrieves documents where "easy" and "passenger" appear within 10 words of each other.
c) Soundex Algorithm for Phonetic Matching
Implements the Soundex algorithm to match words based on pronunciation, handling spelling variations or typos.

Example: For the name "Gogle", retrieves documents with phonetically similar words like "Google."
d) Soundex Search with Boolean Operators
Combines Soundex-based matching with Boolean operators for complex phonetic search queries.

Example: For the query "lehri AND stainford", retrieves documents containing phonetically similar terms to "lehri" and "stainford."

Output:
Boolean Retrieval Model: Returns filenames of matching documents.
Extended Boolean Retrieval Model: Outputs relevant documents based on phrase, proximity, and Soundex-based searches.

Conclusion
The retrieval models developed in this project allow for flexible, precise document searches. The inverted index and additional indexing techniques ensure efficient retrieval for both Boolean and extended Boolean queries.

