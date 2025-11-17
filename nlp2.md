```
# **Algorithm for Information Retrieval using TF–IDF & Cosine Similarity**

### **Step 1: Load Documents**

1. Create a list of text documents.
2. Print each document for verification.

### **Step 2: Import Required Libraries**

3. Load NLTK modules (tokenizer, stopwords, stemmer, lemmatizer).
4. Import TF-IDF vectorizer and cosine similarity from sklearn.

### **Step 3: Download Required NLTK Resources**

5. Check if required NLTK datasets exist.
6. If missing, download them automatically.

### **Step 4: Preprocess Each Document**

For every document:
7. Tokenize the document into words.
8. Convert all tokens to lowercase.
9. Remove punctuation from each token.
10. Remove English stopwords.
11. Apply stemming to reduce words to roots.
12. Apply lemmatization to convert words to their base form.
13. Store the final cleaned tokens.

### **Step 5: Apply Preprocessing on All Documents**

14. Preprocess each document using the preprocessing function.
15. Print original and preprocessed documents for verification.
16. Join token lists back into cleaned text strings.

### **Step 6: Generate TF–IDF Vectors**

17. Initialize a `TfidfVectorizer`.
18. Fit the vectorizer on preprocessed documents.
19. Transform documents into a TF-IDF matrix.
20. Display the shape of the TF-IDF matrix and vocabulary terms.

### **Step 7: Define Search Function**

When a query is given:
21. Preprocess the query using the same preprocessing steps.
22. Convert the query into a TF-IDF vector using the trained vectorizer.
23. Compute cosine similarity between the query vector and document vectors.
24. Rank documents in descending order of similarity score.
25. Prepare the output as tuples containing document index, document text, and similarity score.

### **Step 8: Perform Search**

26. Input a query.
27. Run the search function.
28. Print documents ranked by similarity.

### **Step 9: Test with Multiple Queries**

29. For each example query: compute similarity scores and print ranked matching documents.

### **Step 10: End of Demonstration**

30. Display a completion message.
```

```
# **Algorithm for Probabilistic Context-Free Grammar (PCFG) Parsing**

### **Step 1: Import Libraries**

1. Import PCFG, ViterbiParser, ChartParser, Tree, and Nonterminal from NLTK.

### **Step 2: Define the PCFG Grammar**

2. Write grammar rules with their associated probabilities.
3. Load the grammar using `PCFG.fromstring()`.

### **Step 3: Initialize Parsers**

4. Create a Viterbi parser using the PCFG.
5. Create a Chart parser using the same grammar.

### **Step 4: Input Sentence**

6. Provide an input sentence.
7. Tokenize the sentence by splitting it into words.

### **Step 5: Define Function to Compute Tree Probability**

8. Create a function that walks through a parse tree.
9. For each node, match its production rule with grammar rules.
10. Multiply probabilities of all productions used in the tree.
11. Return the final product as the tree’s total probability.

### **Step 6: Generate Most Probable Parse (Viterbi)**

12. Parse the sentence using the Viterbi parser.
13. Select the highest-probability parse tree returned.
14. Display the parse tree and its probability.

### **Step 7: Generate All Possible Parses (Chart Parser)**

15. Parse the sentence using the Chart parser to get all valid parse trees.
16. For each tree:
    a. Display the parse tree.
    b. Compute its probability using the probability function.
    c. Print the computed probability.

### **Step 8: Compare All Parse Probabilities**

17. Store each tree with its probability.
18. Sort the trees by probability in descending order.
19. Print the sorted list with probabilities.
20. Identify the parse with the highest probability.
21. Indicate that this highest-probability parse is the Viterbi-selected parse.

### **Step 9: End Execution**

22. End the program after displaying all parse trees and probabilities.
```

```
# **Algorithm for Implementing WordNet in NLP**

### **Step 1: Import Libraries**

1. Import `nltk` and download the WordNet corpus.
2. Import `wordnet` from `nltk.corpus`.

### **Step 2: Select a Word for Exploration**

3. Choose the target word (e.g., "great").
4. Print the selected word.

### **Step 3: Retrieve Synsets**

5. Get all synsets of the word using `wordnet.synsets(word)`.
6. For each synset, print its name and definition.

### **Step 4: Extract Synonyms**

7. Create an empty set for storing lemma names.
8. For each synset:
   a. Extract all lemma names.
   b. Add them to the synonyms set.
9. Print all unique synonyms.

### **Step 5: Extract Antonyms**

10. Create an empty antonym set.
11. For each synset:
    a. For each lemma, check if it has antonyms.
    b. Add discovered antonyms to the set.
12. Print antonyms if any exist; otherwise state none found.

### **Step 6: Retrieve Hypernyms**

13. Select the first synset of the word.
14. Retrieve its hypernyms using `.hypernyms()`.
15. Print each hypernym and its definition.

### **Step 7: Retrieve Hyponyms**

16. Retrieve hyponyms of the first synset using `.hyponyms()`.
17. Print each hyponym and its definition.

### **Step 8: Demonstrate Polysemy**

18. Count the number of synsets for the word.
19. Print all synsets with definitions, showing multiple meanings.

### **Step 9: Demonstrate Meronymy and Holonymy**

20. Select another word (e.g., "tree") for part-whole relation analysis.
21. Retrieve its synsets and select the first one.
22. Retrieve meronyms (parts) using `.part_meronyms()`.
23. Retrieve holonyms (wholes) using `.part_holonyms()`.
24. Print all meronyms and holonyms with definitions.

### **Step 10: Illustrate Homographs**

25. Choose a homograph example word (e.g., "bat").
26. Retrieve all synsets of the word.
27. Print each synset definition showing different meanings.

### **Step 11: Illustrate Homophones**

28. Explain that WordNet cannot identify homophones because it is spelling-based.
29. Provide example homophone pairs manually (e.g., "to, too, two").
```

```
# **Algorithm for Implementing N-Gram Probabilities**

### **Step 1: Import Required Libraries**

1. Import `nltk` and `defaultdict` from the `collections` module.
2. Download the NLTK tokenizer resource (`punkt`) if not already installed.

### **Step 2: Tokenize Input Text**

3. Define an input text string.
4. Convert the text to lowercase.
5. Tokenize the text into word tokens using `word_tokenize()`.

### **Step 3: Define Function to Build N-Gram Model**

6. Create a function `build_ngram_model(tokens, n)` that accepts tokens and the n-gram size.
7. Initialize `ngram_counts` to store counts of each n-gram.
8. Initialize `context_counts` to store counts of each (n-1)-gram context.
9. Check if the number of tokens is sufficient to form n-grams; if not, return an empty model.
10. Slide a window of size *n* over the tokens:
    a. Extract the n-gram tuple.
    b. Extract its context (first n−1 tokens).
    c. Update the n-gram count.
    d. Update the context count.
11. Initialize `ngram_probabilities` to store probability values.
12. For each n-gram:
    a. Identify its context.
    b. Calculate probability = n-gram count / context count.
    c. Handle cases where context does not exist by assigning probability zero.
13. Return the dictionary of n-gram probabilities.

### **Step 4: Build Bigram Model**

14. Call the function with `n = 2` to generate bigram probabilities.
15. Print sample bigram probabilities.

### **Step 5: Build Trigram Model**

16. Call the function with `n = 3` to generate trigram probabilities.
17. Print sample trigram probabilities.

### **Step 6: End Program**

18. Finish after displaying both bigram and trigram probability outputs.
```
