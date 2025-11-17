```Python
# Expt. 1: To study and implement word frequency count
import re
from collections import Counter
import nltk

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng') # Added this line to download the specific English POS tagger
nltk.download('punkt_tab') # Added this line to download the missing resource
def tokenize_text(text):
    tokens = {}
    # Sentence tokenization
    tokens['sentences'] = nltk.sent_tokenize(text)
    # Word tokenization
    tokens['words'] = nltk.word_tokenize(text)
    # Line, space, tab, and indentation tokenization
    tokens['lines'] = text.splitlines()
    tokens['spaces'] = re.findall(r'\s+', text)
    tokens['tabs'] = re.findall(r'\t+', text)
    # Indentation: count leading spaces/tabs per line
    tokens['indentation'] = [
        len(re.match(r'^(\s+)', line).group(1)) if re.match(r'^(\s+)',
        line) else 0
        for line in tokens['lines']
    ]
    # Punctuation: find all non-alphanumeric and non-whitespace characters
    tokens['punctuation'] = re.findall(r'[^\w\s]', text)
    return tokens
def count_word_frequency(words):
    return Counter(words)

# Example text
sample_text = """
This is a sample text. It contains multiple sentences.
 This line has a tab.
 This line has indentation.
It also has punctuation, like commas, periods, and exclamation marks!
Let's count the words. This text is for demonstration.
"""

# Tokenize the text
tokenized_data = tokenize_text(sample_text)
# Unique sets
unique_sentences = set(tokenized_data['sentences'])
unique_lines = set(tokenized_data['lines'])
unique_words = set(tokenized_data['words'])
unique_punctuation = set(tokenized_data['punctuation'])
# POS tagging
words_with_pos = nltk.pos_tag(tokenized_data['words'])
# Word frequency
word_freq = count_word_frequency(tokenized_data['words'])
# --- Print results ---
print("\n=== Unique Tokenization Results ===")
print("\nUnique Sentences:", unique_sentences)
print("Count of Unique Sentences:", len(unique_sentences))
print("\nUnique Lines:", unique_lines)
print("Count of Unique Lines:", len(unique_lines))
print("\nUnique Words:", unique_words)
print("Count of Unique Words:", len(unique_words))
print("\nUnique Punctuation:", unique_punctuation)
print("Count of Unique Punctuation:", len(unique_punctuation))
print("\n=== Total Token Counts (including duplicates) ===")
print("Total sentences:", len(tokenized_data['sentences']))
print("Total lines:", len(tokenized_data['lines']))
print("Total words:", len(tokenized_data['words']))
print("Total spaces:", len(tokenized_data['spaces']))
print("Total tabs:", len(tokenized_data['tabs']))
print("Total punctuation marks:", len(tokenized_data['punctuation']))
print("\n=== Words with POS Tags ===")
print(words_with_pos[:10])
print("\n=== Word Frequency (Top 10) ===")
print(word_freq.most_common(10))
```

```Python
# Expt. 2: To implement a python program for case folding, stop word removal and time token ratio
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
# Sample sentence
sentence = """Natural Language Processing (NLP) is a subfield of linguistics,
computer science, and artificial intelligence. It is concerned with the
interactions between computers and human (natural) languages."""
# 1. Case folding: convert to lowercase
sentence = sentence.lower()
print(sentence)
# 2. Remove punctuation
sentence = sentence.translate(str.maketrans('', '', string.punctuation))
# 3. Tokenize the sentence
tokens = word_tokenize(sentence)
# 4. Remove stop words
stop_words = set(stopwords.words('english'))
filtered_tokens = [word for word in tokens if word not in stop_words]
# 5. Calculate Type-Token Ratio (TTR)
types = set(filtered_tokens)
ttr = len(types) / len(filtered_tokens) if filtered_tokens else 0
print(f"Original Sentence: {sentence}")
print(f"Tokens after stop word removal: {filtered_tokens}")
print(f"Number of Types (Unique Words): {len(types)}")
print(f"Number of Tokens (Total Words): {len(filtered_tokens)}")
print(f"Type-Token Ratio (TTR): {ttr:.2f}")
```

```Python
# Expt. 3: To study and implement regular expression
import re
# Sample text
text = """
Hello, my name is John Doe. My email is john.doe@example.com and my
backup is jane_doe@@example and john123@work.net.
I have 2 dogs and 15 cats. Call me at 123-456-7890 and +91-9876543210
or my office +91 1234567890.
"""
# 1. Find all email addresses
emails = re.findall(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}',
text)
print(f"Emails found: {emails}")
possible_emails = re.findall(r'\S+[@#]\S+', text)
wrong_emails = [email for email in possible_emails if not
re.fullmatch(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', email)]
print(f"Wrong Emails: {wrong_emails}")
# 2. Find all numbers
numbers = re.findall(r'\d+', text)
print(f"Numbers found: {numbers}")
indian_numbers = re.findall(r'\+91[-\s]?\d{10}', text)
print(f"Indian Numbers (+91): {indian_numbers}")
# 3. Replace all numbers with '#'
text_with_numbers_masked = re.sub(r'\d+', '#', text)
print(f"Text after replacing numbers:\n{text_with_numbers_masked}")
# 4. Check if the word 'call' appears (case-insensitive)
pattern = r'call'
match = re.search(pattern, text, re.IGNORECASE)
if match:
 print(f"The word 'call' was found at position {match.start()}.")
else:
 print("The word 'call' was not found.")
```

```Python
# Expt. 4: To study and implement n-grams probability

import nltk
from collections import defaultdict

required_nltk_data = ['punkt']
for resource in required_nltk_data:
 try:
  nltk.data.find(f'tokenizers/{resource}')
 except LookupError:
  print(f"Downloading {resource}...")
  nltk.download(resource)
from nltk.tokenize import word_tokenize

text = """This is a simple example text for n gram probability
demonstration."""
# Tokenize the text into words
tokens = word_tokenize(text.lower())
# Function to build n-gram model
def build_ngram_model(tokens, n):
 ngram_counts = defaultdict(int)
 context_counts = defaultdict(int)
 if len(tokens) < n:
  print(f"Warning: Not enough tokens to build {n}-gram model.")
  return {}
 for i in range(len(tokens) - n + 1):
  ngram = tuple(tokens[i:i+n])
  context = tuple(ngram[:-1])
  ngram_counts[ngram] += 1
  context_counts[context] += 1
 ngram_probabilities = defaultdict(float)
 for ngram, count in ngram_counts.items():
  context = tuple(ngram[:-1])
  if context in context_counts and context_counts[context] > 0:
   ngram_probabilities[ngram] = count / \
  context_counts[context]
  else:
   ngram_probabilities[ngram] = 0.0 # Handle cases with no
   context
 return ngram_probabilities
# Build a bigram model (n=2)
bigram_probabilities = build_ngram_model(tokens, 2)
# Print some example bigram probabilities
print("Example Bigram Probabilities:")
if bigram_probabilities:
 for bigram, prob in list(bigram_probabilities.items())[:10]: #
  print(f"{' '.join(bigram)}: {prob:.4f}")
else:
 print("No bigram probabilities to display.")
# Build a trigram model (n=3)
trigram_probabilities = build_ngram_model(tokens, 3)
# Print some example trigram probabilities
print("\nExample Trigram Probabilities:")
if trigram_probabilities:
 for trigram, prob in list(trigram_probabilities.items())[:10]: #
  print(f"{' '.join(trigram)}: {prob:.4f}")
else:
 print("No trigram probabilities to display.")
```

```Python
# Expt. 5: To study and implement segmentation, tokenization, stemming, legalization, lemmatization, and parts of speech tagging

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag

# Example text
text = """Natural Language Processing (NLP) is a subfield of linguistics,
computer science, and artificial intelligence. It is concerned with the
interactions between computers and human (natural) languages."""
print("Original Text:")
print(text)
print("-" * 30)
# 1. Segmentation (Sentence Tokenization)
sentences = sent_tokenize(text)
print("Sentence Segmentation:")
for i, sentence in enumerate(sentences):
 print(f"Sentence {i+1}: {sentence}")
print("-" * 30)
# 2. Tokenization (Word Tokenization)
words = word_tokenize(text)
print("Word Tokenization:")
print(words)
print("-" * 30)
# 3. Lowercasing (part of legalization/normalization)
lowercase_words = [word.lower() for word in words]
print("Lowercase Tokens:")
print(lowercase_words)
print("-" * 30)
# 4. Stemming
stemmer = PorterStemmer()
stemmed_words = [stemmer.stem(word) for word in lowercase_words]
print("Stemming:")
print(stemmed_words)
print("-" * 30)
# 5. Lemmatization
lemmatizer = WordNetLemmatizer()
lemmatized_words = [lemmatizer.lemmatize(word) for word in
lowercase_words]
print("Lemmatization (default POS='n'):")
print(lemmatized_words)
print("-" * 30)
# 6. Parts of Speech Tagging
pos_tags = pos_tag(words)
print("Parts of Speech Tagging:")
print(pos_tags)
print("-" * 30)
```

```Python
# Expt. 6: To study and implement hidden markov model

import numpy as np
# 1. Define states
states = ['Hot', 'Cold']
# 2. Define observations
observations = ['Walk', 'Shop', 'Clean']
# 3. Define initial state probabilities
start_probability = {'Hot': 0.8, 'Cold': 0.2}
# 4. Define transition probabilities
transition_probability = {
 'Hot': {'Hot': 0.7, 'Cold': 0.3},
 'Cold': {'Hot': 0.4, 'Cold': 0.6}
}
# 5. Define emission probabilities
emission_probability = {
 'Hot': {'Walk': 0.2, 'Shop': 0.4, 'Clean': 0.4},
 'Cold': {'Walk': 0.6, 'Shop': 0.3, 'Clean': 0.1}
 }

def viterbi(obs, states, start_p, trans_p, emit_p):
    V = [{}]
    path = {}

    # Initialize base cases (t == 0)
    for st in states:
        V[0][st] = start_p[st] * emit_p[st].get(obs[0], 0)
        path[st] = [st]

    # Run Viterbi for t > 0
    for t in range(1, len(obs)):
        V.append({})
        newpath = {}

        for st in states:
            (prob, state) = max(
                (V[t - 1][prev_st] * trans_p[prev_st].get(st, 0) * emit_p[st].get(obs[t], 0), prev_st)
                for prev_st in states
            )
            V[t][st] = prob
            newpath[st] = path[state] + [st]

        path = newpath

    # Find the maximum probability and the corresponding path
    (prob, state) = max((V[len(obs) - 1][st], st) for st in states)
    return prob, path[state]

print("States:", states)
print("Observations:", observations)
print("Initial Probabilities:", start_probability)
print("Transition Probabilities:", transition_probability)
print("Emission Probabilities:", emission_probability)

example_obs = ['Walk', 'Shop', 'Clean']
probability, path = viterbi(example_obs, states, start_probability,
transition_probability, emission_probability)
print("Most likely state sequence for observation sequence:",
example_obs)
print("Sequence:", path)
print("Probability:", probability)
```

```Python
# Expt. 7: To study and implement Name entity recognition(NER)

!pip install spacy
!python -m spacy download en_core_web_sm
import spacy
!python -m spacy download en_core_web_sm
nlp = spacy.load("en_core_web_sm")
# # Load the English language model
# try:
#  nlp = spacy.load("en_core_web_sm")
# except OSError:
#   print("SpaCy model 'en_core_web_sm' not found. Downloading...")
#   !python -m spacy download en_core_web_sm
#   nlp = spacy.load("en_core_web_sm")
# Example text
text = """Apple Inc. is an American multinational technology company
headquartered in Cupertino, California."""
print("Original Text:")
print(text)
print("-" * 50)
# Process the text with SpaCy
doc = nlp(text)
print("Named Entities:")
# Iterate over the entities in the document
for ent in doc.ents:
 print(f"Entity: {ent.text}, Label: {ent.label_}, Explanation: "
       f"{spacy.explain(ent.label_)}")
print("-" * 50)
```

```Python
# Expt. 8: To study and implement wordnet in NLP

import nltk
nltk.download('wordnet')
from nltk.corpus import wordnet

# Example Word: "great"
word = "great"
print(f"Exploring WordNet for the word: '{word}'")
print("-" * 30)
# 1. Get Synsets (sets of synonyms)
synsets = wordnet.synsets(word)
print(f"Synsets for '{word}':")
for synset in synsets:
    print(f"- {synset.name()}: {synset.definition()}")
print("-" * 30)
# 2. Get all unique lemmas (synonyms) from synsets
lemmas = set()
for synset in synsets:
    for lemma in synset.lemmas():
        lemmas.add(lemma.name())
print(f"Synonyms for '{word}':")
print(lemmas)
print("-" * 30)
# 3. Get Antonyms (opposites) - may not exist for all words/synsets
antonyms = set()
for synset in synsets:
    for lemma in synset.lemmas():
        if lemma.antonyms():
            for antonym in lemma.antonyms():
                antonyms.add(antonym.name())
if antonyms:
    print(f"Antonyms for '{word}':")
    print(antonyms)
else:
    print(f"No direct antonyms found for '{word}'.")
print("-" * 30)
# 4. Explore Relationships (Hypernyms - broader terms)
print(f"Hypernyms (broader terms) for the first synset of '{word}' "
      f"({synsets[0].name()}):")
if synsets:
    for hypernym in synsets[0].hypernyms():
        print(f"- {hypernym.name()}: {hypernym.definition()}")
else:
    print("No synsets found to explore relationships.")
print("-" * 30)
# 5. Explore Relationships (Hyponyms - more specific terms)
print(f"Hyponyms (more specific terms) for the first synset of '{word}' "
      f"({synsets[0].name()}):")
if synsets:
    for hyponym in synsets[0].hyponyms():
        print(f"- {hyponym.name()}: {hyponym.definition()}")
else:
    print("No synsets found to explore relationships.")
print("-" * 30)
# 6. Demonstrate Polysemy
print(f"Demonstrating Polysemy for the word '{word}':")
if synsets:
    print(f"The word '{word}' has {len(synsets)} synsets, indicating "
          f"different meanings (polysemy).")
    for synset in synsets:
        print(f"- {synset.name()}: {synset.definition()}")
else:
    print(f"No synsets found for '{word}' to demonstrate polysemy.")
print("-" * 30)
# 7. Demonstrate Meronymy (parts) and Holonymy (wholes) for a different word
word_for_meroholonymy = "tree"
print(f"Exploring Meronymy (parts) and Holonymy (wholes) for the word: "
      f"'{word_for_meroholonymy}'")
print("-" * 30)
tree_synsets = wordnet.synsets(word_for_meroholonymy)
if tree_synsets:
    first_tree_synset = tree_synsets[0] # Using the first synset as an example
    print(f"Meronyms (parts of a '{first_tree_synset.name()}') :")
    meronyms = first_tree_synset.part_meronyms()
    if meronyms:
        for meronym in meronyms:
            print(f"- {meronym.name()}: {meronym.definition()}")
    else:
        print(f"No part meronyms found for "
              f"'{first_tree_synset.name()}'.")
    print(f"\nHolonyms (wholes that a '{first_tree_synset.name()}' is a "
          f"part of):")
    holonyms = first_tree_synset.part_holonyms()
    if holonyms:
        for holonym in holonyms:
            print(f"- {holonym.name()}: {holonym.definition()}")
    else:
        print(f"No part holonyms found for "
              f"'{first_tree_synset.name()}'.")
else:
    print(f"No synsets found for '{word_for_meroholonymy}' to "
          f"demonstrate meronymy and holonymy.")
print("-" * 30)
# 8. Illustrate Homographs (words spelled the same but with different meanings)
print("Illustrating Homographs:")
print("Homographs are words that are spelled the same but have "
      "different meanings and often different pronunciations.")
print("WordNet helps identify potential homographs by providing "
      "multiple synsets for the same spelling.")
example_homograph_word = "bat"
bat_synsets = wordnet.synsets(example_homograph_word)
if bat_synsets:
    for synset in bat_synsets:
        print(f"- {synset.name()}: {synset.definition()}")
else:
    print(f"No synsets found for '{example_homograph_word}'.")
print("-" * 30)
# 9. Illustrate Homophones (words pronounced the same but with different spellings and meanings)
print("Illustrating Homophones:")
print("Homophones are words that sound the same but have different "
      "spellings and meanings.")
print("WordNet does not directly support finding homophones as it is "
      "based on word spelling, not pronunciation.")
print("Examples: 'to', 'too', 'two' or 'there', 'their', 'they're'.")
print("-" * 30)
```

``` Python
# Expt. 9:  To study and implement Probabilistic context free grammar (PCFG) in NLP

# pip install nltk
from nltk.grammar import PCFG
from nltk.parse import ViterbiParser, ChartParser
from nltk import Tree
from nltk.grammar import Nonterminal
# --- Define a PCFG with explicit probabilities ---
grammar_str = """
S -> NP VP [1.0]
# NP alternatives (sums to 1.0)
NP -> Det N [0.4]
NP -> NP PP [0.3]
NP -> Pronoun [0.3]
# VP alternatives (sums to 1.0)
VP -> V NP [0.5]
VP -> VP PP [0.5]
PP -> P NP [1.0]
Det -> 'the' [0.9]
Det -> 'a' [0.1]
N -> 'man' [0.6]
N -> 'telescope' [0.4]
Pronoun -> 'I' [1.0]
V -> 'saw' [1.0]
P -> 'with' [1.0]
"""
pcfg = PCFG.fromstring(grammar_str)
print("Loaded PCFG:\n")
print(pcfg)
print("\n" + "="*60 + "\n")
# --- Parsers ---
viterbi = ViterbiParser(pcfg)
chart = ChartParser(pcfg)
# Ambiguous sentence
sentence = "I saw the man with a telescope"
tokens = sentence.split()
# Helper: compute probability of a Tree under the PCFG by multiplying
# production probabilities
def tree_probability(tree: Tree, grammar: PCFG) -> float:
    prod_probs = {prod: prod.prob() for prod in grammar.productions()}
    prob = 1.0
    # recursively walk the tree and collect productions
    def walk(t):
        nonlocal prob
        if isinstance(t, Tree):
            lhs = Nonterminal(t.label())
            rhs = []
            for child in t:
                if isinstance(child, Tree):
                    rhs.append(Nonterminal(child.label()))
                else:
                    rhs.append(child)
            # find matching production in grammar (there should be
            # exactly one match for PCFG grammar rules)
            matched = None
            for p in grammar.productions(lhs=lhs):
                # compare RHS symbols as strings/Nonterminals
                # convert p.rhs() to comparable form
                rhs_form = []
                for x in p.rhs():
                    if isinstance(x, Nonterminal):
                        rhs_form.append(Nonterminal(str(x)))
                    else:
                        rhs_form.append(x)
                if tuple(rhs_form) == tuple(rhs):
                    matched = p
                    break
            if matched is None:
                # if no production match found, the probability is 0
                # (shouldn't happen if tree is from this grammar)
                return 0.0
            prob *= matched.prob()
            # recurse
            for child in t:
                if isinstance(child, Tree):
                    walk(child)
    walk(tree)
    return prob
# --- Get most probable parse (Viterbi) ---
print("Most probable parse (Viterbi):")
viterbi_trees = list(viterbi.parse(tokens))
if viterbi_trees:
    best = viterbi_trees[0]
    print(best)
    try:
        best.pretty_print()
    except Exception:
        pass
    print(f"Probability (Viterbi tree .prob if available): "
          f"{getattr(best, 'prob', lambda: tree_probability(best, pcfg))()}\n")
else:
    print("No Viterbi parse found.\n")
# --- Get all parses (ChartParser) to show ambiguity ---
print("="*40)
print("All parses (ChartParser) - demonstrates ambiguity:")
all_trees = list(chart.parse(tokens))
if not all_trees:
    print("No parses found with this grammar.")
else:
    for i, t in enumerate(all_trees, start=1):
        print(f"\nParse #{i}:")
        print(t)
        try:
            t.pretty_print()
        except Exception:
            pass
        # compute probability under PCFG
        prob = tree_probability(t, pcfg)
        print(f"Computed probability (product of rule probs): "
              f"{prob:.8f}")
# --- Compare probabilities and indicate which parse is preferred ---
if len(all_trees) > 1:
    probs = [(i+1, tree_probability(t, pcfg), t) for i,t in
             enumerate(all_trees)]
    probs_sorted = sorted(probs, key=lambda x: x[1], reverse=True)
    print("\n" + "="*40)
    print("Parses sorted by probability (highest first):")
    for idx, p, t in probs_sorted:
        print(f"Parse #{idx} \u2014 probability = {p:.8f}")
    best_idx, best_prob, best_tree = probs_sorted[0]
    print(f"\nViterbi selected parse #{best_idx} with probability "
          f"{best_prob:.8f}.")
```

``` Python
# Expt. 10: To study and implement Information Retrieval in NLP
# Define a list of documents
documents = [
    """Natural Language Processing (NLP) is a subfield of linguistics,
computer science, and artificial intelligence.""",
    """Information Retrieval (IR) is the activity of obtaining
information resources relevant to an information need from a collection
of information resources.""",
    """Machine learning is a field of artificial intelligence that uses
statistical techniques to give computer systems the ability to 'learn'
from data.""",
    """Deep learning is a subset of machine learning that uses neural
networks with many layers.""",
    """A computer science student is studying artificial intelligence and
machine learning."""
]
print("Documents loaded successfully.")
for i, doc in enumerate(documents):
    print(f"Document {i+1}: {doc}")

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download necessary NLTK data (if you haven't already)
required_nltk_data = ['punkt', 'wordnet', 'stopwords', 'punkt_tab']
for resource in required_nltk_data:
    try:
        nltk.data.find(f'tokenizers/{resource}') # Check for tokenizers
    except LookupError:
        try:
            nltk.data.find(f'corpora/{resource}') # Check for corpora
        except LookupError:
            print(f"Downloading {resource}...")
            nltk.download(resource)

def preprocess_text(doc):
    """
    Preprocesses a single document by tokenizing, lowercasing, removing
    punctuation and stop words, and applying stemming and
    lemmatization.
    Args:
    doc (str): The input document string.
    Returns:
    list: A list of preprocessed tokens.
    """
    # Tokenize
    tokens = word_tokenize(doc)
    # Lowercase, remove punctuation, and remove stop words
    stop_words = set(stopwords.words('english'))
    processed_tokens = []
    for token in tokens:
        token = token.lower()
        token = token.translate(str.maketrans('', '', string.punctuation)) # Remove punctuation
        if token and token not in stop_words: # Check if token is not empty after punctuation removal and not a stop word
            processed_tokens.append(token)

    # Stemming and Lemmatization
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    # Apply both and decide which to keep, or keep both.
    # For this example, we will keep the lemmatized version as it's generally preferred.
    # If you wanted to keep both, you could create pairs or separate lists.
    final_tokens = []
    for token in processed_tokens:
        stemmed_token = stemmer.stem(token)
        lemmatized_token = lemmatizer.lemmatize(token)
    # In a real scenario, you might choose one over the other based on your needs
    # Here, we'll just use lemmatization for the final output
        final_tokens.append(lemmatized_token)
    return final_tokens

# Apply preprocessing to each document
preprocessed_documents = [preprocess_text(doc) for doc in documents]

# Print original and preprocessed documents for verification
print("Original Documents:")
for i, doc in enumerate(documents):
    print(f"Document {i+1}: {doc}")
print("\nPreprocessed Documents:")
for i, processed_doc in enumerate(preprocessed_documents):
    print(f"Document {i+1}: {processed_doc}")

# Join the preprocessed tokens back into strings
preprocessed_documents = [" ".join(doc) for doc in preprocessed_documents]

# Instantiate TfidfVectorizer
vectorizer = TfidfVectorizer()
# Fit and transform the preprocessed documents
tfidf_matrix = vectorizer.fit_transform(preprocessed_documents)

# Print the shape of the TF-IDF matrix and feature names
print("Shape of TF-IDF matrix:", tfidf_matrix.shape)
print("Feature names (terms):")
print(vectorizer.get_feature_names_out())

def search(query, vectorizer, tfidf_matrix, documents):
    """
    Preprocesses a query, calculates its TF-IDF representation, and
    ranks
    documents based on cosine similarity to the query.
    Args:
    query (str): The search query.
    vectorizer (TfidfVectorizer): The fitted TF-IDF vectorizer.
    tfidf_matrix (sparse matrix): The TF-IDF matrix of the
    documents.
    documents (list): The list of original documents.
    Returns:
    list: A list of tuples, where each tuple contains the document
    index,
    the original document text, and its similarity score,
    ranked by similarity in descending order.
    """
    # Preprocess the query
    preprocessed_query_tokens = preprocess_text(query)
    preprocessed_query = " ".join(preprocessed_query_tokens)

    # Transform the preprocessed query into TF-IDF
    # Use the same vectorizer fitted on the documents
    query_tfidf = vectorizer.transform([preprocessed_query])

    # Calculate cosine similarity between the query and documents
    cosine_sim = cosine_similarity(query_tfidf, tfidf_matrix).flatten()

    # Get the indices of documents sorted by similarity in descending
    #order
    # Use argpartition for efficiency if you only need top k, but
    #argsort for full ranking
    # sorted_indices = np.argsort(cosine_sim)[::-1] # Using numpy

    # Let's use a simple list comprehension for ranking without numpy
    # dependency
    ranked_results = sorted(enumerate(cosine_sim), key=lambda item: item[1], reverse=True)

    # Prepare the results list
    results = []
    for doc_index, score in ranked_results:
        results.append((doc_index, documents[doc_index], score))
    return results

# Example query demonstration
example_query = "artificial intelligence and machine learning"
search_results = search(example_query, vectorizer, tfidf_matrix, documents)
print(f"\nSearch results for query: '{example_query}'")
print("-" * 50)
for doc_index, doc_text, score in search_results:
    print(f"Document {doc_index+1} (Score: {score:.4f}): {doc_text}")

example_queries = [
    "nlp and information retrieval",
    "machine learning and deep learning",
    "computer science student"
]

for query in example_queries:
    print(f"\n--- Search results for query: '{query}' ---")
    search_results = search(query, vectorizer, tfidf_matrix, documents)
    if search_results:
        for doc_index, doc_text, score in search_results:
        # 4. Print document index (1-based), score, and text
            print(f"Document {doc_index+1} (Score: {score:.4f}):\n{doc_text}")
    else:
        print("No relevant documents found.")

print("\nSearch demonstration complete.")
```
