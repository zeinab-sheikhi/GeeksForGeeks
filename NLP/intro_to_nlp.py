# Pipeline: breaking a complex problem into a number of small problems, making models for each of them and then integrating these models

# Step 1: Sentence Segmentation: Breaking the piece of text in various sentences.
# Step 2: Word Tokenization: Breaking the sentence into individual words called as tokens. 
# We can tokenize them whenever we encounter a space, we can train a model in that way.
# Even punctuations are considered as individual tokens as they have some meaning.
# Step 3: Predicting Parts of Speech for each token
# Step 4: Lemmatization: Feeding the model with the root word
# Step 5: Identifying stop words
# Step 6.1: Dependency Parsing:finding out the relationship between the words in the sentence and how they are related to each other
# Step 6.2: Finding Noun Phrases
# Step 7: Named Entity Recognition
# Step 8: Coreference Resolution

# Stemming is the process of getting the root form of a word. Stem or root is the part to which inflectional 
# affixes (-ed, -ize, -de, -s, etc.) are added. 

# Over-stemming: It refers to the situation where a stemmer produces a root form
# that is not a valid word or is not the correct root form of a word. Over-stemming can also be regarded as false-positives.

# Under-stemming: It refers to the situation where a stemmer does not produce the correct root form of a word
# or does not reduce a word to its base form. It can be interpreted as false-negatives.
# Approaches to this problem is to use techniques like semantic role labeling, sentiment analysis, context-based information, etc.
# that help to understand the context of the text and make the stemming process more precise.
# Stemmer algorithms: 

# Porter's Stemmer: simple, speedy, only supports English

# Lovin's Stemmer: fast, able to handle irregular plurals, but time consuming and fails to orm words from stem

# Dawson Stemmer: fast, but complex

# Krovetz Stemmer: light in nature, but inefficient

# Xerox Stemmer 

# N-Gram Stemmer: based on string comparisons and langugae dependant, but not time efficient

# Snowball Stemmer: multi-lingual stemmer

# Lancaster Stemmer: more aggressive and dynamic compared to the other two stemmers. 
# The stemmer is really faster, but the algorithm is really confusing when dealing with small words.
# But they are not as efficient as Snowball Stemmers.
# The Lancaster stemmers save the rules externally and basically uses an iterative algorithm

# Chunking is the process of extracting phrases from unstructured text and more structure to it. 
# It is also known as shallow parsing. It is done on top of Part of Speech tagging. 
# It groups word into “chunks”, mainly of noun phrases. Chunking is done using regular expressions.

# Named Entity Recognition is used to extract information from unstructured text. 
# It is used to classify entities present in a text into categories like a person, organization, event, places, etc. 
# It gives us detailed knowledge about the text and the relationships between the different entities.
