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

# Chunking is the process of extracting phrases from unstructured text and more structure to it. 
# It is also known as shallow parsing. It is done on top of Part of Speech tagging. 
# It groups word into “chunks”, mainly of noun phrases. Chunking is done using regular expressions.

# Named Entity Recognition is used to extract information from unstructured text. 
# It is used to classify entities present in a text into categories like a person, organization, event, places, etc. 
# It gives us detailed knowledge about the text and the relationships between the different entities.