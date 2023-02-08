# WordNet is a large lexical database of English

from nltk.corpus import wordnet 
syns = wordnet.synsets(('program'))
print(syns)

# An example of a synset:
print(syns[0].name())
  
# Just the word:
print(syns[0].lemmas()[0].name())
  
# Definition of that first synset:
print(syns[0].definition())
  
# Examples of the word in use in sentences:
print(syns[0].examples())

synonyms = []
antonyms = []
  
for syn in wordnet.synsets("good"):
    for lem in syn.lemmas():
        synonyms.append(lem.name())
        if lem.antonyms():
            antonyms.append(lem.antonyms()[0].name())
  
print(set(synonyms))
print(set(antonyms))

# Compare the similarity between ship and boat
w1 = wordnet.synset('ship.n.01')
w2 = wordnet.synset('boat.n.01')  # n denotes noun
print(w1.wup_similarity(w2))