import nltk 
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

print(stopwords.words('french'))

# Removes stop words from a piece of text
def remove_stopwords(text):
    word_tokens = word_tokenize(text)
    stop_words = stopwords.words('french')
    return [word for word in word_tokens if word.lower() not in stop_words]

# Performing the Stopwords operations in a file
def remove_stopwords_from_file(filepath):
    file = open(filepath)
    stop_words = stopwords.words('french')
    line = file.read()
    words = line.split() 
    for w in words:
        if w not in stop_words:
            appendFile = open('filteredtext.txt', 'a') 
            appendFile.write(" " + w) 
            appendFile.close() 