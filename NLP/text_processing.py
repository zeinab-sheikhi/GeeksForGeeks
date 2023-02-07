import nltk
import ssl
import string
import re
import inflect

# try:
#     _create_unverified_https_context = ssl._create_unverified_context
# except AttributeError:
#     pass
# else:
#     ssl._create_default_https_context = _create_unverified_https_context

# nltk.download()

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag, ne_chunk


def text_lowercase(text):
    return text.lower()

def remove_numbers(text):
    return re.sub(r'\d+', '', text)

def convert_numbers_to_text(text):
    temp_str = text.split()
    new_str = []
    for word in temp_str:
        if word.isdigit():
            temp = p.number_to_words(word)
            new_str.append(temp)
        else:
            new_str.append(word)    
    return ' '.join(new_str)

def remove_punctuations(text):
    return text.translate(str.maketrans('', '', string.punctuation))

def remove_whitespace(text):
    return ' '.join(text.split())

def remove_stopwords(text):
    stop_words = set(stopwords.words("english"))
    word_tokens = word_tokenize(text)
    return [word for word in word_tokens if word not in stop_words]

def stem_words(text):
    stemmer = PorterStemmer()
    word_tokens = word_tokenize(text)
    return [stemmer.stem(word) for word in word_tokens]

def lemmatize_word(text): 
    lemmatizer = WordNetLemmatizer()
    word_tokens = word_tokenize(text)
    return [lemmatizer.lemmatize(word, pos='v') for word in word_tokens]

def pos_tagging(text):
    word_tokens = word_tokenize(text)
    return pos_tag(word_tokens)

def chunking(text, grammar):
    word_tokens = word_tokenize(text)
    word_pos = pos_tag(word_tokens)
    chunkParser = nltk.RegexpParser(grammar)
    tree = chunkParser.parse(word_pos)
    for subtree in tree.subtrees():
        print(subtree)
    tree.draw()

def named_entity_recognition(text):
    word_tokens = word_tokenize(text)
    word_pos = pos_tag(word_tokens)
    print(ne_chunk(word_pos))

p = inflect.engine()

input_str = "Hey, did you know that the summer break is coming? Amazing right !! It's only 5 more days !!"
input_str_with_digits = "There are 3 balls in this bag, and 12 in the other one."
input_str_with_stopwords = "This is a sample sentence and we are going to remove the stopwords from this."
input_str_stem  = 'data science uses scientific methods algorithms and many types of processes'
input_str_pos  = 'You just gave me a scare'
input_str_chunk  = 'the little yellow bird is flying in the sky'
input_str_named_entity = 'Bill works for GeeksforGeeks so he went to Delhi for a meetup.'

grammar = "NP: {<DT>?<JJ>*<NN>}"

input_lower_str = text_lowercase(input_str)
# input_wihout_digits_str = remove_numbers(input_str)
input_wihout_digits_str = convert_numbers_to_text(input_str)
input_without_punctuation = remove_punctuations(input_str)
input_without_whitespace = remove_whitespace(input_str)
input_without_stopwords = remove_stopwords(input_str)
print(named_entity_recognition(input_str_named_entity))