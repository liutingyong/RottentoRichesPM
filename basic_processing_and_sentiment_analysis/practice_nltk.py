import nltk
nltk.download("all")

#tokenization
statement = "Codeology is the best tech club. Avomatoes is the best internal project."

tokenized_words = nltk.word_tokenize(statement)
tokenized_sentences = nltk.sent_tokenize(statement)

print(tokenized_words)
print(tokenized_sentences)

#stemming
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()
stemmed_words = [stemmer.stem(word) for word in tokenized_words] #you can see how it's defo not the most accurate
print(stemmed_words)

#it's better for more general use cases
print(stemmer.stem("playful"))
print(stemmer.stem("playing"))
print(stemmer.stem("played"))
print(stemmer.stem("plays"))

#lemmatization
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
lemmatized_words = [lemmatizer.lemmatize(word, "v") for word in tokenized_words]
print(lemmatized_words)
