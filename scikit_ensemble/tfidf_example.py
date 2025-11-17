#we are aiming for higher accuracy
#vectorizing first --> turning into a numerical list, numerical fingerprint like

#tf-idf:
#tf = term frequency, number of times a word appears in a doc / total terms in a doc
#idf = inverse document frequency, how many docs contain the word (x) vs total docs (n) -> log(n/(1+x))
#tf-idf score is the product of the two, rare words have greater weights

import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
#pd.set_option("max_rows", 600)
from pathlib import Path
import glob

directory = Path("src/webscraping/scraped_data")
text_paths = glob.glob(str(directory / "*.txt"))
text_files = [Path(text).stem for text in text_paths]

#calculating tf-idf for our files (we only have two right now but in practice there should be like 10)
tfidf_vectorizer = TfidfVectorizer(input='filename', stop_words='english')
#fit_transform learns the vocabulary and idf from the files and transforms them into a tf-idf matrix
tfidf_vector = tfidf_vectorizer.fit_transform(text_paths)

#creating data frame (aka table) with tf-idf scores
#index is just rows, set to be the name of docs
#get_feature_names_out() returns the list of vocab learned by the vectorizer
tfidf_df = pd.DataFrame(tfidf_vector.toarray(), index = text_files, columns = tfidf_vectorizer.get_feature_names_out())

tfidf_df.loc['Overall_frequency'] = (tfidf_df > 0).sum()

tfidf_df