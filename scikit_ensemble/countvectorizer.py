#converting text into bag-of-words representation
#createsa matrix of word counts for each document
#similar to tfidf but simpler, tfidf downweighs common words and upweighs rarer words

import sklearn
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
#pd.set_option("max_rows", 600)
from pathlib import Path
import glob

#this is super similar to tfidf, just without the idf part lol
#might be better for random forest bc it works with more raw features + rf is more robust to noise
directory = Path("src/webscraping/scraped_data")
text_paths = glob.glob(str(directory / "*.txt"))
text_files = [Path(text).stem for text in text_paths]

vectorizer = CountVectorizer(input='filename', stop_words='english', lowercase=True, ngram_range=(1, 2))

vector = vectorizer.fit_transform(text_paths)

vocab = vectorizer.get_feature_names_out()

df = pd.DataFrame(vector.toarray(), index=text_files, columns=vocab)

df.loc['Overall_frequency'] = (df > 0).sum()

df

