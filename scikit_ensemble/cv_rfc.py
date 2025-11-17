#countvectorizer + random forest classifier
#why are they so similar? it's because they're both tree-based models AND A PART OF SCIKITLEARN LOL
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from pathlib import Path
import glob

directory = Path("src/webscraping/scraped_data")
text_paths = glob.glob(str(directory / "*.txt"))
text_files = [Path(text).stem for text in text_paths]

y = [0, 1, 1, 1, 0, 0, 0] #defo need more data

pipeline = Pipeline([
    ('countvectorizer', CountVectorizer(input='filename', stop_words='english', lowercase=True, ngram_range=(1, 2))), 
    ('rfclassifier', RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1))
])

#other parameters  can be added
#min_samples_split (min number of samples required to split a node, higher = less overfitting)
#min_samples_leaf (min samples required to be at a leaf node, higher = smoother predictions)
#max_features (number of features to consider at each split, sqrt vs log2 vs None)

X_train, X_test, Y_train, Y_test = train_test_split(text_paths, y, test_size=0.5, random_state=42, stratify=y)
pipeline.fit(X_train, Y_train)

y_prediction = pipeline.predict(X_test)
y_prediction

prediction = sum(y_prediction) / len(y_prediction)
prediction
if prediction > 0.6:
    print("Overall Sentiment: Positive")
elif prediction < 0.4:
    print("Overall Sentiment: Negative")

print(f"Accuracy: {accuracy_score(Y_test, y_prediction):.4f}")