# now using tf-idf for xgboost, which does the actual predictions
# we should tune later
# tfidf finds more relative importance of words in documents
# xgboost benefits from more weighted features


from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
# import joblib
import glob

from pathlib import Path

# Get path relative to script location (works from any directory)
# directory = Path("src/webscraping/review_data/training/scraped_data")
script_dir = Path(__file__).parent
project_root = script_dir.parent  # Go up from scikit_ensemble -> project root
directory = project_root / "webscaping" / \
    "review_data" / "training" / "scraped_data"

text_paths = glob.glob(str(directory / "*.txt"))
text_files = [Path(text).stem for text in text_paths]

y = [1, 1, 1, 1, 0, 0, 0, 0, 0] + 31 * [0] + 31 * [1]  # defo need more data

# honestly we can change the parameters later, idk if these are the best
X_train, X_test, Y_train, Y_test = train_test_split(
    text_paths, y, test_size=0.2, random_state=42, stratify=y)
pipe = Pipeline([("tfidf", TfidfVectorizer(input='filename', stop_words='english', lowercase=True, ngram_range=(1, 2), min_df=1, max_df=0.95, max_features=30000)),
                 ("xgb", XGBClassifier(objective='binary:logistic', n_estimators=800, learning_rate=0.03, max_depth=8, subsample=0.9, colsample_bytree=0.9, reg_lambda=1.5, reg_alpha=0.5, tree_method='hist', eval_metric='logloss', n_jobs=-1))])
# binary:logistic: binary classification with logistic regression
# n_estimators is the number of trees to build, mroe trees = more learning power but chance of overfitting
# higher depth (max_depth) means more complex trees --> more prone to overfitting
# logloss is logarithmic loss, standard for binary classification, penalizes confident wrong predictions
# n_jobs = -1 means it uses all available cores for training in parallel
# xgb trains 400 shallow-to-medium depth trees,
# sampling of both rows and features for regularization (prevents overfitting), subsample for rows and colsample_bytree for features
# learns slowly but carefully with low learning rate
# outputs probabilities for our two classes
# reg_lambda is L2 regularization term, discourages large weights, model now relies on more features


pipe.fit(X_train, Y_train)

pred = pipe.predict(X_test)
# here our prediction will be marked through at a threshold of 0.5. if not confident enoug it will just classify as 0
# we can adjust this threshold later if we want to be more/less conservative
# left is the probability of class 0, right is the probability of class 1
probability = pipe.predict_proba(X_test)[:, 1]
print(f"Accuracy: {accuracy_score(Y_test, pred):.4f}")

# overall decision
prediction = sum(pred) / len(pred)
prediction
if prediction > 0.6:
    print("Overall Sentiment: Positive")
elif prediction < 0.4:
    print("Overall Sentiment: Negative")
