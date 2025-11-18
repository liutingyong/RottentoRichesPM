#now using tf-idf for xgboost, which does the actual predictions
#we should tune later
#tfidf finds more relative importance of words in documents
#xgboost benefits from more weighted features


from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
#import joblib
import glob

from pathlib import Path

# Get path relative to script location (works from any directory)
#directory = Path("src/webscraping/review_data/training/scraped_data")
script_dir = Path(__file__).parent
project_root = script_dir.parent  # Go up from scikit_ensemble -> project root
directory = project_root / "webscaping" / "review_data" / "training" / "scraped_data"

text_paths = glob.glob(str(directory / "*.txt"))
text_files = [Path(text).stem for text in text_paths]

y = [1, 1, 1, 1, 0, 0,0,0,0] + 31 *[0] + 27 * [1] #defo need more data

X_train, X_test, Y_train, Y_test = train_test_split(text_paths, y, test_size=0.5, random_state=42, stratify=y)

# ============================================================================
# HYPERPARAMETERS TO EXPERIMENT WITH
# ============================================================================
# These are the most important parameters to tune for better performance
# Try changing one at a time and see how accuracy changes

pipe = Pipeline([
    ("tfidf", TfidfVectorizer(
        input='filename', 
        stop_words='english', 
        lowercase=True, 
        # ngram_range=(1, 2): Controls word combinations
        #   (1, 1) = single words only
        #   (1, 2) = single words + 2-word phrases (current)
        #   (1, 3) = includes 3-word phrases (more features, slower)
        #   Try: (1, 1) for speed, (1, 3) for more context
        ngram_range=(1, 3),
        
        # min_df=1: Minimum document frequency for a word to be included
        #   min_df=1 = include all words (even if only in 1 document)
        #   min_df=2 = word must appear in at least 2 documents
        #   Higher = removes rare words, reduces noise, fewer features
        #   Try: 2 or 3 to filter out typos/rare words
        min_df=2,
        
        # max_df=0.95: Maximum document frequency (removes very common words)
        #   0.95 = remove words in >95% of documents (like "the", "a")
        #   Lower = removes more common words
        #   Try: 0.8-0.9 to remove more stopwords
        max_df=0.95,
        
        # max_features=30000: Maximum number of features (words/phrases) to use
        #   Higher = more words, more detail, but slower and risk of overfitting
        #   Lower = faster, simpler model, but might miss important words
        #   Try: 10000 (faster), 50000 (more detail), or None (use all)
        max_features=50000
    )),
    ("xgb", XGBClassifier(
        objective='binary:logistic',  # Binary classification with logistic regression
        
        # n_estimators=400: Number of trees (boosting rounds)
        #   More trees = better learning but slower and risk of overfitting
        #   Fewer trees = faster but might underfit
        #   Try: 100 (fast), 200 (balanced), 600-800 (more power, watch for overfitting)
        n_estimators=600,
        
        # learning_rate=0.05: How much each tree contributes (step size)
        #   Lower = slower learning, more trees needed, but more careful/stable
        #   Higher = faster learning, fewer trees needed, but risk of overshooting
        #   Try: 0.01 (very careful), 0.1 (faster), 0.2 (aggressive)
        #   Rule: learning_rate × n_estimators should stay roughly constant
        learning_rate=0.03,
        
        # max_depth=6: Maximum depth of each tree
        #   Deeper = more complex patterns, better fit, but risk of overfitting
        #   Shallower = simpler, faster, more generalizable
        #   Try: 3-4 (simple), 8-10 (complex), 6 is balanced
        max_depth=10,
        
        # subsample=0.9: Fraction of training data used per tree (row sampling)
        #   Lower = more randomness, prevents overfitting, but less data per tree
        #   Higher = uses more data, might overfit
        #   Try: 0.7-0.8 (more regularization), 1.0 (use all data)
        subsample=0.9,
        
        # colsample_bytree=0.9: Fraction of features used per tree (column sampling)
        #   Lower = more feature randomness, prevents overfitting
        #   Higher = uses more features, might overfit
        #   Try: 0.7-0.8 (more regularization), 1.0 (use all features)
        colsample_bytree=0.9,
        
        # reg_lambda=1.0: L2 regularization (penalizes large weights)
        #   Higher = stronger regularization, simpler model, prevents overfitting
        #   Lower = allows more complex patterns, risk of overfitting
        #   Try: 0.5 (less regularization), 2.0-5.0 (more regularization if overfitting)
        reg_lambda=1.0,
        
        tree_method='hist',  # Fast histogram-based tree construction
        eval_metric='logloss',  # Logarithmic loss - standard for binary classification
        n_jobs=-1  # Use all CPU cores for parallel training
    ))
])


pipe.fit(X_train, Y_train)

pred = pipe.predict(X_test)
#here our prediction will be marked through at a threshold of 0.5. if not confident enoug it will just classify as 0
#we can adjust this threshold later if we want to be more/less conservative
#left is the probability of class 0, right is the probability of class 1
probability = pipe.predict_proba(X_test)[:, 1]
print(f"Accuracy: {accuracy_score(Y_test, pred):.4f}")

#overall decision
prediction = sum(pred) / len(pred)
prediction
if prediction > 0.6:
    print("Overall Sentiment: Positive")
elif prediction < 0.4:
    print("Overall Sentiment: Negative")