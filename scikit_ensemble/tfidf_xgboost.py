#now using tf-idf for xgboost, which does the actual predictions
#OPTIMIZED WITH TUNED HYPERPARAMETERS
#tfidf finds more relative importance of words in documents
#xgboost benefits from more weighted features

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from xgboost import XGBClassifier
import joblib
import glob
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Get path relative to script location (works from any directory)
script_dir = Path(__file__).parent
project_root = script_dir.parent  # Go up from scikit_ensemble -> project root
directory = project_root / "webscaping" / "review_data" / "training" / "scraped_data"

text_paths = glob.glob(str(directory / "*.txt"))
text_files = [Path(text).stem for text in text_paths]

# Labels: 9 initial labels + 31 zeros + 31 ones = 71 total
y = [1, 1, 1, 1, 0, 0, 0, 0, 0] + 31 * [0] + 31 * [1]

print(f"Total samples: {len(text_paths)}")
print(f"Total labels: {len(y)}")
print(f"Positive samples: {sum(y)}, Negative samples: {len(y) - sum(y)}")

# Split data
X_train, X_test, Y_train, Y_test = train_test_split(
    text_paths, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining samples: {len(X_train)}, Test samples: {len(X_test)}")

# Optimized pipeline with tuned hyperparameters
# These parameters were found through hyperparameter tuning (RandomizedSearchCV)
pipe = Pipeline([
    ("tfidf", TfidfVectorizer(
        input='filename', 
        stop_words='english', 
        lowercase=True, 
        ngram_range=(2, 2),  # Tuned: bigrams only
        min_df=3,  # Tuned: minimum document frequency
        max_df=0.95,  # Tuned: maximum document frequency
        max_features=30000  # Tuned: feature limit
    )),
    ("xgb", XGBClassifier(
        objective='binary:logistic', 
        n_estimators=300,  # Tuned: number of trees
        learning_rate=0.01,  # Tuned: learning rate
        max_depth=5,  # Tuned: tree depth
        subsample=0.8,  # Tuned: row sampling
        colsample_bytree=0.9,  # Tuned: feature sampling
        reg_lambda=2.0,  # Tuned: L2 regularization
        reg_alpha=0,  # Tuned: L1 regularization
        tree_method='hist', 
        eval_metric='logloss', 
        n_jobs=-1,
        random_state=42
    ))
])

#binary:logistic: binary classification with logistic regression
#n_estimators is the number of trees to build, more trees = more learning power but chance of overfitting
#higher depth (max_depth) means more complex trees --> more prone to overfitting
#logloss is logarithmic loss, standard for binary classification, penalizes confident wrong predictions
#n_jobs = -1 means it uses all available cores for training in parallel
#xgb trains 300 trees with tuned parameters
#sampling of both rows and features for regularization (prevents overfitting), subsample for rows and colsample_bytree for features
#learns slowly but carefully with low learning rate
#outputs probabilities for our two classes
#reg_lambda is L2 regularization term, discourages large weights, model now relies on more features

print("\n" + "="*60)
print("TRAINING MODEL")
print("="*60)
pipe.fit(X_train, Y_train)

print("\n" + "="*60)
print("TEST SET EVALUATION")
print("="*60)

# Predictions
pred = pipe.predict(X_test)
#here our prediction will be marked through at a threshold of 0.5. if not confident enough it will just classify as 0
#we can adjust this threshold later if we want to be more/less conservative
#left is the probability of class 0, right is the probability of class 1
probability = pipe.predict_proba(X_test)[:, 1]

# Calculate metrics
accuracy = accuracy_score(Y_test, pred)
precision = precision_score(Y_test, pred, zero_division=0)
recall = recall_score(Y_test, pred, zero_division=0)
f1 = f1_score(Y_test, pred, zero_division=0)

print(f"\nAccuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")

print("\nClassification Report:")
print(classification_report(Y_test, pred, target_names=['Negative', 'Positive']))

print("\nConfusion Matrix:")
print(confusion_matrix(Y_test, pred))

# Save the model
model_save_path = project_root / "scikit_ensemble" / "best_tfidf_xgboost_model.pkl"
joblib.dump(pipe, model_save_path)
print(f"\n✓ Model saved to: {model_save_path}")

# Overall sentiment prediction
print("\n" + "="*60)
print("OVERALL SENTIMENT PREDICTION")
print("="*60)
prediction = sum(pred) / len(pred)
print(f"Positive prediction ratio: {prediction:.4f}")

if prediction > 0.6:
    print("Overall Sentiment: Positive")
elif prediction < 0.4:
    print("Overall Sentiment: Negative")
else:
    print("Overall Sentiment: Neutral")

# Additional: Show probability distribution
print(f"\nAverage positive probability: {np.mean(probability):.4f}")
print(f"Probability range: [{np.min(probability):.4f}, {np.max(probability):.4f}]")
