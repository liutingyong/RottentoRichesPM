import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import NaiveBayesClassifier
import string
import pickle
import os
from pathlib import Path

class OptimizedSentimentClassifier:
    """
    Optimized sentiment classifier that caches downloads and trained models
    """
    
    def __init__(self):
        self.stop_words = None
        self.lemmatizer = None
        self.classifier = None
        self.classifier_path = "sentiment_classifier.pkl"
        self._setup_nltk()
    
    def _setup_nltk(self):
        """Setup NLTK resources (only download if not already present)"""
        try:
            # Check if resources are already downloaded
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
            nltk.data.find('corpora/wordnet')
            nltk.data.find('corpora/movie_reviews')
        except LookupError:
            print("Downloading NLTK resources (one-time setup)...")
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
            nltk.download('movie_reviews', quiet=True)
        
        # Initialize components
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
    
    def preprocess_text_fast(self, text):
        """Optimized text preprocessing"""
        # Tokenize and lowercase in one step
        tokens = word_tokenize(text.lower())
        
        # Filter tokens in one pass
        filtered_tokens = []
        for token in tokens:
            if (token not in string.punctuation and 
                token not in self.stop_words and 
                len(token) > 2):  # Skip very short words
                filtered_tokens.append(self.lemmatizer.lemmatize(token))
        
        return filtered_tokens
    
    def extract_features_fast(self, words):
        """Optimized feature extraction"""
        return {word: True for word in words if len(word) > 2}
    
    def train_classifier(self):
        """Train classifier and cache it"""
        if os.path.exists(self.classifier_path):
            print("Loading cached classifier...")
            with open(self.classifier_path, 'rb') as f:
                self.classifier = pickle.load(f)
            return
        
        print("Training classifier (this may take a moment)...")
        
        # Import movie_reviews here to avoid import issues
        from nltk.corpus import movie_reviews
        
        # Use smaller dataset for faster training
        docs = []
        for category in ['pos', 'neg']:  # Only use pos/neg categories
            fileids = movie_reviews.fileids(category)[:500]  # Use only 500 files per category
            for fileid in fileids:
                words = movie_reviews.words(fileid)
                docs.append((words, category))
        
        # Shuffle and create featuresets
        import random
        random.shuffle(docs)
        
        featuresets = []
        for words, label in docs:
            try:
                processed_words = self.preprocess_text_fast(" ".join(words))
                features = self.extract_features_fast(processed_words)
                featuresets.append((features, label))
            except Exception as e:
                continue  # Skip problematic entries
        
        # Split data
        split_point = int(len(featuresets) * 0.8)
        training_set = featuresets[:split_point]
        testing_set = featuresets[split_point:]
        
        # Train classifier
        self.classifier = NaiveBayesClassifier.train(training_set)
        
        # Test accuracy
        accuracy = nltk.classify.accuracy(self.classifier, testing_set)
        print(f"Classifier accuracy: {accuracy:.3f}")
        
        # Cache the classifier
        with open(self.classifier_path, 'wb') as f:
            pickle.dump(self.classifier, f)
        print("Classifier cached for future use")
    
    def classify_text(self, text):
        """Classify a single text"""
        if self.classifier is None:
            self.train_classifier()
        
        processed_words = self.preprocess_text_fast(text)
        features = self.extract_features_fast(processed_words)
        label = self.classifier.classify(features)
        
        return 1 if label == 'pos' else 0
    
    def analyze_scraped_data(self, directory_path="../../src/webscraping/scraped_data"):
        """Analyze all scraped data files"""
        directory = Path(directory_path)
        
        if not directory.exists():
            print(f"Directory {directory_path} not found")
            return None
        
        classifications = []
        file_results = []
        
        for filename in directory.glob("*.txt"):
            try:
                with open(filename, "r", encoding="utf-8") as file:
                    text = file.read()
                    sentiment = self.classify_text(text)
                    classifications.append(sentiment)
                    file_results.append((filename.name, sentiment))
                    print(f"{filename.name}: {'Positive' if sentiment == 1 else 'Negative'}")
            except Exception as e:
                print(f"Error processing {filename.name}: {e}")
        
        if classifications:
            percentage_pos = sum(classifications) / len(classifications)
            print(f"\nPercentage of positive sentiment: {percentage_pos:.2%}")
            
            if percentage_pos > 0.6:
                overall_sentiment = "Positive"
            elif percentage_pos < 0.4:
                overall_sentiment = "Negative"
            else:
                overall_sentiment = "Neutral"
            
            print(f"Overall Sentiment: {overall_sentiment}")
            return {
                'percentage_positive': percentage_pos,
                'overall_sentiment': overall_sentiment,
                'file_results': file_results,
                'total_files': len(classifications)
            }
        
        return None

# Quick usage function
def quick_sentiment_analysis():
    """Quick sentiment analysis of scraped data"""
    classifier = OptimizedSentimentClassifier()
    return classifier.analyze_scraped_data()

if __name__ == "__main__":
    # Run the analysis
    results = quick_sentiment_analysis()
    if results:
        print(f"\nAnalysis complete!")
        print(f"Files analyzed: {results['total_files']}")
        print(f"Overall sentiment: {results['overall_sentiment']}")
