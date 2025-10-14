import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
from pathlib import Path
#should we discard some stopwords?

#punkt is a pretrained sentence/word tokenizer, splits better (knows abbreviations, punctuation, etc.)
#nltk.download('punkt')
#stopwords are irrelevant words that don't contribute to meaning but important for grammar like i, me, the, etc.
#nltk.download('stopwords')
#synonym/antonym 
#nltk.download('wordnet') #wordnet is lexical database for english

def preprocess_text(text):
    # Tokenize the text into words
    tokens = word_tokenize(text.lower())
    
    # Remove punctuation and stopwords
    #stopwords are actually in multiple languages
    #set for faster lookup
    #string.punctuation is a string containing all punctuation characters
    stop_words = set(stopwords.words('english'))
    tokens = [t for t in tokens if t not in string.punctuation]
    tokens = [t for t in tokens if t not in stop_words]
    
    # Lemmatize the words (aka converting word to base def)
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    
    #return ' '.join(tokens)
    return tokens

#testing
directory = Path("src/webscraping/scraped_data")

for filename in directory.glob("*.txt"):
    with open(filename, "r", encoding="utf-8") as file:
        text = file.read()
        print(f"Data preprocessing file: {filename.name}")
        data = preprocess_text(text)

#print(data)

def extract_features(words):
    return {word: True for word in words}

from nltk.corpus import movie_reviews
#nltk.download('movie_reviews')
import random

#categories are positiev or negative
docs = []
for category in movie_reviews.categories():
    #this part gets all the fileids of a specifi ccategory
    for fileid in movie_reviews.fileids(category):
        #movie_reviews.words(fileid) gets all the words in that file (tokenized)
        #docs is a list of tuples, where each tuple contains a list of words and their corresponding category (pos, neg)
        docs.append((list(movie_reviews.words(fileid)), category))

random.shuffle(docs)
featuresets = []
for (words, label) in docs:
    try:
        featuresets.append((extract_features(preprocess_text("".join(words))), label))
    except Exception as e:
        print(f"Error processing {words}: {e}")
#featuresets = [(extract_features(preprocess_text(words)), label) for (words, label) in docs]
#do we split in half or just split by first 1500 and rest?
training_set = featuresets[:1500]
testing_set = featuresets[1500:]

#naivebayesclassifier, supervised ML (learns from labeled data)
#bayes theorem for probability calcs
#naive bc assumes independence between features
#counts how often each word appears in each category (pos, neg)
from nltk import NaiveBayesClassifier
print(f'# of training examples: {len(training_set)}')
print(f'# of featuresets: {len(featuresets)}')
classifier = NaiveBayesClassifier.train(training_set)
#test accuracy
print(f"Classifier accuracy: {nltk.classify.accuracy(classifier, testing_set)}")


classifications = []
for filename in directory.glob("*.txt"):
    with open(filename, "r", encoding="utf-8") as file:
        text = file.read()
        print(f"Classifying file: {filename.name}")
        features = extract_features(preprocess_text(text))
        label = classifier.classify(features)
        if label == 'pos':
            classifications.append(1)
        else:
            classifications.append(0)
        print(f"Sentiment for {filename.name}: {label}")

#we can improve accuracy by using more data, better preprocessing, or more advanced models

#change our current data to be more relevant to our use case so we can make actual bets
if len(classifications) == 0:
    print("No files were successfully processed. Check if the scraped_data directory exists and contains .txt files.")
else:
    percentage_pos = classifications.count(1) / len(classifications)
    print(f"Percentage of positive sentiment: {percentage_pos:.2%}")
    if percentage_pos > 0.6:
        print("Overall Sentiment: Positive")
    elif percentage_pos < 0.4:
        print("Overall Sentiment: Negative")
    else:
        print("Overall Sentiment: Neutral")


class SentimentAnalyzer:
    """
    A class for analyzing sentiment of text data using NLTK's Naive Bayes classifier
    """
    
    def __init__(self):
        """Initialize the sentiment analyzer and train the classifier"""
        self.classifier = None
        self._train_classifier()
    
    def _train_classifier(self):
        """Train the Naive Bayes classifier on movie reviews data"""
        print("Training sentiment classifier...")
        
        # Load and prepare training data
        docs = []
        for category in movie_reviews.categories():
            for fileid in movie_reviews.fileids(category):
                docs.append((list(movie_reviews.words(fileid)), category))
        
        random.shuffle(docs)
        featuresets = []
        for (words, label) in docs:
            try:
                featuresets.append((extract_features(preprocess_text("".join(words))), label))
            except Exception as e:
                print(f"Error processing {words}: {e}")
        
        # Split into training and testing sets
        training_set = featuresets[:1500]
        testing_set = featuresets[1500:]
        
        # Train the classifier
        self.classifier = NaiveBayesClassifier.train(training_set)
        
        # Print accuracy
        accuracy = nltk.classify.accuracy(self.classifier, testing_set)
        print(f"Sentiment classifier trained with {accuracy:.2%} accuracy")
    
    def analyze_sentiment(self, text: str) -> dict:
        """
        Analyze sentiment of a single text
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with sentiment analysis results
        """
        if not self.classifier:
            raise ValueError("Classifier not trained")
        
        # Preprocess and extract features
        features = extract_features(preprocess_text(text))
        
        # Get classification
        label = self.classifier.classify(features)
        
        # Get probability distribution
        prob_dist = self.classifier.prob_classify(features)
        
        # Calculate confidence as the difference between positive and negative probabilities
        # This gives us a measure of how decisive the classification is
        positive_prob = prob_dist.prob('pos')
        negative_prob = prob_dist.prob('neg')
        confidence = abs(positive_prob - negative_prob)
        
        return {
            'label': label,
            'confidence': confidence,
            'positive_prob': positive_prob,
            'negative_prob': negative_prob
        }
    
    def analyze_multiple_texts(self, texts: list) -> dict:
        """
        Analyze sentiment of multiple texts and return aggregate results
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            Dictionary with aggregate sentiment analysis
        """
        if not texts:
            return {'overall_sentiment': 'neutral', 'confidence': 0.0, 'positive_percentage': 0.5}
        
        results = []
        positive_count = 0
        
        for text in texts:
            result = self.analyze_sentiment(text)
            results.append(result)
            if result['label'] == 'pos':
                positive_count += 1
        
        # Avoid division by zero
        if len(texts) > 0:
            positive_percentage = positive_count / len(texts)
        else:
            positive_percentage = 0.5  # Default to neutral
        
        if len(results) > 0:
            avg_confidence = sum(r['confidence'] for r in results) / len(results)
        else:
            avg_confidence = 0.0
        
        # Determine overall sentiment (more sensitive thresholds)
        if positive_percentage > 0.55:  # Lowered from 0.6
            overall_sentiment = 'positive'
        elif positive_percentage < 0.45:  # Raised from 0.4
            overall_sentiment = 'negative'
        else:
            overall_sentiment = 'neutral'
        
        return {
            'overall_sentiment': overall_sentiment,
            'confidence': avg_confidence,
            'positive_percentage': positive_percentage,
            'individual_results': results,
            'total_texts': len(texts)
        }
    
    def get_betting_recommendation(self, texts: list, market_title: str = "") -> dict:
        """
        Get a betting recommendation based on sentiment analysis
        
        Args:
            texts: List of texts to analyze
            market_title: Title of the market for context
            
        Returns:
            Dictionary with betting recommendation
        """
        sentiment_results = self.analyze_multiple_texts(texts)
        
        # Determine betting side based on sentiment
        if sentiment_results['overall_sentiment'] == 'positive':
            side = 'yes'
            reasoning = f"Positive sentiment ({sentiment_results['positive_percentage']:.1%} positive) suggests bullish outlook"
        elif sentiment_results['overall_sentiment'] == 'negative':
            side = 'no'
            reasoning = f"Negative sentiment ({sentiment_results['positive_percentage']:.1%} positive) suggests bearish outlook"
        else:
            side = None  # No clear recommendation for neutral sentiment
            reasoning = f"Neutral sentiment ({sentiment_results['positive_percentage']:.1%} positive) - no clear betting direction"
        
        # Calculate confidence based on both sentiment strength and classifier confidence
        sentiment_strength = abs(sentiment_results['positive_percentage'] - 0.5) * 2  # 0 to 1 scale
        confidence = (sentiment_strength + sentiment_results['confidence']) / 2
        
        return {
            'side': side,
            'confidence': confidence,
            'reasoning': reasoning,
            'sentiment_data': sentiment_results,
            'market_title': market_title
        }


# Create a global instance for easy import
sentiment_analyzer = SentimentAnalyzer()