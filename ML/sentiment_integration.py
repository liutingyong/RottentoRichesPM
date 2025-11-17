"""
Integration module for sentiment analysis with betting system
"""

import sys
import os
from pathlib import Path

# Add the sentiment analysis directory to path
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

from sentiment_classifier_optimized import OptimizedSentimentClassifier

def get_sentiment_recommendation(market_data, scraped_data_path="src/webscraping/scraped_data"):
    """
    Get betting recommendation based on sentiment analysis
    
    Args:
        market_data: Dictionary of market data from main.py
        scraped_data_path: Path to scraped data directory
    
    Returns:
        Dictionary with sentiment analysis results and recommendations
    """
    
    # Initialize classifier
    classifier = OptimizedSentimentClassifier()
    
    # Analyze scraped data
    sentiment_results = classifier.analyze_scraped_data(scraped_data_path)
    
    if not sentiment_results:
        return {
            'sentiment_score': 0.5,  # Neutral
            'recommendation': 'no_bet',
            'confidence': 0.0,
            'reasoning': 'No sentiment data available'
        }
    
    sentiment_score = sentiment_results['percentage_positive']
    overall_sentiment = sentiment_results['overall_sentiment']
    
    # Generate betting recommendations based on sentiment
    recommendations = []
    
    for ticker, data in market_data.items():
        # Simple sentiment-based logic
        if sentiment_score > 0.7:  # Very positive sentiment
            if data.get('yes_bid', 0) > 30:  # Strong yes bid
                recommendations.append({
                    'ticker': ticker,
                    'side': 'yes',
                    'confidence': min(sentiment_score, 0.8),
                    'reasoning': f'Strong positive sentiment ({sentiment_score:.1%}) suggests bullish outcome',
                    'market_title': data.get('title', 'Unknown'),
                    'current_price': data.get('yes_bid', 0)
                })
        elif sentiment_score < 0.3:  # Very negative sentiment
            if data.get('no_bid', 0) > 30:  # Strong no bid
                recommendations.append({
                    'ticker': ticker,
                    'side': 'no',
                    'confidence': min(1 - sentiment_score, 0.8),
                    'reasoning': f'Strong negative sentiment ({sentiment_score:.1%}) suggests bearish outcome',
                    'market_title': data.get('title', 'Unknown'),
                    'current_price': data.get('no_bid', 0)
                })
    
    return {
        'sentiment_score': sentiment_score,
        'overall_sentiment': overall_sentiment,
        'recommendations': recommendations,
        'total_files_analyzed': sentiment_results['total_files']
    }

def quick_sentiment_check():
    """Quick sentiment check for testing"""
    classifier = OptimizedSentimentClassifier()
    results = classifier.analyze_scraped_data()
    
    if results:
        print(f"Sentiment Analysis Results:")
        print(f"Files analyzed: {results['total_files']}")
        print(f"Positive sentiment: {results['percentage_positive']:.1%}")
        print(f"Overall sentiment: {results['overall_sentiment']}")
        return results
    else:
        print("No sentiment data available")
        return None

if __name__ == "__main__":
    # Test the sentiment analysis
    quick_sentiment_check()

