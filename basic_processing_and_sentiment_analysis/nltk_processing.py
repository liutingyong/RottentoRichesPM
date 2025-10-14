import nltk
from pathlib import Path
nltk.download('vader_lexicon')
from nltk.sentiment import SentimentIntensityAnalyzer

#one issue that im worried about is the amount of unrelated data that we have. we need to filter more, but will worry about that later
analyzer = SentimentIntensityAnalyzer()

scores = []

directory = Path("src/webscraping/scraped_data")  
for filename in directory.glob("*.txt"):
    #r is read mode
    with open(filename, "r", encoding="utf-8") as file:
        text = file.read()
        print(f"Processing file: {filename.name}")
        score = analyzer.polarity_scores(text)
        print(f"Sentiment score for {filename.name}: {score}")
        scores.append((filename.name, score))

print(f"All sentiment scores: {scores}")