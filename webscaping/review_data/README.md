# Review Data Organization

This folder contains URLs for scraping reviews, organized by purpose.

## Folder Structure

```
review_data/
├── training_urls/
│   └── training_reviews.txt    # URLs of reviews for TRAINING (labeled)
└── new_review_urls/
    └── new_reviews.txt         # URLs of NEW reviews for PREDICTION (unlabeled)
```

## How to Use

### Step 1: Add URLs

**Training URLs** (`training_urls/training_reviews.txt`):
- Add URLs of reviews you want to use for training
- These should be reviews where you know the sentiment
- You'll label these manually (0=negative, 1=positive)

**New Review URLs** (`new_review_urls/new_reviews.txt`):
- Add URLs of NEW reviews you want to predict sentiment for
- These are reviews you haven't labeled yet
- The model will predict their sentiment

### Step 2: Scrape Reviews

Run the scraping script:
```bash
python src/webscraping/scrape_reviews.py
```

This will:
- Scrape training URLs → save to `scraped_data/training/`
- Scrape new review URLs → save to `scraped_data/new_reviews/`

### Step 3: Label Training Data

After scraping training reviews:
1. Open `scraped_data/training/` folder
2. Review each file and determine sentiment
3. Update labels in `hyperparameter_tuning_tutorial.py`:
   ```python
   y = [0, 1, 1, 0, 1, ...]  # Match order of files in training/
   ```

### Step 4: Train Model

Run the hyperparameter tuning tutorial:
```bash
python src/scikit_ensemble/hyperparameter_tuning_tutorial.py
```

This trains on labeled reviews in `scraped_data/training/`

### Step 5: Predict on New Reviews

Use the trained model to predict sentiment of new reviews in `scraped_data/new_reviews/`

## Example Workflow

1. Add 20 review URLs to `training_urls/training_reviews.txt`
2. Run `scrape_reviews.py` → 20 files saved to `scraped_data/training/`
3. Label each file → `y = [1, 0, 1, 1, 0, ...]` (20 labels)
4. Train model on labeled training data
5. Add 5 new review URLs to `new_review_urls/new_reviews.txt`
6. Run `scrape_reviews.py` → 5 files saved to `scraped_data/new_reviews/`
7. Use trained model to predict sentiment of the 5 new reviews
8. Make betting decisions based on predictions!

