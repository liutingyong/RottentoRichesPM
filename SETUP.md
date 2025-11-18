# Conda Environment Setup

This project uses a conda environment to manage dependencies.

## Quick Start

### 1. Create the Conda Environment

```bash
conda env create -f environment.yml
```

This will create an environment named `rottentoriches` with all required packages.

### 2. Activate the Environment

```bash
conda activate rottentoriches
```

### 3. Install Playwright Browsers

After activating the environment, install the Playwright browsers:

```bash
playwright install chromium
```

Or install all browsers:

```bash
playwright install
```

### 4. Download NLTK Data

Some scripts require NLTK data. Run this once:

```bash
python -c "import nltk; nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('movie_reviews')"
```

## Daily Usage

Every time you work on this project:

```bash
conda activate rottentoriches
```

## Updating the Environment

If you add new dependencies:

1. Update `environment.yml` or `requirements.txt`
2. Update the environment:
   ```bash
   conda env update -f environment.yml --prune
   ```

## Deactivating

When you're done:

```bash
conda deactivate
```

## Removing the Environment

If you need to remove the environment:

```bash
conda env remove -n rottentoriches
```

## Alternative: Using requirements.txt with pip

If you prefer to use pip instead of conda:

```bash
conda create -n rottentoriches python=3.9
conda activate rottentoriches
pip install -r requirements.txt
playwright install chromium
```

## Troubleshooting

### Playwright not found
Make sure you've installed the browsers:
```bash
playwright install chromium
```

### NLTK data missing
Download required NLTK data:
```bash
python -c "import nltk; nltk.download('all')"
```

### Import errors
Make sure the environment is activated:
```bash
conda activate rottentoriches
```

