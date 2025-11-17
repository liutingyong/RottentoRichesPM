"""
REVIEW SCRAPING SCRIPT
======================

Scrapes reviews from two URL files:
- Training reviews: review_data/training/urls.txt → review_data/training/scraped_data/
- New reviews: review_data/new_reviews/urls.txt → review_data/new_reviews/scraped_data/
"""

import sys 
import asyncio
from playwright.async_api import async_playwright
import re
from pathlib import Path

# =============================================================================
# HELPER FUNCTIONS (already complete - from webscrape.py)
# =============================================================================

def create_safe_filename(title, save_directory):
    safe_title = re.sub(r"[^\w\s-]", "", title).strip().replace(" ", "_")
    safe_title = safe_title[:100] or "untitled"
    return save_directory / f"{safe_title}.txt"

# Get script directory and set up paths
script_dir = Path(__file__).parent
review_data_dir = script_dir / "review_data"

# Two separate directories for training and new reviews
training_dir = review_data_dir / "training"
new_reviews_dir = review_data_dir / "new_reviews"

training_urls_file = training_dir / "urls.txt"
new_reviews_urls_file = new_reviews_dir / "urls.txt"

training_scraped_dir = training_dir / "scraped_data"
new_reviews_scraped_dir = new_reviews_dir / "scraped_data"

# Create directories if they don't exist
training_scraped_dir.mkdir(parents=True, exist_ok=True)
new_reviews_scraped_dir.mkdir(parents=True, exist_ok=True)

# =============================================================================
# ASYNC VERSION OF WEBSCRAPE.PY FUNCTIONS
# =============================================================================

async def capture_screenshot(page, save_directory):
    title = await page.title()
    filename = create_safe_filename(title, save_directory).with_suffix(".png")
    await page.screenshot(path=filename, full_page=True)
    print(f"Screenshot saved as {filename}")

async def save_page_text(page, selector, url, save_directory):
    title = await page.title()
    content = await page.query_selector(selector)
    text = (
        await content.inner_text() if content else "No requested selector found."
    )
    filename = create_safe_filename(title, save_directory)

    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"URL: {url}\n")
        f.write(f"Title: {title}\n\n")
        f.write(text)
    print(f"Data saved as {filename}")

# =============================================================================
# SINGLE URL PROCESSING (like webscrape.py run function)
# =============================================================================

async def process_single_url(browser, url, save_directory, take_screenshot=False):
    """
    Process a single URL - based on webscrape.py run function
    """
    try:
        page = await browser.new_page()
        await page.goto(url, wait_until="domcontentloaded", timeout=30000)

        if take_screenshot:
            await capture_screenshot(page, save_directory)
        else:
            await save_page_text(page, "body", url, save_directory)

        await page.close()
    except Exception as e:
        print(f"⚠️  Error processing {url}: {str(e)}")
        try:
            await page.close()
        except:
            pass

# =============================================================================
# MULTIPLE URL PROCESSING (what you need to implement)
# =============================================================================

async def process_multiple_urls(browser, urls, save_directory, take_screenshot=False):
    """
    Process multiple URLs using async
    """
    for url in urls:
        await process_single_url(browser, url, save_directory, take_screenshot)

# =============================================================================
# MAIN RUN FUNCTION (like webscrape.py)
# =============================================================================

async def run(playwright, urls, save_directory, take_screenshot=False):
    """
    Main run function - based on webscrape.py
    """
    browser = await playwright.chromium.launch(headless=True, channel="chrome")
    await process_multiple_urls(browser, urls, save_directory, take_screenshot)
    await browser.close()

# =============================================================================
# FILE READING (simple implementation)
# =============================================================================

def read_urls_from_file(file_path):
    """
    Read URLs from file, filtering out comments and empty lines
    """
    urls = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                url = line.strip()
                # Skip comments, empty lines, and label markers (0 or 1)
                if url and not url.startswith('#') and url not in ['0', '1']:
                    if url.startswith(('http://', 'https://')):
                        urls.append(url)
    except FileNotFoundError:
        print(f"ERROR: File not found: {file_path}")
        return []
    except Exception as e:
        print(f"ERROR reading file {file_path}: {e}")
        return []
    return urls

# =============================================================================
# MAIN FUNCTION (like webscrape.py)
# =============================================================================

async def main(urls, save_directory, take_screenshot):
    async with async_playwright() as playwright:
        await run(playwright, urls, save_directory, take_screenshot)

if __name__ == "__main__":
    print("="*60)
    print("REVIEW SCRAPING - Training & New Reviews")
    print("="*60)
    
    # Scrape training reviews
    if training_urls_file.exists():
        training_urls = read_urls_from_file(training_urls_file)
        if training_urls:
            print(f"\nScraping {len(training_urls)} training review URLs...")
            take_screenshot = "--screenshot" in sys.argv
            asyncio.run(main(training_urls, training_scraped_dir, take_screenshot))
        else:
            print(f"\n⚠️  No training URLs found in {training_urls_file}")
    else:
        print(f"\n⚠️  Training URLs file not found: {training_urls_file}")
        print(f"   Expected location: {training_urls_file}")
    
    # Scrape new reviews
    if new_reviews_urls_file.exists():
        new_review_urls = read_urls_from_file(new_reviews_urls_file)
        if new_review_urls:
            print(f"\nScraping {len(new_review_urls)} new review URLs...")
            take_screenshot = "--screenshot" in sys.argv
            asyncio.run(main(new_review_urls, new_reviews_scraped_dir, take_screenshot))
        else:
            print(f"\n⚠️  No new review URLs found in {new_reviews_urls_file}")
    else:
        print(f"\n⚠️  New reviews URLs file not found: {new_reviews_urls_file}")
        print(f"   Expected location: {new_reviews_urls_file}")
    
    print("\n" + "="*60)
    print("Scraping complete!")
    print(f"Training data: {training_scraped_dir}")
    print(f"New reviews: {new_reviews_scraped_dir}")
