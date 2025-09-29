"""
WEBSCRAPING TUTORIAL - Based on webscrape.py
===========================================

Learn webscraping by filling in simple blanks using webscrape.py as reference.
Only basic async knowledge needed - most systems are set up for you.

Complete in ~15 minutes.
"""

import sys 
import asyncio
from playwright.async_api import async_playwright
import re
from pathlib import Path

# =============================================================================
# HELPER FUNCTIONS (already complete - from webscrape.py)
# =============================================================================

def create_safe_filename(title):
    safe_title = re.sub(r"[^\w\s-]", "", title).strip().replace(" ", "_")
    safe_title = safe_title[:100] or "untitled"
    return save_directory / f"{safe_title}.txt"

save_directory = Path(__file__).parent / "scraped_data"
save_directory.mkdir(parents=True, exist_ok=True)

# =============================================================================
# ASYNC VERSION OF WEBSCRAPE.PY FUNCTIONS
# =============================================================================

async def capture_screenshot(page):
    title = await page.title()
    filename = create_safe_filename(title).with_suffix(".png")
    await page.screenshot(path=filename, full_page=True)
    print(f"Screenshot saved as {filename}")

async def save_page_text(page, selector, url):
    title = await page.title()
    content = await page.query_selector(selector)
    text = (
        await content.inner_text() if content else "No requested selector found."
    )
    filename = create_safe_filename(title)

    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"URL: {url}\n")
        f.write(f"Title: {title}\n\n")
        f.write(text)
    print(f"Data saved as {filename}")

# =============================================================================
# SINGLE URL PROCESSING (like webscrape.py run function)
# =============================================================================

async def process_single_url(browser, url, take_screenshot=False):
    """
    Process a single URL - based on webscrape.py run function
    """
    page = await browser.new_page()
    await page.goto(url, wait_until="domcontentloaded", timeout=30000)

    if take_screenshot:
        await capture_screenshot(page)
    else:
        await save_page_text(page, "body", url)

    await page.close()

# =============================================================================
# MULTIPLE URL PROCESSING (what you need to implement)
# =============================================================================

async def process_multiple_urls(browser, urls, take_screenshot=False):
    """
    TODO: Process multiple URLs using async
    Hint: Look at webscrape.py run function and make it work for multiple URLs
    You need to call process_single_url for each URL
    """
    # TODO: Loop through urls and call process_single_url for each
    # Hint: Use a for loop like: for url in urls:
    pass

# =============================================================================
# MAIN RUN FUNCTION (like webscrape.py)
# =============================================================================

async def run(playwright, urls, take_screenshot=False):
    """
    Main run function - based on webscrape.py
    """
    browser = await playwright.chromium.launch(headless=True, channel="chrome")
    
    # TODO: Call your process_multiple_urls function here
    # Hint: await process_multiple_urls(browser, urls, take_screenshot)
    
    await browser.close()

# =============================================================================
# FILE READING (simple implementation)
# =============================================================================

def read_urls_from_file(file_path):
    """
    TODO: Read URLs from file
    Hint: Open file, read lines, strip whitespace, filter out comments and empty lines
    """
    urls = []
    # TODO: Open file and read URLs
    # Hint: with open(file_path, 'r') as f:
    #       for line in f:
    #           url = line.strip()
    #           if url and not url.startswith('#') and url.startswith(('http://', 'https://')):
    #               urls.append(url)
    return urls

# =============================================================================
# MAIN FUNCTION (like webscrape.py)
# =============================================================================

async def main(urls, take_screenshot):
    async with async_playwright() as playwright:
        await run(playwright, urls, take_screenshot)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python webscraping_tutorial.py <URL> [--screenshot]")
        print("       python webscraping_tutorial.py file <file_path>")
        sys.exit(1)

    if sys.argv[1] == "file":
        file_path = sys.argv[2]
        urls = read_urls_from_file(file_path)
        take_screenshot = "--screenshot" in sys.argv
        asyncio.run(main(urls, take_screenshot))
    else:
        urls = sys.argv[1:]
        take_screenshot = "--screenshot" in urls
        urls = [url for url in urls if url != "--screenshot"]
        asyncio.run(main(urls, take_screenshot))