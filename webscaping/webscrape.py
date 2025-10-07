import sys
import re

from playwright.sync_api import sync_playwright

def capture_screenshot(page):
    page.screenshot(path="screenshot.png", full_page=True)
    print("Screenshot saved as screenshot.png")

def save_page_text(page, selector):
    title = page.title()
    content = page.query_selector(selector)
    text = (
        content.inner_text() if content else "No requested selector found."
    )
    filename = create_safe_filename(title)

    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"Title: {title}\n\n")
        f.write(text)
    print(f"Data saved as {filename}")

def create_safe_filename(title):
    safe_title = re.sub(r"[^\w\s-]", "", title).strip().replace(" ", "_")
    return f"{safe_title}.txt"


def run(playwright, url, take_screenshot=False):
    browser = playwright.chromium.launch()
    page = browser.new_page()
    page.goto(url)

    if take_screenshot:
        capture_screenshot(page)
    else:
        save_page_text(page, "body")

    browser.close()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python playwright.py <URL> [--screenshot]")
        sys.exit(1)

    url = sys.argv[1]
    take_screenshot = "--screenshot" in sys.argv

    with sync_playwright() as playwright:
        run(playwright, url, take_screenshot)