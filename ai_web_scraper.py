"""
Simple AI Web Scraper
======================
Pipeline:
    1. fetch_page()            -> download raw HTML from a URL
    2. clean_html()             -> strip out noisy tags/elements from the HTML
    3. html_to_markdown()       -> convert cleaned HTML into compact Markdown
    4. answer_query_from_page() -> ask an LLM to answer a query using only that Markdown
    5. ai_web_scraper()         -> glue function that runs steps 1-4 end to end

Setup:
    pip install requests beautifulsoup4 markdownify google-genai ftfy python-dotenv

    Create a .env file next to this script containing:
        GEMINI_API_KEY=your_api_key_here
"""

import argparse          # parse command-line arguments (url, query)
import os              # read environment variables (e.g. GEMINI_API_KEY)
import re               # regex cleanup of Markdown (strip images, collapse whitespace/blank lines)
import requests         # HTTP client used to download the raw webpage HTML

from bs4 import BeautifulSoup  # parse HTML and remove noisy tags/elements; 
from ftfy import fix_text               # "fixes text" - repairs broken/garbled/mojibake text encoding
from markdownify import markdownify as markdownify_html  # converts (cleaned) HTML into Markdown
from google import genai                # client used to call the LLM and get an answer to the user's query
from dotenv import load_dotenv          # loads variables (the API key) from a local .env file into the environment


# ---------------------------------------------------------------------------
# Configuration / setup
# ---------------------------------------------------------------------------

# Load environment variables (expects GEMINI_API_KEY inside a local .env file)
load_dotenv()

# Fail fast if the API key was not found, so errors surface early instead of
# during the first API call.
if not os.getenv("GEMINI_API_KEY"):
    raise ValueError("GEMINI_API_KEY is missing. Add it to your .env file first.")

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Smaller/cheaper model is enough here: the task is "read text, answer a
# question", not complex multi-step reasoning.
MODEL_NAME = "gemini-3.6-flash"


# ---------------------------------------------------------------------------
# 1. Fetch the webpage
# ---------------------------------------------------------------------------
def fetch_page(url: str) -> str:
    """
    Download the raw HTML content from a webpage.

    - Sends a custom User-Agent header so sites that block "headless"
      requests (i.e. requests with no user agent) are more likely to respond.
    - Uses a 15-second timeout so the script never hangs indefinitely on a
      slow/unresponsive server.
    - Calls response.raise_for_status() so the program stops immediately
      with a clear error if the page returns a 4xx/5xx HTTP status.

    Args:
        url: The webpage URL to download.

    Returns:
        The raw HTML of the page as a string.
    """
    headers = {
        "User-Agent": "SimpleAIScraper/1.0"
    }

    response = requests.get(url, headers=headers, timeout=15)
    response.raise_for_status()

    return response.text


# ---------------------------------------------------------------------------
# 2. Clean the HTML
# ---------------------------------------------------------------------------
def clean_html(html: str) -> str:
    """
    Strip a raw HTML page down to just the useful content.

    Steps:
        1. Fix broken/garbled text encoding with ftfy's fix_text().
        2. Parse the HTML with BeautifulSoup.
        3. Remove entire tag types that are almost never useful for Q&A,
           e.g. <script>, <style>, <nav>, <header>, <footer>, <form>, etc.
        4. Remove any remaining tag whose class or id name contains a
           "noise word" (things like "popup", "cookie", "navbar",
           "newsletter", "modal", "login"...), since these are almost
           always layout/marketing clutter rather than page content.
        5. Return just the <body> (or the whole soup if no body exists)
           as a string.

    Args:
        html: Raw HTML string (as returned by fetch_page()).

    Returns:
        A cleaned HTML string with noisy elements removed.
    """
    html = fix_text(html)

    soup = BeautifulSoup(html, "html.parser")

    # Remove obvious noisy tags outright
    for tag in soup([
        "script", "style", "noscript", "svg", "img", "iframe",
        "nav", "header", "footer", "aside", "form", "button"
    ]):
        tag.decompose()

    noise_words = [
        "cursor",
        "modal",
        "popup",
        "floating",
        "signup",
        "login",
        "cookie",
        "banner",
        "navbar",
        "menu",
        "footer",
        "header",
        "subscribe",
        "newsletter",
        "loading",
        "wait",
        "success",
        "auth",
        "w-nav",
        "w-form"
    ]

    # First collect noisy tags (based on class/id names)
    tags_to_remove = []

    for tag in soup.find_all(True):
        if tag.attrs is None:
            continue

        class_value = tag.get("class", [])
        id_value = tag.get("id", "")

        if isinstance(class_value, list):
            class_text = " ".join(class_value).lower()
        else:
            class_text = str(class_value).lower()

        id_text = str(id_value).lower()

        if any(word in class_text or word in id_text for word in noise_words):
            tags_to_remove.append(tag)

    # Then remove them safely (after collecting, to avoid mutating while iterating)
    for tag in tags_to_remove:
        tag.decompose()

    body = soup.body if soup.body else soup

    return str(body)


# ---------------------------------------------------------------------------
# 3. Convert cleaned HTML to Markdown
# ---------------------------------------------------------------------------
def html_to_markdown(html: str) -> str:
    """
    Convert cleaned HTML into compact, readable Markdown.

    Steps:
        1. Convert HTML -> Markdown with markdownify, using ATX-style
           headings (#, ##, ###) and "-" for bullet lists.
        2. Run fix_text() again to clean any leftover encoding issues.
        3. Strip out image Markdown (![alt](src)) since images aren't
           useful for a text-based Q&A task.
        4. Collapse repeated spaces/tabs and excessive blank lines.
        5. Drop known boilerplate lines (form confirmation messages,
           nav labels like "Product"/"Resources"/"Company", etc.) via the
           skip_lines list, and drop any now-empty lines.

    Args:
        html: Cleaned HTML string (as returned by clean_html()).

    Returns:
        A compact Markdown string ready to send to the LLM.
    """
    markdown_text = markdownify_html(
        html,
        heading_style="ATX",
        bullets="-"
    )

    markdown_text = fix_text(markdown_text)

    # Remove image markdown
    markdown_text = re.sub(r"!\[.*?\]\(.*?\)", "", markdown_text)

    # Remove extra spaces and blank lines
    markdown_text = re.sub(r"[ \t]+", " ", markdown_text)
    markdown_text = re.sub(r"\n{3,}", "\n\n", markdown_text)

    lines = []

    skip_lines = [
        "click to try",
        "wait...",
        "you've successfully reserved your spot.",
        "thank you! your submission has been received!",
        "oops! something went wrong while submitting the form.",
        "product",
        "resources",
        "company"
    ]

    for line in markdown_text.splitlines():
        line = line.strip()

        if not line:
            continue

        if line.lower() in skip_lines:
            continue

        lines.append(line)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 4. Ask a user query against the page content
# ---------------------------------------------------------------------------
def answer_query_from_page(markdown_text: str, user_query: str) -> str:
    """
    Send the cleaned page Markdown + a user question to the LLM and get
    back a focused Markdown answer.

    The prompt instructs the model to:
        - Act as a web-scraping assistant.
        - Answer ONLY using the supplied page Markdown (no guessing).
        - Ignore navigation, buttons, popups, and marketing fragments.
        - Say explicitly if the page doesn't contain the answer.
        - Reply in clean Markdown, kept short and focused.

    Args:
        markdown_text: The cleaned page content in Markdown (from html_to_markdown()).
        user_query: The question the user wants answered about the page.

    Returns:
        The model's Markdown-formatted answer as a string.
    """
    prompt = f"""
You are an AI web scraping assistant.

You will receive Markdown extracted from a webpage.

Your task is to answer the user's query using only the useful page content.

User query:
{user_query}

Webpage Markdown:
{markdown_text}

Instructions:
- Return only clean Markdown.
- Use only information from the webpage Markdown.
- Do not invent missing details.
- Ignore navigation links, buttons, CTAs, popups, decorative labels, image captions, and repeated marketing fragments.
- Ignore lines like "Start for free", "Contact Sales", "Your AI Agent", and decorative workflow examples unless they directly answer the query.
- Focus on headings, paragraphs, product descriptions, feature sections, pricing details, documentation text, and factual claims.
- If the page does not contain the answer, say: "The page does not contain this information."
- Keep the answer short, clear, and focused.
"""

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt
    )

    return response.text


# ---------------------------------------------------------------------------
# 5. Full pipeline: fetch -> clean -> markdown -> ask LLM
# ---------------------------------------------------------------------------
def ai_web_scraper(url: str, user_query: str) -> str:
    """
    Run the full AI web scraping pipeline on a single URL and query.

    1. fetch_page()            - download the raw HTML
    2. clean_html()             - remove noisy tags/elements
    3. html_to_markdown()       - convert to compact Markdown
    4. answer_query_from_page() - ask the LLM for a focused answer

    Args:
        url: The webpage to scrape.
        user_query: The question to ask about that webpage's content.

    Returns:
        The LLM's Markdown answer, grounded in the page content.
    """
    raw_html = fetch_page(url)
    cleaned_html = clean_html(raw_html)
    markdown_text = html_to_markdown(cleaned_html)
    answer = answer_query_from_page(markdown_text, user_query)

    return answer


# ---------------------------------------------------------------------------
# Demo / entry point
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch a webpage and answer a question using only its content."
    )
    parser.add_argument("url", nargs="?", help="Webpage URL to scrape.")
    parser.add_argument("query", nargs="?", help="Question to ask about the page.")
    parser.add_argument("-u", "--url", dest="url_flag", help="Webpage URL to scrape.")
    parser.add_argument("-q", "--query", dest="query_flag", help="Question to ask about the page.")
    parser.add_argument(
        "-o", "--output",
        default="ai_scraper_result.md",
        help="Path to save the Markdown answer (default: ai_scraper_result.md).",
    )

    args = parser.parse_args()
    args.url = args.url_flag or args.url
    args.query = args.query_flag or args.query

    if not args.url or not args.query:
        parser.error("both a URL and a query are required (as positional args or --url/--query).")

    return args


if __name__ == "__main__":
    args = parse_args()

    result = ai_web_scraper(args.url, args.query)
    print(result)

    with open(args.output, "w", encoding="utf-8") as file:
        file.write(result)

    print(f"\nMarkdown saved to {args.output}")
