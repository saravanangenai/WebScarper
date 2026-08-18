# AI Web Scraper

A simple AI-powered web scraper that fetches a webpage, strips out the noise
(nav bars, popups, scripts, ads), converts what's left to Markdown, and asks
an LLM to answer a question using only that page's content.

## How it works

The pipeline in [ai_web_scraper.py](ai_web_scraper.py) runs in four steps:

1. **`fetch_page(url)`** — downloads the raw HTML with a custom User-Agent
   and a 15-second timeout.
2. **`clean_html(html)`** — repairs garbled text encoding, then strips
   scripts, styles, nav/header/footer/forms, and any element whose class/id
   looks like layout or marketing clutter (popups, cookie banners, login
   forms, newsletters, etc.).
3. **`html_to_markdown(html)`** — converts the cleaned HTML to compact
   Markdown, removes images, collapses extra whitespace, and drops known
   boilerplate lines.
4. **`answer_query_from_page(markdown, query)`** — sends the Markdown and
   your question to an LLM, instructed to answer only from the page content
   and say so explicitly if the answer isn't there.

`ai_web_scraper(url, user_query)` glues all four steps together.

## Setup

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Create a `.env` file next to the script with your Gemini API key:

   ```
   GEMINI_API_KEY=your_api_key_here
   ```

   Get a key from [Google AI Studio](https://aistudio.google.com/apikey).

## Usage

Run from the command line, passing the URL and your question as positional
arguments (or `--url`/`--query` flags). The answer prints to the terminal and
is saved to `ai_scraper_result.md` by default:

```bash
python ai_web_scraper.py "https://example.com" "What does this page offer?"

# or with flags, plus a custom output path
python ai_web_scraper.py --url "https://example.com" --query "What does this page offer?" --output answer.md
```

Or import it and use it in your own code:

```python
from ai_web_scraper import ai_web_scraper

answer = ai_web_scraper("https://example.com", "What does this page offer?")
print(answer)
```

## Notes

- The LLM used is Google Gemini, via the `google-genai` SDK. The model is set
  via `MODEL_NAME` in [ai_web_scraper.py](ai_web_scraper.py) (default: `gemini-3.6-flash`) — a small/cheap
  model is sufficient since the task is reading text and answering a
  question, not complex reasoning.
- The `.env` file (and your API key) should never be committed to version
  control.
