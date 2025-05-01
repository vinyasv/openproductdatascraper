# E-commerce Product Scraper using Crawl4AI

This project scrapes product information from e-commerce sites using the `crawl4ai` library, extracts structured data with OpenAI's GPT models, generates embeddings, and stores the results in a Pinecone vector database.

## Overview

The scraper starts by fetching product URLs from a target website's sitemap (or other URL source). It then crawls these pages in parallel, executing JavaScript to handle dynamic content loading. The raw HTML is converted to Markdown, which is then fed to an OpenAI model (e.g., GPT-4o) along with a structured schema (Pydantic model) to extract key product details like name, price, description, ingredients, and usage instructions.

Finally, it generates embeddings for the extracted text using an OpenAI embedding model (e.g., `text-embedding-3-small`) and upserts the product data (as metadata) along with the embeddings into a specified Pinecone index.

## Features

*   **Sitemap Parsing:** Automatically fetches and parses product sitemaps (adaptable to other URL sources).
*   **Parallel Crawling:** Uses `crawl4ai`'s `arun_many` for efficient, concurrent crawling.
*   **JavaScript Execution:** Scrolls pages and waits to handle dynamically loaded content.
*   **Markdown Conversion:** Converts HTML to Markdown before LLM processing.
*   **LLM-Powered Extraction:** Uses OpenAI's GPT models for structured data extraction based on a Pydantic schema.
*   **Embedding Generation:** Creates vector embeddings using OpenAI models.
*   **Vector Storage:** Upserts data and embeddings to a Pinecone index.
*   **Resume Capability:** Logs processed URLs (`processed_urls.log`) to avoid re-crawling.
*   **Error Handling:** Includes retries for OpenAI calls and handles Pinecone exceptions.
*   **Configuration:** Uses a `.env` file for API keys and Pinecone settings.
*   **Optional JSON Output:** Can save extracted data locally to a JSON file.

## Technology Stack

*   Python 3.10+
*   Crawl4AI
*   OpenAI API (GPT-4o, text-embedding-3-small)
*   Pinecone
*   Pydantic
*   python-dotenv
*   Requests
*   lxml
*   psutil

## Setup

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/vinyasv/openproductdatascraper.git
    ```
    (You will likely be inside the `openproductdatascraper` directory after cloning)

2.  **Create Virtual Environment:**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure Environment Variables:**
    *   Rename `.env.example` to `.env`.
    *   Open the `.env` file and fill in your actual API keys and Pinecone index name:
        ```env
        OPENAI_API_KEY="your_openai_api_key_here"
        PINECONE_API_KEY="your_pinecone_api_key_here"
        PINECONE_INDEX_NAME="your_pinecone_index_name_here"
        ```

5.  **Set up Pinecone Index:**
    *   Ensure you have a Pinecone account ([pinecone.io](https://www.pinecone.io/)).
    *   Create a Pinecone index with the name specified in your `.env` file.
    *   **Crucially**, ensure the index dimension matches the OpenAI embedding model used (the default is `text-embedding-3-small`, which has a dimension of **1536**).

## Usage

Run the scraper from the command line using the following structure:

```bash
export PYTHONPATH="$PYTHONPATH:/path/to/your/project/root" && python -m src.scraper [OPTIONS]
```
*(Note: Adjust the `PYTHONPATH` export for your specific project location if needed, or run from the directory containing the `src` folder. You may need to rename `src/sephora_scraper.py` to `src/scraper.py` or update the command).* 

**Common Options:**

*   `--limit <N>`: Limit the number of new product URLs to process (e.g., `--limit 10`). Defaults to processing all found URLs.
*   `--send-to-pinecone`: Add this flag to enable upserting data to your Pinecone index. (Required for Pinecone storage).
*   `--output-file <filename.json>`: Optionally save the extracted data to a local JSON file (e.g., `--output-file products_output.json`).
*   `--log-level <LEVEL>`: Set the logging level (DEBUG, INFO, WARNING, ERROR). Defaults to INFO.
*   `--resume-file <filepath>`: Specify a different file path for tracking processed URLs (defaults to `processed_urls.log`).
*   `--max-concurrent <N>`: Set the maximum number of concurrent crawl requests (defaults to 10).

**Example (Crawl 50 products and send to Pinecone):**

```bash
export PYTHONPATH="$PYTHONPATH:/path/to/your/project/root" && python -m src.scraper --limit 50 --log-level INFO --send-to-pinecone
```

## Disclaimer

Web scraping can be resource-intensive for target websites. Please ensure you are respecting the target website's `robots.txt` file and Terms of Service.

*   Use the `--limit` flag during testing.
*   Avoid running the scraper excessively frequently.
*   The creators of this tool are not responsible for any misuse or violations of terms of service.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. 