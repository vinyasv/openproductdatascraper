"""
3-crawl_docs_FAST.py -> sephora_scraper.py
---------------------
Batch-crawls a list of documentation URLs in parallel using Crawl4AI's arun_many and a memory-adaptive dispatcher.
Tracks memory usage, prints a summary of successes/failures, and is suitable for large-scale doc scraping jobs.
Usage: Call main() or run as a script. Adjust max_concurrent for parallelism.
"""
import os
import sys
import psutil
import asyncio
import requests
import argparse # Added for command-line arguments
import json # Added for JSON output
import time # Added for Pinecone potentially
import hashlib # Added for Pinecone ID generation
from typing import List, Optional, Set, Tuple, Dict # Added Dict
from xml.etree import ElementTree # Standard library XML parsing
import lxml.etree # Added for potentially more robust XML/HTML parsing
import logging

from crawl4ai import (
    AsyncWebCrawler, 
    BrowserConfig, 
    CrawlerRunConfig, 
    CacheMode, 
    MemoryAdaptiveDispatcher, 
    CrawlResult,
    DefaultMarkdownGenerator # Import the generator
)

# Import local processing and model modules
from .processing import extract_product_data
from .models import ProductData
from .config import current_settings # Added for config access
from .llm import get_openai_client # Added for embedding client

# Added Pinecone imports
from pinecone import Pinecone, Index
from pinecone.exceptions import PineconeException
from openai import AsyncOpenAI, OpenAIError # Re-added OpenAIError
# -----------------------

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Resume Logic --- #
def load_processed_urls(filepath: str) -> Set[str]:
    """Loads previously processed URLs/IDs from a file."""
    processed = set()
    if not os.path.exists(filepath):
        logger.info(f"Resume file '{filepath}' not found. Starting fresh.")
        return processed
    try:
        with open(filepath, 'r') as f:
            for line in f:
                processed.add(line.strip())
        logger.info(f"Loaded {len(processed)} processed URLs/IDs from '{filepath}'.")
    except IOError as e:
        logger.error(f"Error reading resume file '{filepath}': {e}. Starting fresh.")
        processed.clear() # Ensure we start fresh if read fails
    return processed

def append_processed_batch(filepath: str, batch_data: List[Tuple[str, List[float], dict]]):
    """Appends successfully processed IDs to the resume file."""
    try:
        with open(filepath, 'a') as f:
            for item_id, _, _ in batch_data:
                f.write(f"{item_id}\n")
        logger.debug(f"Appended {len(batch_data)} IDs to resume file '{filepath}'.")
    except IOError as e:
        logger.error(f"Error writing to resume file '{filepath}': {e}")
# --------------------

# --- Process Page Function (Modified for Task 9.1) ---
async def process_page(result: CrawlResult) -> Optional[ProductData]:
    """Callback function to process a single crawled page using Markdown.

    Args:
        result: The CrawlResult object from Crawl4AI.

    Returns:
        A ProductData object if extraction is successful, otherwise None.
    """
    logger.info(f"Processing page: {result.url} (Status: {result.status_code})")

    # Use generated Markdown instead of HTML
    markdown_content = getattr(result.markdown, "raw_markdown", None)

    # Skip when crawl failed or no Markdown content is available
    if not result.success or not markdown_content:
        logger.warning(
            f"Skipping processing for {result.url}: Crawl failed or no Markdown content available."
        )
        return None

    logger.debug(f"Attempting LLM data extraction for {result.url} using Markdown")
    # Pass the markdown content to the extraction function
    product_data = await extract_product_data(markdown_content, result.url)

    if product_data:
        logger.info(f"Successfully extracted data for: {result.url}")
    else:
        logger.warning(f"Failed to extract data for: {result.url}")

    return product_data
# -----------------------------------------------------

# --- Pinecone Utility Functions (Added Here) ---

_pinecone_index: Index = None
_embedding_client: AsyncOpenAI = None

def initialize_pinecone() -> Optional[Index]:
    """Initializes the Pinecone client and connects to the specified index."""
    global _pinecone_index
    if _pinecone_index is not None:
        logger.debug("Pinecone index already initialized.")
        return _pinecone_index

    if not current_settings or not current_settings.pinecone_api_key or not current_settings.pinecone_index_name:
        logger.error("Pinecone API key or index name not found in settings. Cannot initialize.")
        return None

    try:
        logger.info(f"Initializing Pinecone client for index '{current_settings.pinecone_index_name}'...")
        pc = Pinecone(api_key=current_settings.pinecone_api_key)
        index_name = current_settings.pinecone_index_name
        # Correct way to check if index exists in v3+
        existing_indexes = pc.list_indexes()
        if index_name not in [idx.name for idx in existing_indexes.indexes]:
             logger.error(f"Pinecone index '{index_name}' does not exist. Please create it first.")
             return None
        _pinecone_index = pc.Index(index_name)
        logger.info(f"Successfully connected to Pinecone index '{index_name}'.")
        return _pinecone_index
    except PineconeException as e:
        logger.error(f"Pinecone API Error during initialization: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"Unexpected error during Pinecone initialization: {e}", exc_info=True)
    _pinecone_index = None # Ensure it's None on failure
    return None

async def get_embeddings(texts: List[str], model="text-embedding-3-small") -> Optional[List[List[float]]]:
    """Generates embeddings for a list of texts using OpenAI API."""
    global _embedding_client
    if _embedding_client is None:
        _embedding_client = get_openai_client()
        if _embedding_client is None:
             logger.error("Failed to get OpenAI client for embeddings.")
             return None
    if not texts: return []
    try:
        response = await _embedding_client.embeddings.create(input=texts, model=model)
        return [item.embedding for item in response.data]
    except OpenAIError as e:
        logger.error(f"OpenAI API Error during embedding generation: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"Unexpected error during embedding generation: {e}", exc_info=True)
    return None

def generate_pinecone_id(url: str) -> str:
    """Creates a stable, unique ID for Pinecone based on the URL."""
    return hashlib.sha256(url.encode('utf-8')).hexdigest()

async def prepare_and_upsert_batch(
    product_data_list: List[ProductData],
    batch_size: int = 100
) -> bool:
    """Prepares and upserts product data to Pinecone in batches."""
    pinecone_index = initialize_pinecone()
    if not pinecone_index: return False
    if not product_data_list: return True

    total_upserted = 0
    overall_success = True
    for i in range(0, len(product_data_list), batch_size):
        batch_data = product_data_list[i:i + batch_size]
        logger.info(f"Processing Pinecone batch {i // batch_size + 1} ({len(batch_data)} items)...")
        texts_to_embed = [f"Product: {item.product_name or ''}\nDescription: {item.description or ''}\nIngredients: {item.ingredients_list or ''}\nHow to use: {item.how_to_use or ''}".strip() for item in batch_data]
        embeddings = await get_embeddings(texts_to_embed)
        if embeddings is None or len(embeddings) != len(batch_data):
            logger.error(f"Failed to generate embeddings for batch {i // batch_size + 1}. Skipping.")
            overall_success = False; continue
        vectors_to_upsert = []
        for item, vector in zip(batch_data, embeddings):
            pinecone_id = generate_pinecone_id(str(item.url))
            # Exclude the embedding source text from metadata if desired
            # Ensure all metadata values are Pinecone-compatible types (str, float, int, bool, list[str])
            # Create metadata dict, excluding None values
            metadata_full = item.model_dump(mode='json', exclude={'url'})
            metadata = {k: v for k, v in metadata_full.items() if v is not None} # Filter out None values
            metadata['original_url'] = str(item.url) # Add url back as string
            # Convert any non-string lists or complex types if necessary, though model_dump(mode='json') should handle most
            vectors_to_upsert.append((pinecone_id, vector, metadata))
        try:
            upsert_response = pinecone_index.upsert(vectors=vectors_to_upsert)
            logger.info(f"Upserted batch {i // batch_size + 1}. Response: {upsert_response}")
            total_upserted += upsert_response.upserted_count or 0
        except PineconeException as e:
            logger.error(f"Pinecone API Error upserting batch {i // batch_size + 1}: {e}", exc_info=True)
            overall_success = False
        except Exception as e:
            logger.error(f"Unexpected error upserting batch {i // batch_size + 1}: {e}", exc_info=True)
            overall_success = False
    logger.info(f"Pinecone upsert finished. Total upserted (estimated): {total_upserted}")
    return overall_success

async def crawl_parallel(urls: List[str], resume_file: str, max_concurrent: int = 10) -> List[ProductData]:
    logger.info("\n=== Parallel Crawling with arun_many + Dispatcher ===")

    # Track the peak memory usage for observability
    peak_memory = 0
    process = psutil.Process(os.getpid())
    def log_memory(prefix: str = ""):
        nonlocal peak_memory
        current_mem = process.memory_info().rss  # in bytes
        if current_mem > peak_memory:
            peak_memory = current_mem
        logger.info(f"{prefix} Current Memory: {current_mem // (1024 * 1024)} MB, Peak: {peak_memory // (1024 * 1024)} MB")

    # Configure the crawler
    crawler = AsyncWebCrawler(
        verbose=False, # Added speculative fix for config attribute error
        config=BrowserConfig(
            headless=True,
            java_script_enabled=True # Explicitly enable JavaScript
        ),
        run_config=CrawlerRunConfig(
            cache_mode=CacheMode.BYPASS,
            page_timeout=90000, # Increased page timeout to 90 seconds
            # Add JS execution and wait time
            js_code="window.scrollTo(0, document.body.scrollHeight);",
            wait_for=3, # Wait 3 seconds after JS execution/page load
            markdown_generator=DefaultMarkdownGenerator(), # Add markdown generation
            # process_page_func=None # We will handle results after arun_many
        ),
        dispatcher=MemoryAdaptiveDispatcher(
            memory_threshold_percent=80.0, # Use this instead of high/low water mark
            check_interval=1.0,            # Check memory every second
            max_session_permit=max_concurrent # Control concurrency here
        )
    )

    results = []
    failed_urls = []

    log_memory("Before arun_many:")
    try:
        logger.info(f"Starting parallel crawl for {len(urls)} URLs with max_concurrent={max_concurrent}...")
        crawl_results = await crawler.arun_many(urls)
        log_memory("After arun_many:")

        # Process results *after* crawling is complete
        logger.info("Processing crawled results...")
        successful_extractions = []
        for result in crawl_results:
            log_memory(f"Processing {result.url[:50]}...:")
            product_data = await process_page(result)
            if product_data:
                # Convert Pydantic model to dict for serialization
                successful_extractions.append(product_data)
            else:
                failed_urls.append(result.url)

        logger.info("--- Crawl & Extraction Summary ---")
        logger.info(f"Total URLs attempted: {len(urls)}")
        logger.info(f"Successfully crawled & extracted: {len(successful_extractions)}")
        logger.info(f"Failed crawls/extractions: {len(failed_urls)}")
        logger.info(f"Peak Memory Usage: {peak_memory // (1024 * 1024)} MB")

        # Return extracted data
        return successful_extractions # Ensure we return the list

    except Exception as e:
        logger.error(f"An error occurred during parallel crawling: {e}", exc_info=True)
        return [] # Return empty list on major error
    finally:
        # Ensure browser is closed even if errors occur
        await crawler.close()
        logger.info("Crawler closed.")

    # --- Formatting for Pinecone --- #
    if not _pinecone_index:
        logger.error("Pinecone index not available. Skipping upsert.")
        return # Or raise an error, depending on desired flow

    logger.info(f"Formatting {len(successful_extractions)} products for Pinecone...")
    pinecone_formatted_data = []
    formatting_failures = 0
    for product in successful_extractions:
        formatted = format_data_for_pinecone(product)
        if formatted:
            pinecone_formatted_data.append(formatted)
        else:
            formatting_failures += 1
    logger.info(f"Successfully formatted {len(pinecone_formatted_data)} products.")
    if formatting_failures > 0:
        logger.warning(f"Failed to format {formatting_failures} products for Pinecone.")

    # --- Upserting to Pinecone (with resume logging) --- #
    if not pinecone_formatted_data:
        logger.info("No data to upsert to Pinecone.")
        return # Nothing more to do in this function

    logger.info(f"Starting Pinecone upsert for {len(pinecone_formatted_data)} vectors...")
    batch_size = 100
    total_upserted_count = 0
    failed_batches = 0

    for i, batch in enumerate(create_batches(pinecone_formatted_data, batch_size)):
        logger.info(f"Upserting batch {i + 1}... ({len(batch)} vectors)")
        success = await upsert_batch_pinecone(_pinecone_index, batch)
        if success:
            total_upserted_count += len(batch)
            # Log successfully processed batch to resume file
            append_processed_batch(resume_file, batch)
        else:
            failed_batches += 1
            logger.error(f"Failed to upsert batch {i + 1}.")

    logger.info("--- Pinecone Upsert Summary ---")
    logger.info(f"Attempted to upsert {len(pinecone_formatted_data)} vectors.")
    logger.info(f"Successfully upserted: {total_upserted_count} vectors.")
    if failed_batches > 0:
        logger.error(f"Failed to upsert {failed_batches} batches.")
    # Note: Does not return anything specific now, main handles overall finish

# Updated function to handle recursion for sitemap index files
def fetch_sitemap_urls(initial_sitemap_url: str, max_depth: int = 5) -> List[str]:
    """
    Fetches URLs from a sitemap, handling both single sitemaps and sitemap index files recursively.

    Args:
        initial_sitemap_url: The URL of the starting sitemap or sitemap index file.
        max_depth: Maximum recursion depth for sitemap indexes to prevent infinite loops.

    Returns:
        List[str]: A list of unique URLs extracted.
    """
    all_urls = set() # Use a set to automatically handle duplicates
    urls_to_process = [(initial_sitemap_url, 0)] # Store (url, depth)
    processed_sitemaps = set()

    while urls_to_process:
        current_url, current_depth = urls_to_process.pop(0)

        if current_url in processed_sitemaps:
            logger.debug(f"Skipping already processed sitemap: {current_url}")
            continue

        if current_depth > max_depth:
            logger.warning(f"Reached max depth ({max_depth}), skipping sitemap index: {current_url}")
            continue

        processed_sitemaps.add(current_url)
        logger.info(f"Processing sitemap/index (Depth {current_depth}): {current_url}")
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

        try:
            response = requests.get(current_url, headers=headers, timeout=30)
            response.raise_for_status()
            content = response.content

            # Use lxml to parse and check the root tag
            try:
                # Remove potential XML declaration before parsing if causing issues
                # content = re.sub(b'^<\?xml.*?\?>', b'', content, count=1)
                root = lxml.etree.fromstring(content)
                # Detect namespace automatically (more robust)
                detected_namespaces = root.nsmap
                ns = detected_namespaces.get(None, 'http://www.sitemaps.org/schemas/sitemap/0.9') # Default if no specific ns
                namespaces = {'ns': ns}

                # Check the root tag's local name (ignoring namespace)
                tag_name = lxml.etree.QName(root.tag).localname

                if tag_name == 'sitemapindex':
                    logger.info(f"Detected sitemap index: {current_url}")
                    # Extract child sitemap URLs
                    sitemap_elements = root.xpath('//ns:sitemap/ns:loc', namespaces=namespaces)
                    child_sitemap_urls = [elem.text for elem in sitemap_elements if elem.text]
                    logger.info(f"Found {len(child_sitemap_urls)} child sitemaps in index.")

                    # Add new sitemaps to the list to be processed (recursive step)
                    for child_url in child_sitemap_urls:
                         if child_url not in processed_sitemaps:
                             urls_to_process.append((child_url, current_depth + 1))
                         else:
                             logger.debug(f"Skipping already processed child sitemap: {child_url}")

                elif tag_name == 'urlset':
                    logger.info(f"Detected regular sitemap: {current_url}")
                    # Extract URLs directly from this sitemap
                    url_elements = root.xpath('//ns:url/ns:loc', namespaces=namespaces)
                    extracted_urls = {elem.text for elem in url_elements if elem.text}
                    new_urls_count = len(extracted_urls - all_urls)
                    logger.info(f"Extracted {len(extracted_urls)} URLs ({new_urls_count} new) from {current_url}")
                    all_urls.update(extracted_urls)

                else:
                    logger.warning(f"Unknown root tag '{tag_name}' in {current_url}. Skipping.")

            except lxml.etree.XMLSyntaxError as e:
                logger.error(f"XML Parsing Error for {current_url}: {e}")
            except Exception as e: # Catch other potential parsing/xpath errors
                 logger.error(f"Error processing XML content for {current_url}: {e}")


        except requests.exceptions.RequestException as e:
            logger.error(f"HTTP Error fetching {current_url}: {e}")
        except Exception as e:
            logger.error(f"Unexpected error processing {current_url}: {e}")

    logger.info(f"Finished processing sitemaps. Found {len(all_urls)} unique URLs in total.")
    return list(all_urls)

def filter_product_urls(urls: List[str]) -> List[str]:
    """
    Filters a list of URLs to include only potential Sephora product pages.
    Looks for patterns like '/p/' in the URL path.

    Args:
        urls: A list of URLs to filter.

    Returns:
        List[str]: A list containing only the filtered product URLs.
    """
    product_urls = []
    rejected_urls_sample = [] # Store sample of rejected URLs
    original_count = len(urls)
    # Corrected product pattern
    product_pattern = '/product/' # Sephora product pages use /product/
    # Example exclusion patterns (can be expanded)
    excluded_patterns = ['/beauty/landing-pages', '/sale', '/content/', '/happening/']
    sample_size = 3 # How many samples to log

    logger.info(f"Filtering {original_count} URLs for product pages...")
    filtered_out_count = 0
    for url in urls:
        # Basic check for product pattern
        is_product = product_pattern in url
        # Check for exclusion patterns
        is_excluded = any(pattern in url for pattern in excluded_patterns)

        if is_product and not is_excluded:
            product_urls.append(url)
        else:
            filtered_out_count += 1
            if len(rejected_urls_sample) < sample_size:
                 rejected_urls_sample.append(url)
            # logger.debug(f"Filtered out URL: {url}") # Keep this if detailed per-URL logging is needed at DEBUG

    logger.info(f"Filtering complete: Kept {len(product_urls)} URLs, Filtered out {filtered_out_count} URLs from original {original_count}.")

    # Log samples at DEBUG level
    if product_urls:
        logger.debug(f"Sample Accepted URLs (up to {sample_size}): {product_urls[:sample_size]}")
    if rejected_urls_sample:
        logger.debug(f"Sample Rejected URLs (up to {sample_size}): {rejected_urls_sample}")

    return product_urls

# --- Main Execution Logic (Modified) ---

async def main(): # Changed to async def
    parser = argparse.ArgumentParser(description="Sephora Product Scraper")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of product URLs to process.")
    parser.add_argument("--resume-file", type=str, default="processed_urls.log", help="File to store/load processed URLs.")
    parser.add_argument("--batch-size", type=int, default=100, help="Batch size for writing processed URLs.")
    parser.add_argument("--max-concurrent", type=int, default=10, help="Maximum concurrent crawl requests.")
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Set the logging level.")
    parser.add_argument("--output-file", type=str, default=None, help="Optional JSON file to save extracted product data.")
    # Added Pinecone flag
    parser.add_argument("--send-to-pinecone", action="store_true", help="Send extracted data to Pinecone index specified in config.")

    args = parser.parse_args()

    # --- Logging Setup ---
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s', force=True)

    # --- Sitemap Fetching ---
    initial_sitemap = "https://www.sephora.com/sitemaps/products-sitemap.xml"
    logger.info(f"Fetching URLs from sitemap: {initial_sitemap}")
    all_sitemap_urls = fetch_sitemap_urls(initial_sitemap)
    logger.info(f"Found {len(all_sitemap_urls)} total URLs in sitemaps.")

    # --- Filtering ---
    product_urls = filter_product_urls(all_sitemap_urls)
    logger.info(f"Filtered down to {len(product_urls)} product URLs.")

    # --- Resume Logic ---
    processed_urls = load_processed_urls(args.resume_file)
    urls_to_crawl = [url for url in product_urls if url not in processed_urls]
    logger.info(f"{len(processed_urls)} URLs already processed. Need to crawl {len(urls_to_crawl)} URLs.")

    # --- Apply Limit ---
    if args.limit is not None:
        urls_to_crawl = urls_to_crawl[:args.limit]
        logger.info(f"Applying limit: processing {len(urls_to_crawl)} URLs.")

    if not urls_to_crawl:
        logger.info("No new URLs to crawl.")
        return

    # --- Crawling & Extraction ---
    # Ensure crawl_parallel is called correctly and its result is captured
    # The actual crawl_parallel function needs to be defined or imported correctly
    # Assuming it's defined elsewhere and returns List[ProductData]
    start_crawl_time = time.time()
    # Use await instead of asyncio.run
    extracted_data = await crawl_parallel(urls_to_crawl, args.resume_file, args.max_concurrent)
    crawl_duration = time.time() - start_crawl_time
    logger.info(f"Crawling and extraction finished in {crawl_duration:.2f} seconds.")

    # --- Save to JSON File (Existing Logic) ---
    if args.output_file and extracted_data:
        logger.info(f"Saving {len(extracted_data)} extracted products to {args.output_file}...")
        data_to_save = [item.model_dump(mode='json') for item in extracted_data]
        try:
            with open(args.output_file, 'w', encoding='utf-8') as f:
                json.dump(data_to_save, f, indent=4, ensure_ascii=False)
            logger.info(f"Successfully saved data to {args.output_file}")
        except IOError as e:
            logger.error(f"Failed to save data to {args.output_file}: {e}")
        except Exception as e:
             logger.error(f"Unexpected error saving to JSON: {e}", exc_info=True)

    # --- Send to Pinecone (New Logic) ---
    if args.send_to_pinecone:
        if extracted_data:
            logger.info("Preparing to send data to Pinecone...")
            # Use await instead of asyncio.run
            upsert_success = await prepare_and_upsert_batch(extracted_data)
            if upsert_success:
                logger.info("Pinecone upsert process completed successfully.")
            else:
                logger.error("Pinecone upsert process encountered errors.")
        else:
            logger.warning("No extracted data available to send to Pinecone.")
    else:
        logger.info("Skipping Pinecone upsert (flag not set).")

    logger.info("Sephora scraper finished.")

if __name__ == "__main__":
    # Load environment variables (needed for API keys used in sub-modules)
    from dotenv import load_dotenv
    # Assumes .env file is in the parent directory relative to src/
    dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env')
    if os.path.exists(dotenv_path):
        load_dotenv(dotenv_path=dotenv_path)
        logger.info("Loaded environment variables from .env file.")
    else:
        logger.warning(".env file not found, relying on system environment variables.")

    # Initialize settings (important for OpenAI/Pinecone keys)
    try:
        from .config import get_settings
        get_settings() # Ensures settings are loaded and validated early
    except ImportError:
        from config import get_settings
        get_settings()
    except SystemExit:
        logger.critical("Failed to load required settings. Exiting.")
        sys.exit(1)

    asyncio.run(main()) 