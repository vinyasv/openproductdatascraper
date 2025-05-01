import logging
import os
import asyncio # Added for retry delay
from typing import Optional
from openai import OpenAI, AsyncOpenAI, AuthenticationError, RateLimitError, APITimeoutError, APIConnectionError

# Import settings from the config module
# Assuming config.py is in the same directory or accessible via python path
try:
    from .config import current_settings
except ImportError:
    # Fallback for running llm.py directly or if structure differs
    from config import current_settings

logger = logging.getLogger(__name__)

# Global variable to hold the client instance (optional, can be instantiated per request)
_openai_client: AsyncOpenAI = None

def get_openai_client() -> Optional[AsyncOpenAI]:
    """Initializes and returns the AsyncOpenAI client.

    Uses the API key from the loaded settings.
    Handles AuthenticationError if the key is invalid or missing.

    Returns:
        An initialized AsyncOpenAI client instance, or None if initialization fails.
    """
    global _openai_client
    # Initialize only if not already done (simple singleton pattern)
    if _openai_client is None:
        if not current_settings or not current_settings.openai_api_key:
            logger.error("OpenAI API key not found in settings. Cannot initialize client.")
            return None
        try:
            logger.debug("Initializing AsyncOpenAI client...")
            _openai_client = AsyncOpenAI(
                api_key=current_settings.openai_api_key,
                # Add other configurations like timeout if needed
                # timeout=30.0,
            )
            logger.info("AsyncOpenAI client initialized successfully.")
            # You could add a simple test call here if desired, e.g., listing models
            # await _openai_client.models.list()
        except AuthenticationError as e:
            logger.error(f"OpenAI Authentication Error: {e}. Please check your API key.")
            _openai_client = None # Ensure client is None if auth fails
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}", exc_info=True)
            _openai_client = None

    return _openai_client

def create_extraction_prompt(html_content: str, url: str) -> str:
    """Creates a detailed prompt for the LLM to extract product data.

    Args:
        html_content: The cleaned HTML content of the product page.
        url: The URL of the product page (for context and the 'url' field).

    Returns:
        A formatted prompt string.
    """

    # Define the desired JSON structure based on models.ProductData
    json_schema = """
{
  "product_name": "string (Product's full name)",
  "url": "string (The exact URL provided: """ + url + """)",
  "price": "float | null (Numerical price, null if not found)",
  "currency": "string | null (Currency code like USD, EUR, null if not found or price is null)",
  "stock_info": "string | null (Availability status like 'In Stock', 'Out of Stock', 'Coming Soon', null if not found)",
  "image_url": "string | null (Full URL of the main product image, null if not found)",
  "description": "string | null (Product description text, null if not found)",
  "ingredients_list": "string | null (The COMPLETE list of ingredients, exactly as written. If not found, use null. Do NOT summarize or omit ingredients.)",
  "how_to_use": "string | null (The FULL instructions on how to use the product. If not found, use null.)"
}
"""

    prompt = f"""
You are an expert data extraction AI. Your task is to extract information about a Sephora product from the provided HTML content.

The URL of the page is: {url}

Please analyze the following cleaned HTML content carefully:
```html
{html_content}
```

Extract the required information and format it STRICTLY as a JSON object matching the schema below.
- Provide the exact URL given above in the "url" field.
- For optional fields (price, currency, stock_info, image_url, description, ingredients_list, how_to_use), use `null` if the information cannot be found in the HTML content. Do NOT guess or make up information.
- It is CRUCIAL to extract the *entire* ingredient list if present under "ingredients_list". Do not truncate or summarize.
- Similarly, extract the *complete* "how_to_use" instructions if available.
- Ensure the output is ONLY the JSON object, with no introductory text, explanations, or markdown formatting around it.

JSON Schema:
{json_schema}

Output ONLY the JSON object:
"""
    # logger.debug(f"Generated extraction prompt for {url}") # Optional: Log the full prompt at DEBUG level if needed
    return prompt.strip()

# Updated function with retry logic
# Changed to accept a messages list instead of a single prompt string
async def call_openai_extract(
    messages: list, # Changed from prompt: str
    model: str = "gpt-4o",
    max_retries: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 16.0
) -> Optional[str]:
    """Makes an extraction call to OpenAI with retry logic for transient errors."""
    client = get_openai_client()
    if not client:
        return None

    current_delay = initial_delay
    for attempt in range(max_retries + 1):
        try:
            logger.debug(f"Making OpenAI call (Attempt {attempt + 1}/{max_retries + 1}) with model: {model}")
            response = await client.chat.completions.create(
                model=model,
                messages=messages, # Use the provided messages list directly
                temperature=0.2,
                # Potentially add response_format={'type': 'json_object'} if using compatible models
                # response_format={"type": "json_object"}, # Ensure model supports this!
            )
            content = response.choices[0].message.content
            logger.debug("Received response from OpenAI successfully.")
            return content # Success, return content

        except (RateLimitError, APITimeoutError, APIConnectionError) as e:
            if attempt == max_retries:
                logger.error(f"OpenAI API error after {max_retries + 1} attempts: {e}")
                return None # Max retries reached
            else:
                logger.warning(f"OpenAI API error (Attempt {attempt + 1}): {e}. Retrying in {current_delay:.2f}s...")
                await asyncio.sleep(current_delay)
                # Exponential backoff with cap
                current_delay = min(current_delay * 2, max_delay)

        except AuthenticationError as e:
             logger.error(f"OpenAI Authentication Error: {e}. Aborting retries.")
             return None # No point retrying auth errors

        except Exception as e:
            # Catch unexpected errors
            if attempt == max_retries:
                logger.error(f"Unexpected error during OpenAI API call after {max_retries + 1} attempts: {e}", exc_info=True)
                return None
            else:
                 logger.warning(f"Unexpected error (Attempt {attempt+1}): {e}. Retrying in {current_delay:.2f}s...")
                 await asyncio.sleep(current_delay)
                 current_delay = min(current_delay * 2, max_delay)

    # Should not be reached if loop logic is correct, but as a safeguard
    logger.error("Exited retry loop unexpectedly.")
    return None 