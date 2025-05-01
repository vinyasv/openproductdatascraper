import logging
import time
import asyncio # Added for retry logic
from typing import Optional, List, Tuple, Generator, Dict
import hashlib

# Revert main import but add specific import for exception handling
from pinecone import Pinecone, Index, ApiException
from pinecone.exceptions import PineconeException
from openai import AsyncOpenAI, OpenAIError # Import OpenAIError for broader catching

# Import settings and models
try:
    from .config import current_settings
    from .models import ProductData
except ImportError:
    # Fallback for running directly
    from config import current_settings
    from models import ProductData

logger = logging.getLogger(__name__)

# Global variable to hold the index instance (optional)
_pinecone_index: Optional[Index] = None
# Global variable for the embedding client (sync version for simplicity here, or use async)
# Using the existing async client from llm might be better if we make this async
_embedding_client: AsyncOpenAI = None

def setup_pinecone(index_name: str = None, dimension: int = None) -> Optional[Index]:
    """Initializes Pinecone client, checks/creates the index, and returns the index object.

    Args:
        index_name (str, optional): Pinecone index name. Defaults to settings.
        dimension (int, optional): Embedding dimension. Defaults to settings.

    Returns:
        An initialized Pinecone Index object, or None if setup fails.
    """
    global _pinecone_index

    # Use settings if arguments are not provided
    if index_name is None:
        index_name = current_settings.pinecone_index_name
    if dimension is None:
        dimension = current_settings.pinecone_embedding_dim

    logger.info(f"Setting up Pinecone index '{index_name}' (dimension: {dimension})...")
    start_time = time.time()

    try:
        # 1. Initialize Pinecone Client
        if not current_settings.pinecone_api_key:
            logger.error("Pinecone API key not found in settings.")
            return None

        pc = Pinecone(
            api_key=current_settings.pinecone_api_key
        )
        logger.info("Pinecone client initialized.")

        # 2. Check if index exists
        existing_index_names = [idx.name for idx in pc.list_indexes()]
        if index_name not in existing_index_names:
            logger.info(f"Index '{index_name}' does not exist. Creating...")
            # 3. Create index if it doesn't exist
            # Specify metric (e.g., cosine, dotproduct, euclidean)
            # Specify pod type (e.g., 'p1.x1', 's1.x1') - check Pinecone docs for current options
            pc.create_index(
                name=index_name,
                dimension=dimension,
                metric='cosine', # Common choice for text embeddings
                spec={
                    "serverless": {
                        "cloud": "aws", # Specify your cloud provider
                        "region": "us-east-1" # Specify your region
                    }
                    # OR use pod spec:
                    # "pod": {
                    #     "environment": current_settings.pinecone_environment,
                    #     "pod_type": "p1.x1" # Example pod type
                    # }
                }
            )
            # Wait for index to be ready (optional but recommended)
            while not pc.describe_index(index_name).status['ready']:
                logger.info("Waiting for index to become ready...")
                time.sleep(5)
            logger.info(f"Index '{index_name}' created successfully.")
        else:
            logger.info(f"Index '{index_name}' already exists.")

        # 4. Get the index object
        _pinecone_index = pc.Index(index_name)
        logger.info(f"Successfully connected to Pinecone index '{index_name}'.")

        # Optional: Describe index to confirm details
        # index_description = pc.describe_index(index_name)
        # logger.debug(f"Index description: {index_description}")

        end_time = time.time()
        logger.info(f"Pinecone setup complete. Duration: {end_time - start_time:.3f}s")
        return _pinecone_index

    except PineconeException as e:
        logger.error(f"Pinecone Error during setup: {e}", exc_info=True)
        _pinecone_index = None
        return None
    except Exception as e:
        logger.error(f"Unexpected error during Pinecone setup: {e}", exc_info=True)
        _pinecone_index = None
        return None

def format_data_for_pinecone(product: ProductData) -> Optional[Tuple[str, List[float], dict]]:
    """Formats a ProductData object into the tuple format required for Pinecone upsert.

    Uses the product URL as the ID, generates placeholder embeddings, and uses
    the ProductData.to_pinecone_dict() method for metadata.

    Args:
        product: A validated ProductData object.

    Returns:
        A tuple (id, values, metadata) suitable for pinecone.upsert, or None if formatting fails.
    """
    if not product or not product.url:
        logger.warning("Cannot format data for Pinecone: Invalid ProductData object or missing URL.")
        return None

    try:
        # 1. Use URL as the unique ID
        vector_id = str(product.url)

        # 2. Generate placeholder embedding values (replace with actual embeddings later)
        # TODO: Integrate actual embedding generation (e.g., using OpenAI)
        dimension = current_settings.pinecone_embedding_dim
        placeholder_values = [0.0] * dimension

        # 3. Get flattened metadata from the Pydantic model
        metadata = product.to_pinecone_dict()
        # Remove URL from metadata if it's only used as ID (optional, depends on use case)
        # metadata.pop('url', None)

        # Pinecone metadata values must be str, float, int, bool, or list of strings.
        # The to_pinecone_dict method should handle this, but an extra check can be added if needed.
        # for k, v in metadata.items():
        #     if not isinstance(v, (str, float, int, bool, list)):
        #          if isinstance(v, list) and any(not isinstance(item, str) for item in v):
        #               logger.warning(f"Metadata field '{k}' has non-string list elements, may cause issues.")
        #     elif not isinstance(v, (str, float, int, bool)):
        #          logger.warning(f"Metadata field '{k}' has incompatible type {type(v)}, may cause issues.")

        return (vector_id, placeholder_values, metadata)

    except Exception as e:
        logger.error(f"Failed to format product data for Pinecone (URL: {product.url}): {e}", exc_info=True)
        return None

def create_batches(data: List[Tuple[str, List[float], dict]], batch_size: int = 100) -> Generator[List[Tuple[str, List[float], dict]], None, None]:
    """Yields successive n-sized chunks from lst."""
    if not data:
        return
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]

# Implemented batch upsert function with retries
async def upsert_batch_pinecone(
    index: Index,
    batch_data: List[Tuple[str, List[float], dict]],
    max_retries: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 16.0
) -> bool:
    """Upserts a single batch of data to the specified Pinecone index with retry logic.

    Args:
        index: The initialized Pinecone Index object.
        batch_data: A list of tuples, where each tuple is (id, values, metadata).
        max_retries: Maximum number of retry attempts for the batch.
        initial_delay: Initial delay in seconds for retries.
        max_delay: Maximum delay in seconds for retries.

    Returns:
        True if the batch was upserted successfully within retry limits, False otherwise.
    """
    if not index or not batch_data:
        logger.warning("Upsert attempt skipped: Invalid index or empty batch data.")
        return False

    batch_size = len(batch_data)
    current_delay = initial_delay

    for attempt in range(max_retries + 1):
        logger.debug(f"Attempting to upsert batch of {batch_size} vectors (Attempt {attempt + 1}/{max_retries + 1})...")
        try:
            upsert_response = index.upsert(vectors=batch_data)
            upserted_count = upsert_response.get('upserted_count', 0)

            if upserted_count == batch_size:
                logger.info(f"Successfully upserted batch of {upserted_count} vectors (Attempt {attempt + 1}).")
                return True # Success!
            else:
                # Partial success or complete failure reported by API without exception
                logger.warning(f"Pinecone upsert attempt {attempt + 1} reported {upserted_count} vectors upserted out of {batch_size}. Response: {upsert_response}")
                # Decide if retry is needed based on this - treating as failure for now
                if attempt == max_retries:
                    logger.error(f"Batch upsert failed after {max_retries + 1} attempts (partial success on last attempt?).")
                    return False
                # Fall through to retry logic

        except PineconeException as e:
            logger.warning(f"Pinecone Error during batch upsert (Attempt {attempt + 1}): {e}")
            if attempt == max_retries:
                logger.error(f"Batch upsert failed after {max_retries + 1} attempts due to API error.", exc_info=True)
                if batch_data:
                    logger.error(f"Failed batch started with ID: {batch_data[0][0]}")
                return False
            # Fall through to retry logic

        except Exception as e:
            logger.warning(f"Unexpected error during batch upsert (Attempt {attempt + 1}): {e}", exc_info=True)
            if attempt == max_retries:
                logger.error(f"Batch upsert failed after {max_retries + 1} attempts due to unexpected error.")
                return False
            # Fall through to retry logic

        # If we reach here, it means the attempt failed and it's not the last one
        logger.info(f"Retrying batch upsert in {current_delay:.2f}s...")
        await asyncio.sleep(current_delay)
        current_delay = min(current_delay * 2, max_delay) # Exponential backoff

    # Should technically not be reached if logic is correct
    logger.error("Exited upsert retry loop unexpectedly.")
    return False

# Example usage (optional - for testing)
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    # Assuming .env is present in the parent directory relative to src
    from dotenv import load_dotenv
    import os
    load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))
    
    # Load settings explicitly for the test run
    from config import load_config
    load_config()
    
    index = setup_pinecone()
    if index:
        print("Pinecone setup successful!")
        # Example: Get index stats
        try:
             stats = index.describe_index_stats()
             print(f"Index Stats: {stats}")
        except Exception as e:
             print(f"Error getting stats: {e}")
    else:
        print("Pinecone setup failed.")

def initialize_pinecone() -> Optional[Index]:
    """Initializes the Pinecone client and connects to the specified index.

    Uses API key and index name from loaded settings.
    Handles common Pinecone API exceptions.

    Returns:
        An initialized Pinecone Index object, or None if initialization fails.
    """
    global _pinecone_index
    if _pinecone_index is not None:
        logger.debug("Pinecone index already initialized.")
        return _pinecone_index

    if not current_settings or not current_settings.pinecone_api_key or not current_settings.pinecone_index_name:
        logger.error("Pinecone API key or index name not found in settings. Cannot initialize.")
        return None

    try:
        logger.info(f"Initializing Pinecone client for index '{current_settings.pinecone_index_name}'...")
        # Pinecone client v3+ initialization
        pc = Pinecone(api_key=current_settings.pinecone_api_key)

        index_name = current_settings.pinecone_index_name
        if index_name not in pc.list_indexes().names:
             logger.error(f"Pinecone index '{index_name}' does not exist. Please create it first.")
             # Optionally, add code to create the index if it doesn't exist
             # pc.create_index(index_name, dimension=current_settings.pinecone_embedding_dim, metric='cosine') # Example
             # logger.info(f"Index '{index_name}' created.")
             # time.sleep(5) # Give time for index to initialize
             return None # Return None if index needs manual creation

        _pinecone_index = pc.Index(index_name)
        logger.info(f"Successfully connected to Pinecone index '{index_name}'.")
        # Optionally log index stats
        # logger.debug(f"Index stats: {_pinecone_index.describe_index_stats()}")
        return _pinecone_index

    except ApiException as e:
        logger.error(f"Pinecone API Error during initialization: {e}", exc_info=True)
        _pinecone_index = None
    except Exception as e:
        logger.error(f"Unexpected error during Pinecone initialization: {e}", exc_info=True)
        _pinecone_index = None

    return None

async def get_embeddings(texts: List[str], model="text-embedding-3-small") -> Optional[List[List[float]]]:
    """Generates embeddings for a list of texts using OpenAI API."""
    global _embedding_client
    if _embedding_client is None:
        _embedding_client = get_openai_client() # Reuse the async client from llm utils
        if _embedding_client is None:
             logger.error("Failed to get OpenAI client for embeddings.")
             return None

    if not texts:
        return []

    try:
        logger.debug(f"Requesting embeddings for {len(texts)} texts using model '{model}'...")
        response = await _embedding_client.embeddings.create(input=texts, model=model)
        embeddings = [item.embedding for item in response.data]
        logger.debug(f"Successfully generated {len(embeddings)} embeddings.")
        return embeddings
    except OpenAIError as e:
        logger.error(f"OpenAI API Error during embedding generation: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"Unexpected error during embedding generation: {e}", exc_info=True)

    return None

def generate_pinecone_id(url: str) -> str:
    """Creates a stable, unique ID for Pinecone based on the URL."""
    # Using SHA256 hash of the URL for a consistent ID
    return hashlib.sha256(url.encode('utf-8')).hexdigest()

async def prepare_and_upsert_batch(
    product_data_list: List[ProductData],
    batch_size: int = 100 # Pinecone recommended batch size
) -> bool:
    """
    Prepares product data (generates embeddings, metadata) and upserts it
    to Pinecone in batches.

    Args:
        product_data_list: List of ProductData objects to upsert.
        batch_size: Number of vectors to upsert per batch.

    Returns:
        True if all batches were upserted successfully, False otherwise.
    """
    pinecone_index = initialize_pinecone()
    if not pinecone_index:
        logger.error("Cannot upsert data: Pinecone index not initialized.")
        return False

    if not product_data_list:
        logger.info("No product data provided for upserting.")
        return True # Nothing to do

    total_upserted = 0
    overall_success = True

    for i in range(0, len(product_data_list), batch_size):
        batch_data = product_data_list[i:i + batch_size]
        logger.info(f"Processing batch {i // batch_size + 1} for Pinecone upsert ({len(batch_data)} items)...")

        # 1. Prepare data for embedding (combine relevant fields)
        texts_to_embed = []
        for item in batch_data:
            # Combine key text fields for embedding
            text = f"Product: {item.product_name or ''}\nDescription: {item.description or ''}\nIngredients: {item.ingredients_list or ''}\nHow to use: {item.how_to_use or ''}"
            texts_to_embed.append(text.strip())

        # 2. Get embeddings
        embeddings = await get_embeddings(texts_to_embed)
        if embeddings is None or len(embeddings) != len(batch_data):
            logger.error(f"Failed to generate embeddings for batch {i // batch_size + 1}. Skipping batch.")
            overall_success = False
            continue # Skip this batch

        # 3. Prepare vectors for upsert (ID, vector, metadata)
        vectors_to_upsert: List[Tuple[str, List[float], Dict]] = []
        for item, vector in zip(batch_data, embeddings):
            pinecone_id = generate_pinecone_id(str(item.url)) # Use helper for ID
            # Exclude the embedding source text from metadata if desired
            # Ensure all metadata values are Pinecone-compatible types (str, float, int, bool, list[str])
            metadata = item.model_dump(mode='json', exclude={'url'}) # Already handles HttpUrl serialization
            metadata['original_url'] = str(item.url) # Add url back as string
            # Convert any non-string lists or complex types if necessary, though model_dump(mode='json') should handle most
            vectors_to_upsert.append((pinecone_id, vector, metadata))

        # 4. Upsert batch
        try:
            logger.debug(f"Upserting batch {i // batch_size + 1} with {len(vectors_to_upsert)} vectors...")
            upsert_response = pinecone_index.upsert(vectors=vectors_to_upsert)
            logger.info(f"Successfully upserted batch {i // batch_size + 1}. Response: {upsert_response}")
            total_upserted += upsert_response.upserted_count or 0
        except ApiException as e:
            logger.error(f"Pinecone API Error during upsert for batch {i // batch_size + 1}: {e}", exc_info=True)
            overall_success = False
        except Exception as e:
            logger.error(f"Unexpected error during upsert for batch {i // batch_size + 1}: {e}", exc_info=True)
            overall_success = False

    logger.info(f"Pinecone upsert process finished. Total items processed: {len(product_data_list)}, Total vectors successfully upserted (estimated): {total_upserted}")
    return overall_success 