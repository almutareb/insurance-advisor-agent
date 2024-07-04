# documents loader function
from langchain_community.document_loaders import RecursiveUrlLoader
from bs4 import BeautifulSoup as Soup
from validators import url as url_validator
from langchain_core.documents import Document
import time
import logging
import sys

logger = logging.getLogger(__name__)

def load_docs_from_urls(
        urls: list = ["https://docs.python.org/3/"], 
        max_depth: int = 5,
        ) -> list[Document]:
    """
    Load documents from a list of URLs.

    ## Args:
        urls (list, optional): A list of URLs to load documents from. Defaults to ["https://docs.python.org/3/"].
        max_depth (int, optional): Maximum depth to recursively load documents from each URL. Defaults to 5.

    ## Returns:
        list: A list of documents loaded from the given URLs.
        
    ## Raises:
        ValueError: If any URL in the provided list is invalid.
    """
    stf = time.time()  # Start time for performance measurement
    docs = []
    for url in urls:
        st = time.time()  # Start time for outer performance measurement
        if not url_validator(url):
            raise ValueError(f"Invalid URL: {url}")
        try:
            st = time.time()  # Start time for inner performance measurement
            loader = RecursiveUrlLoader(url=url, max_depth=max_depth, extractor=lambda x: Soup(x, "html.parser").text)
            docs.extend(loader.load())

            et = time.time() - st  # Calculate time taken for splitting
            logMessage=f'Time taken for downloading documents from {url}: {et} seconds.'
            logger.info(logMessage)
            print(logMessage)
        except Exception as e:
            logMessage=f"Failed to load or parse the URL {url}. Error: {e}"
            logger.error(logMessage)
            print(logMessage, file=sys.stderr)
    etf = time.time() - stf  # Calculate time taken for scrapping all URLs
    print(f'Total time taken for downloading {len(docs)} documents: {etf} seconds.')
    return docs