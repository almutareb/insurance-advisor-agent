import hashlib
import datetime
import os
import uuid
from typing import Dict
import re
# from rag_app.utils import logger

# logger = logger.get_console_logger("utils")



def extract_urls(data_list):
    """
    Extracts URLs from a list of of dictionaries.

    Parameters:
    - formatted_list (list): A list of dictionaries, each containing 'Title:', 'link:', and 'summary:'.

    Returns:
    - list: A list of URLs extracted from the dictionaries.
    """
    urls = []
    print(data_list)
    for item in data_list:
        try:
            # Find the start and end indices of the URL
            lower_case = item.lower()
            link_prefix = 'link: '
            summary_prefix = ', summary:'
            start_idx = lower_case.index(link_prefix) + len(link_prefix)
            end_idx = lower_case.index(summary_prefix, start_idx)
            # Extract the URL using the indices found
            url = item[start_idx:end_idx]
            urls.append(url)
        except ValueError:
            # Handles the case where 'link: ' or ', summary:' is not found in the string
            print("Could not find a URL in the item:", item)
    last_sources = urls[-3:]
    return last_sources

def format_search_results(search_results):
    """
    Formats a list of dictionaries containing search results into a list of strings.
    Each dictionary is expected to have the keys 'title', 'link', and 'snippet'.

    Parameters:
    - search_results (list): A list of dictionaries, each containing 'title', 'link', and 'snippet'.

    Returns:
    - list: A list of formatted strings based on the search results.
    """
    if len(search_results)>1:
        formatted_results = [
            "Title: {title}, Link: {link}, Summary: {snippet}".format(**i)
            for i in search_results
        ]
    return formatted_results

def parse_list_to_dicts(items: list) -> list:
    parsed_items = []
    for item in items:
        # Extract title, link, and summary from each string
        title_start = item.find('Title: ') + len('Title: ')
        link_start = item.find('Link: ') + len('Link: ')
        summary_start = item.find('Summary: ') + len('Summary: ')

        title_end = item.find(', Link: ')
        link_end = item.find(', Summary: ')
        summary_end = len(item)

        title = item[title_start:title_end]
        link = item[link_start:link_end]
        summary = item[summary_start:summary_end]

        # Use the hash_text function for the hash_id
        hash_id = hash_text(link)

        # Construct the dictionary for each item
        parsed_item = {
            "url": link,
            "title": title,
            "hash_id": hash_id,
            "summary": summary
        }
        parsed_items.append(parsed_item)
    return parsed_items

def hash_text(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()


def convert_timestamp_to_datetime(timestamp: str) -> str:
    return datetime.datetime.fromtimestamp(int(timestamp)).strftime("%Y-%m-%d %H:%M:%S")

def create_folder_if_not_exists(folder_path: str) -> None:
    """
    Create a folder if it doesn't already exist.

    Args:
    - folder_path (str): The path of the folder to create.
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder '{folder_path}' created.")
    else:
        print(f"Folder '{folder_path}' already exists.")
        
def generate_uuid() -> str:
    """
    Generate a UUID (Universally Unique Identifier) and return it as a string.

    Returns:
        str: A UUID string.
    """
    return str(uuid.uuid4())

def extract_responses(text: str) -> Dict[str, str]:
    """
    Extracts the user response and AI response from the provided text.

    Args:
        text (str): The input text containing user and AI responses.

    Returns:
        Dict[str, str]: A dictionary with keys 'USER' and 'AI' containing the respective responses.
    """
    user_pattern = re.compile(r'USER: (.*?) \n', re.DOTALL)
    ai_pattern = re.compile(r'AI: (.*?)$', re.DOTALL)
    
    user_match = user_pattern.search(text)
    ai_match = ai_pattern.search(text)
    
    responses = {
        "USER": user_match.group(1) if user_match else "",
        "AI": ai_match.group(1) if ai_match else ""
    }
    
    return responses