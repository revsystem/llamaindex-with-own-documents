"""Set of constants."""
import os

import openai

FOLDERPATH_DOCUMENTS = os.path.join("data", "documents")
FOLDERPATH_INDEX = os.path.join("data", "indexes")
FILEPATH_CACHE_INDEX = os.path.join(FOLDERPATH_INDEX, "index.json")
FILEPATH_ARTICLE_URLS = os.path.join("data", "article_urls.json")
FILEPATH_RSS_URLS = os.path.join("data", "rss_urls.json")
FILEPATH_CACHE_STOREDDATA_URLS = os.path.join(FOLDERPATH_INDEX, "stored_urls.json")

openai.api_key = os.getenv("OPENAI_API_KEY")

VARIABLES_FILE = "variables.txt"
