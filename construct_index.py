import json
import logging
import os
import sys

import nest_asyncio
from llama_index import (
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    download_loader,
    get_response_synthesizer,
    set_global_service_context,
)
from llama_index.indices.document_summary import DocumentSummaryIndex
from llama_index.indices.loading import load_index_from_storage

# from llama_index.llms import OpenAI
from llama_index.query_engine.router_query_engine import RouterQueryEngine
from llama_index.selectors.pydantic_selectors import PydanticSingleSelector
from llama_index.tools.query_engine import QueryEngineTool

from constants import (
    FILEPATH_ARTICLE_URLS,
    FILEPATH_CACHE_INDEX,
    FILEPATH_CACHE_STOREDDATA_URLS,
    FILEPATH_RSS_URLS,
    FOLDERPATH_DOCUMENTS,
    VARIABLES_FILE,
)
from create_context import get_service_context
from prompt_tmpl import (
    CHAT_TEXT_QA_PROMPT,
    CHAT_TREE_SUMMARIZE_PROMPT,
    DEFAULT_CHOICE_SELECT_PROMPT,
    SINGLE_PYD_SELECT_PROMPT_TMPL,
    SUMMARY_QUERY,
)

# enable asynchronous processing
nest_asyncio.apply()

service_context = get_service_context()
set_global_service_context(service_context)


def load_variables():
    """
    Loads the global variables list_id and vector_id from a file.
    """

    global list_id, vector_id

    logging.info("check if the variable file exists.")
    if os.path.exists(VARIABLES_FILE):
        with open(VARIABLES_FILE, "r", encoding="utf-8") as file:
            logging.info("read the values from the file")
            values = file.read().split(",")
            list_id = values[0]
            vector_id = values[1]
    else:
        logging.info("There is no variables.txt. It will be created.")


def save_variables():
    """
    Saves the global variables list_id and vector_id to a file.
    """

    global list_id, vector_id

    # write the values to the file.
    with open(VARIABLES_FILE, "w", encoding="utf-8") as file:
        file.write(f"{list_id},{vector_id}")


def construct_index_with_file():
    """
    constructs and persists DocumentSummaryIndex and VectorStoreIndex from a directory of documents.
    It also updates the global variables list_id and vector_id with the respective index ids.

    Args:
        None

    Returns:
        None

    Part of this code is based on Apache 2.0 licensed code.
    Ref:
    - https://betterprogramming.pub/experimenting-llamaindex-routerqueryengine-with-document-management-19b17f2e3a32
    """

    directory_reader = SimpleDirectoryReader(
        FOLDERPATH_DOCUMENTS,
        filename_as_id=True,
    )

    os.makedirs(FILEPATH_CACHE_INDEX, exist_ok=True)
    documents = directory_reader.load_data()

    nodes = service_context.node_parser.get_nodes_from_documents(documents)
    storage_context = StorageContext.from_defaults()
    storage_context.docstore.add_documents(nodes)

    # construct list_index and vector_index from storage_context.
    summary_index = DocumentSummaryIndex(
        nodes,
        storage_context=storage_context,
        response_synthesizer=get_response_synthesizer(
            response_mode="tree_summarize",
            use_async=True,
            text_qa_template=CHAT_TEXT_QA_PROMPT,
            summary_template=CHAT_TREE_SUMMARIZE_PROMPT,
        ),
        summary_query=SUMMARY_QUERY,
        show_progress=True,
    )

    vector_index = VectorStoreIndex(
        nodes,
        storage_context=storage_context,
    )

    # persist both indexes to disk
    summary_index.storage_context.persist(persist_dir=FILEPATH_CACHE_INDEX)
    vector_index.storage_context.persist(persist_dir=FILEPATH_CACHE_INDEX)

    # update the global variables of list_id and vector_id
    global list_id, vector_id
    list_id = summary_index.index_id
    vector_id = vector_index.index_id

    save_variables()

    # run refresh_ref_docs function to check for document updates
    list_refreshed_docs = summary_index.refresh_ref_docs(
        documents, update_kwargs={"delete_kwargs": {"delete_from_docstore": True}}
    )
    print(list_refreshed_docs)
    print(
        f"Number of newly inserted/refreshed SummaryRefreshed docs: {sum(list_refreshed_docs)}"
    )

    summary_index.storage_context.persist(persist_dir=FILEPATH_CACHE_INDEX)
    logging.info("list_index refreshed and persisted to storage.")

    vector_refreshed_docs = vector_index.refresh_ref_docs(
        documents, update_kwargs={"delete_kwargs": {"delete_from_docstore": True}}
    )
    print(vector_refreshed_docs)
    print(
        f"Number of newly inserted/refreshed VectorRefreshed docs: {sum(vector_refreshed_docs)}"
    )

    vector_index.storage_context.persist(persist_dir=FILEPATH_CACHE_INDEX)
    logging.info("vector_index refreshed and persisted to storage.")


def query_with_index(user_input: str):
    """
    Executes a query based on user input and returns the result.

    Parameters:
    user_input (str): The input string from the user.

    Returns:
    response: The result of the query.

    Part of this code is based on Apache 2.0 licensed code.
    Ref:
    - https://betterprogramming.pub/experimenting-llamaindex-routerqueryengine-with-document-management-19b17f2e3a32
    - https://note.com/npaka/n/n0a068497ac96
    """

    storage_context = StorageContext.from_defaults(persist_dir=FILEPATH_CACHE_INDEX)

    print("Load DocumentSummaryIndex...")
    summary_index = load_index_from_storage(
        storage_context=storage_context,
        index_id=list_id,
    )
    print("Load VectorIndex...")
    vector_index = load_index_from_storage(
        storage_context=storage_context,
        index_id=vector_id,
    )

    # build list_tool and vector_tool
    # ref:https://gpt-index.readthedocs.io/en/latest/examples/query_engine/RouterQueryEngine.html
    print("Load ListQueryEngineTool...")
    list_tool = QueryEngineTool.from_defaults(
        query_engine=summary_index.as_query_engine(
            choice_select_prompt=DEFAULT_CHOICE_SELECT_PROMPT,
            response_synthesizer=get_response_synthesizer(
                response_mode="tree_summarize",
                use_async=True,
                summary_template=CHAT_TREE_SUMMARIZE_PROMPT,
            ),
        ),
        description="Useful for summarization questions related to the data source",
    )

    print("Load VectorQueryEngineTool...")
    vector_tool = QueryEngineTool.from_defaults(
        query_engine=vector_index.as_query_engine(
            similarity_top_k=3,
            response_synthesizer=get_response_synthesizer(
                response_mode="compact",
                text_qa_template=CHAT_TEXT_QA_PROMPT,
            ),
        ),
        description="Useful for retrieving specific context related to the data source",
    )

    # construct RouteQueryEngine
    print("Load RouterQueryEngine...")
    query_engine = RouterQueryEngine(
        selector=PydanticSingleSelector.from_defaults(
            prompt_template_str=SINGLE_PYD_SELECT_PROMPT_TMPL,
        ),
        query_engine_tools=[
            list_tool,
            vector_tool,
        ],
    )

    response = query_engine.query(user_input)

    return response


def update_index_with_rss() -> None:
    """
    Args:
        None

    Returns:
        None
    """
    if not os.path.exists(FILEPATH_CACHE_INDEX):
        print(
            f"Index dir {FILEPATH_CACHE_INDEX} does not exist. Create Indexs with files first."
        )
        sys.exit()

    create_urllist_with_rss()
    construct_index_with_urls()


def construct_index_with_urls() -> None:
    """
    Constructs an index with URLs from RSS feeds and saves it to a JSON file.

    This function reads URLs from an RSS feed, removes duplicates, and constructs an index with the unique URLs.
    The index is then saved to a JSON file. If the JSON file does not exist, it is created. If it does exist,
    the new URLs are added to it. The function uses the BeautifulSoupWebReader to read the RSS feed.

    Args:
        None

    Returns:
        None
    """

    beautiful_soup_web_reader = download_loader("BeautifulSoupWebReader")
    loader = beautiful_soup_web_reader()

    remove_duplicate_elements(FILEPATH_ARTICLE_URLS, "urls")
    new_urls = get_unique_elements(
        FILEPATH_CACHE_STOREDDATA_URLS, FILEPATH_ARTICLE_URLS
    )
    new_urls = set(new_urls)

    if new_urls:
        logging.info("[URL] Loading URL list...")

        documents = loader.load_data(urls=new_urls)
        nodes = service_context.node_parser.get_nodes_from_documents(documents)
        storage_context = StorageContext.from_defaults(persist_dir=FILEPATH_CACHE_INDEX)
        storage_context.docstore.add_documents(nodes)

        summary_index = load_index_from_storage(
            storage_context=storage_context,
            index_id=list_id,
        )
        vector_index = load_index_from_storage(
            storage_context=storage_context,
            index_id=vector_id,
        )

        summary_index.insert_nodes(nodes)
        vector_index.insert_nodes(nodes)
        summary_index.storage_context.persist(persist_dir=FILEPATH_CACHE_INDEX)
        vector_index.storage_context.persist(persist_dir=FILEPATH_CACHE_INDEX)

        if not os.path.exists(FILEPATH_CACHE_STOREDDATA_URLS):
            storeddata_data = {"urls": [{"url": link} for link in new_urls]}
        else:
            with open(
                FILEPATH_CACHE_STOREDDATA_URLS, "r", encoding="utf-8"
            ) as storeddata_urls:
                try:
                    storeddata_data = json.load(storeddata_urls)
                except json.JSONDecodeError:
                    print(f"Error decoding JSON from {FILEPATH_RSS_URLS}")
                    return

            storeddata_data["urls"].extend({"url": link} for link in new_urls)

        with open(
            FILEPATH_CACHE_STOREDDATA_URLS, "w", encoding="utf-8"
        ) as storeddata_urls:
            json.dump(storeddata_data, storeddata_urls, indent=2)

        logging.info("[URL] Cleaning %s...", FILEPATH_ARTICLE_URLS)
        with open(FILEPATH_ARTICLE_URLS, "w", encoding="utf-8") as article_urls:
            article_data = {"urls": []}
            json.dump(article_data, article_urls, indent=2)

    else:
        logging.info("[URL] Nothing to do...")


def create_urllist_with_rss() -> None:
    """
    Creates a list of website URLs from RSS feeds and saves it to a JSON file.
    Returns:
        None: This function does not return anything.
    """
    rss_reader = download_loader("RssReader")
    reader = rss_reader()

    logging.info("[RSS] Loading RSS list...")

    if not os.path.exists(FILEPATH_RSS_URLS):
        print(f"RSS URL file {FILEPATH_RSS_URLS} does not exist.")
        return
    else:
        with open(FILEPATH_RSS_URLS, "r", encoding="utf-8") as rss_urls:
            try:
                rss_data = json.load(rss_urls)
            except json.JSONDecodeError:
                print(f"Error decoding JSON from {FILEPATH_RSS_URLS}")
                return

    rss_list = [item["url"] for item in rss_data["urls"]]

    documents = reader.load_data(rss_list)

    website_urls = set()
    for doc in documents:
        doc_dict = doc.dict()
        website_urls.add(doc_dict["metadata"]["link"])

    if website_urls:
        logging.info("[RSS] Saving Website URSs to JSON...")

        if not os.path.exists(FILEPATH_ARTICLE_URLS):
            print(f"Article URL file {FILEPATH_ARTICLE_URLS} does not exist.")
            return
        else:
            with open(FILEPATH_ARTICLE_URLS, "r", encoding="utf-8") as article_urls:
                try:
                    urls_data = json.load(article_urls)
                except json.JSONDecodeError:
                    print(f"Error decoding JSON from {FILEPATH_ARTICLE_URLS}")
                    return

        urls_data["urls"].extend({"url": link} for link in website_urls)

        with open(FILEPATH_ARTICLE_URLS, "w", encoding="utf-8") as article_urls:
            json.dump(urls_data, article_urls, indent=2)
    else:
        logging.info("[RSS] Nothing to do...")


def get_unique_elements(storeddata_path, articledata_path) -> list[str]:
    """
    Extracts unique URLs from two JSON files.

    Args:
        storeddata_path (str): Path to the JSON file containing storeddata.
        articledata_path (str): Path to the JSON file containing article data.

    Returns:
        list[str]: List of new URLs. These are URLs from the article data that do not exist in the
        storeddata.
    """

    if not os.path.exists(storeddata_path):
        print(f"File {storeddata_path} does not exist.")
        existing_urls = set()
    else:
        with open(storeddata_path, "r", encoding="utf-8") as file_stored:
            try:
                data_stored = json.load(file_stored)

            except json.JSONDecodeError:
                print(f"Error decoding JSON from {storeddata_path}")
                return []

        existing_urls = set(item["url"] for item in data_stored["urls"])

    if not os.path.exists(articledata_path):
        print(f"Article data file {articledata_path} does not exist.")
        new_urls = []
    else:
        with open(articledata_path, "r", encoding="utf-8") as file_article:
            try:
                data_article = json.load(file_article)
            except json.JSONDecodeError:
                print(f"Error decoding JSON from {articledata_path}")
                return []
        url_list = [item["url"] for item in data_article["urls"]]
        new_urls = [url for url in url_list if url not in existing_urls]

    return new_urls


def remove_duplicate_elements(json_file: str, key: str) -> None:
    """
    Removes duplicate URLs from a JSON file.

    Args:
        json_file (str): Path to the JSON file.
        key (str): The key in the JSON file to remove duplicates from.

    Returns:
        None: This function does not return anything.
    """

    if not os.path.exists(json_file):
        print(f"File {json_file} does not exist.")
        return

    with open(json_file, "r", encoding="utf-8") as file:
        try:
            data = json.load(file)
        except json.JSONDecodeError:
            print(f"Error decoding JSON from {json_file}")
            return

    if key not in data:
        print(f"Key {key} not found in {json_file}")
        return

    unique_elements = list({v["url"]: v for v in data[key]}.values())

    filtered_data = {key: unique_elements}

    with open(json_file, "w", encoding="utf-8") as file:
        json.dump(filtered_data, file, indent=2)
