import argparse
import logging
import sys

from llama_index.response.schema import RESPONSE_TYPE

from construct_index import (
    construct_index_with_file,
    load_variables,
    query_with_index,
    update_index_with_rss,
)

logging.disable()
logging.basicConfig(stream=sys.stdout, level=logging.INFO, force=True)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))


def show_response(response: RESPONSE_TYPE, user_input):
    """
    This function takes a response and user input, and displays the response in a formatted manner.
    It first prints the user query and the answer. Then it prints each source node's id and score.
    If a node has storeddata, it prints the reference source, URL, link, and index position.
    If a node has a score, it prints the cosine similarity.

    Args:
        response (RESPONSE_TYPE): The response to be displayed.
        user_input (str): The user's input query.
    """

    print("==========")
    print("Query:")
    print(user_input)
    print("Answer:")
    print(response)
    print("==========\n")

    node_list = response.source_nodes
    node_count = len(node_list)
    logging.debug("node_count: %s", node_count)

    for node in node_list:
        print(f"{node.node.id_=}, {node.score=}")

    for node in node_list:
        node_dict = node.dict()
        print("----------")
        if node.node.metadata is not None:
            if "file_name" in node.node.metadata:
                print("Reference source:")
                print(f"{node_dict['node']['metadata']['file_name']}\n")

            if "URL" in node.node.metadata:
                print("Reference URL:")
                print(f"{node_dict['node']['metadata']['URL']}\n")

            if "link" in node.node.metadata:
                print("Reference Link:")
                print(f"{node_dict['node']['metadata']['link']}\n")

            print("Index position:")
            print(
                f"start_char_idx={node_dict['node']['start_char_idx']}, end_char_idx={node_dict['node']['end_char_idx']}\n"
            )

        if node.score is not None:
            print("Cosine Similarity:")
            print(f"{node_dict['score']}\n")

        # print("Reference text:")
        # print(f"{node_dict['node']['text']}")
        # print("----------\n")


def main():
    """
    Construct indexes with files. (You need to execute first to create indexes directory)
        $ python3 ./chatgpt_llamaindex.py.py -u files
    Construct indexes with RSS, URL.
        $ python3 ./chatgpt_llamaindex.py.py -u rss
    Execute query.
        $ python3 ./chatgpt_llamaindex.py.py
    """

    parser = argparse.ArgumentParser(description="Program with command line options.")
    parser.add_argument(
        "-u",
        "--update_index",
        choices=["files", "rss", "url"],
        required=False,
        help="Update indeces with files or rss, and exit.",
    )
    args = parser.parse_args()

    load_variables()
    if args.update_index == "rss" or args.update_index == "url":
        update_index_with_rss()
        sys.exit()
    elif args.update_index == "files":
        construct_index_with_file()
        sys.exit()

    while True:
        user_input = input("Input query:")
        if user_input == "exit":
            break
        else:
            response = query_with_index(user_input)

            show_response(response, user_input)


if __name__ == "__main__":
    main()
