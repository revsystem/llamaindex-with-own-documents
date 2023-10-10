# llamaindex-with-own-documents

Indexing and query your documents with LlamaIndex.

## Usage

### Indexing your PDF documents

Put your PDF documents into ./data/documents directory.
Then, indexing data.

```console
python3 ./llamaindex_indexing.py -u files
```

Then, index data will be in ./data/indexes/index.json

### Indexing Website using URL or RSS

When we want to store RSS URLs, put them into ./data/rss_url.json like the one below.

```json
{
    "urls": [
        {
            "url": "https://www.formula1.com/content/fom-website/en/latest/all.xml"
        }
    ]
}
```

And/or, if we want to store Website URLs, put them into ./data/article_url.json like the below. The URL is expanded into a JSON file when you execute the command described below.

```json
{
    "urls": [
        {
            "url": "https://www.formula1.com/en/latest/article.pirelli-to-continue-as-formula-1s-exclusive-tyre-supplier-until-2027.7xJIxJyMe84N3p7k4iIMjK.html"
        }
    ]
}
```

```console
python3 ./llamaindex_indexing.py -u rss
```

```console
Input query: <INPUT_YOUR_QUERY>
```

### Execute query

```console
python3 ./llamaindex_indexing.py
```

### Response

We can get a streaming answer like the ChatGPT.

```console
==========
Query:
<QUERY_YOU_INPUTED>
Answer:
<ANSWER_FROM_AI>
==========

node.node.id_='876f8bdb-xxxx-xxxx-xxxx-xxxxxxxxxxxx', node.score=0.8484xxxxxxxxxxxxxx
----------

Cosine Similarity:
0.84xxxxxxxxxxxxxx

Reference text:
<THE_PART_AI_REFERRED_TO>
```

#### When you exit the console, input 'exit'

```console
Input query: exit
```

## Setup

### Recommended System Requirements

- Python 3.11 or higher.

### Setup venv environment

To create a venv environment and activate:

```console
python3 -m venv .venv
source .venv/bin/activate
```

To deactivate:

```console
deactivate
```

### Setup Python environment

```console
pip3 install --upgrade pip
pip3 install -r requirements.txt
```

The main libraries installed are as follows:

```console
pip freeze | grep -e "openai" -e "pypdf" -e "llama-index" -e "tiktoken"

llama-index==0.8.42
openai==0.28.1
pypdf==3.16.3
tiktoken==0.5.1
```

### Requirement OpenAI API Key

Set your API Key to environment variables or shell dotfile like '.zshenv':

```console
export OPENAI_API_KEY= 'YOUR_OPENAI_API_KEY'
```

## Reference

- [Experimenting LlamaIndex RouterQueryEngine with Document Management | by Wenqi Glantz | Better Programming](https://betterprogramming.pub/experimenting-llamaindex-routerqueryengine-with-document-management-19b17f2e3a32)
- [LlamaIndex の RouterQueryEngine を試す](https://note.com/npaka/n/n0a068497ac96)
- [Router Query Engine - LlamaIndex 🦙 0.8.4](https://gpt-index.readthedocs.io/en/latest/examples/query_engine/RouterQueryEngine.html)
- [ServiceContext - LlamaIndex 🦙 0.8.4](https://gpt-index.readthedocs.io/en/latest/core_modules/supporting_modules/service_context.html)
