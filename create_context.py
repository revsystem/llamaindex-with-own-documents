import logging

import tiktoken
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index import LangchainEmbedding, LLMPredictor, PromptHelper, ServiceContext
from llama_index.callbacks import CallbackManager, LlamaDebugHandler
from llama_index.llms import OpenAI
from llama_index.logger.base import LlamaLogger
from llama_index.node_parser import SimpleNodeParser
from llama_index.text_splitter import TokenTextSplitter


def get_service_context() -> ServiceContext:
    """
    Ref:
    - https://gpt-index.readthedocs.io/en/latest/core_modules/supporting_modules/service_context.html
    """
    logging.info("Get Service Context.")
    # for debug
    llama_debug_handler = LlamaDebugHandler()
    callback_manager = CallbackManager([llama_debug_handler])

    # Customize the LLM
    # Language model for obtaining textual responses (Completion)
    # llm = ChatOpenAI(
    #     model="gpt-3.5-turbo-0613",
    #     temperature=1.0,
    #     max_tokens=512,
    # )

    llm = OpenAI(
        model="gpt-3.5-turbo-0613",
        temperature=1.0,
        max_tokens=512,
    )

    llm_predictor = LLMPredictor(llm=llm)

    # Customize behavior of splitting into chunks
    # ref: https://gpt-index.readthedocs.io/en/stable/core_modules/data_modules/node_parsers/usage_pattern.html#text-splitter-customization
    text_splitter = TokenTextSplitter(
        separator=" ",
        chunk_size=1024,
        chunk_overlap=20,
        backup_separators=["\n\n", "\n", "。", "。 ", "、", "、 "],
        tokenizer=tiktoken.encoding_for_model("gpt-3.5-turbo").encode,
    )

    # Split text into chunks and create nodes
    node_parser = SimpleNodeParser.from_defaults(text_splitter=text_splitter)

    # Customize Embedded Models
    embed_model = LangchainEmbedding(
        HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")
    )

    # Split text to meet token count limit on LLM side
    prompt_helper = PromptHelper(
        context_window=3900,  # default
        num_output=256,  # default
        chunk_overlap_ratio=0.1,  # default
        chunk_size_limit=None,  # default
    )

    service_context = ServiceContext.from_defaults(
        callback_manager=callback_manager,
        llama_logger=LlamaLogger(),
        node_parser=node_parser,
        embed_model=embed_model,
        llm_predictor=llm_predictor,
        prompt_helper=prompt_helper,
    )

    return service_context
