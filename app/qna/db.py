# from redis import Redis
from typing import List

from langchain.schema import Document
# from langchain.vectorstores.redis import Redis as RedisVDB
from langchain.vectorstores import Chroma

from qna.llm import get_embeddings
# from qna.constants import CACHE_TYPE, REDIS_INDEX_NAME, REDIS_URL

# def get_cache():
    # construct cache implementation based on env var
    # if CACHE_TYPE == "semantic":
    #     from langchain.cache import RedisSemanticCache
    #     print("Using semantic cache")

    #     # TODO change to using huggingface embeddings
    #     # so that caching is cheaper and faster.
    #     embeddings = get_embeddings()
    #     return RedisSemanticCache(
    #         redis_url=REDIS_URL,
    #         embedding=embeddings,
    #         score_threshold=0.01,
        # )
    # return None



def get_vectorstore(documents: List[Document]=None) -> "Chroma":
    """Create the Redis vectorstore."""

    embeddings = get_embeddings()

    try:
        vectorstore = Chroma.from_existing_index(
            embedding=embeddings,
            # index_name=REDIS_INDEX_NAME,
            # redis_url=REDIS_URL
        )
        return vectorstore
    except:
        pass

    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        # persist_directory='/save'
        # index_name=REDIS_INDEX_NAME,
        # redis_url=REDIS_URL
    )
    return vectorstore