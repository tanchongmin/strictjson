import inspect
import numpy as np

from strictjson.utils import ensure_awaitable


class BaseRanker:
    ''' Base class that defines shared properties for Rankers '''
    def __init__(self, model="text-embedding-3-small", ranking_fn=None, database=None):
        if database is None:
            database = {}
        self.model = model
        # For AsyncRanker ranking_fn must be async if provided
        self.ranking_fn = ranking_fn
        self.database = database

    def get_python_representation(self):
        return f"{self.__class__.__name__}(model='{self.model}', ranking_fn={inspect.getsource(self.ranking_fn) if self.ranking_fn else None})"
    
    
class Ranker(BaseRanker):
    def __call__(self, query, key) -> float:
        query, key = str(query), str(key)
        if self.ranking_fn is None:

            from openai import OpenAI
            client = OpenAI()
            query_embedding = self.get_or_create_embedding(query, client)
            key_embedding = self.get_or_create_embedding(key, client)
            return np.dot(query_embedding, key_embedding)
        else:
            return self.ranking_fn(query, key)

    def get_or_create_embedding(self, text, client):
        if text in self.database:
            return self.database[text]
        else:
            cleaned_text = text.replace("\n", " ")
            embedding = client.embeddings.create(input=[cleaned_text], model=self.model).data[0].embedding
            self.database[text] = embedding
            return embedding
        
class AsyncRanker(BaseRanker):
    
    async def __call__(self, query, key) -> float:
        query, key = str(query), str(key)
        if self.ranking_fn is None:

            from openai import AsyncOpenAI
            client = AsyncOpenAI()
            query_embedding = await self.get_or_create_embedding_async(query, client)
            key_embedding = await self.get_or_create_embedding_async(key, client)
            return np.dot(query_embedding, key_embedding)
        else:
            ensure_awaitable(self.ranking_fn, 'ranking_fn')
            return await self.ranking_fn(query, key)

    async def get_or_create_embedding_async(self, text, client):
        if text in self.database:
            return self.database[text]
        else:
            cleaned_text = text.replace("\n", " ")
            response = await client.embeddings.create(input=[cleaned_text], model=self.model)
            embedding = response.data[0].embedding
            self.database[text] = embedding
            return embedding