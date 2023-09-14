# Copyright (c) Microsoft. All rights reserved.
from __future__ import annotations

import time
import uuid
from logging import Logger
from typing import Any, List, Mapping, Optional, Tuple

from motor import MotorCommandCursor, motor_asyncio
from numpy import ndarray
from pymongo import DeleteOne, ReadPreference, UpdateOne, results
from pymongo.operations import SearchIndexModel
from semantic_kernel.connectors.memory.mongodb_atlas_vector_search.utils import (
    _DEFAULT_SEARCH_INDEX_NAME,
    MONGODB_FIELD_EMBEDDING,
    MONGODB_FIELD_ID,
    document_to_memory_record,
    generate_search_index_model,
    memory_record_to_mongo_document,
)
from semantic_kernel.memory.memory_record import MemoryRecord
from semantic_kernel.memory.memory_store_base import MemoryStoreBase
from semantic_kernel.utils.null_logger import NullLogger
from semantic_kernel.utils.settings import mongodb_atlas_settings_from_dot_env

_DEFAULT_DB_NAME = "default"
_POLL_INTERVAL = 1.0


class MongoDBAtlasVectorSearchMemoryStore(MemoryStoreBase):
    """Memory Store for MongoDB Atlas Vector Search Connections"""

    __slots__ = ("_mongo_client", "_logger", "__database_name", "__search_index_model")

    _mongo_client: motor_asyncio.AsyncIOMotorClient
    _logger: Logger
    __database_name: str
    __search_index_model: SearchIndexModel

    def __init__(
        self,
        dimensions: int,
        connection_string: Optional[str] = None,
        similarity: Optional[str] = None,
        database_name: Optional[str] = None,
        logger: Optional[Logger] = None,
        read_preference: Optional[ReadPreference] = ReadPreference.PRIMARY,
    ):
        self._mongo_client = motor_asyncio.AsyncIOMotorClient(
            connection_string or mongodb_atlas_settings_from_dot_env(),
            read_preference=read_preference,
        )
        self._logger = logger or NullLogger()
        self.__database_name = database_name or _DEFAULT_DB_NAME
        self.__search_index_model = generate_search_index_model(dimensions, similarity)

    @property
    def search_index_model(self):
        return self.__search_index_model

    @property
    def database_name(self):
        return self.__database_name

    @property
    def database(self):
        return self._mongo_client[self.database_name]

    def _is_queryable(self, indices) -> bool:
        return indices and indices[0].get("queryable") is True

    async def wait_for_search_index_ready(
        self, collection_name: str, timeout: float = 60.0
    ) -> None:
        """Wait for a search index to be ready."""
        wait_time = timeout

        while not self._is_queryable(
            await self.database[collection_name]
            .list_search_indexes(_DEFAULT_SEARCH_INDEX_NAME)
            .to_list(length=1)
        ):
            if wait_time <= 0:
                raise TimeoutError(f"Index unavailable after waiting {timeout} seconds")
            time.sleep(_POLL_INTERVAL)
            wait_time -= _POLL_INTERVAL

    async def close_async(self):
        """Async close connection, invoked by MemoryStoreBase.__aexit__()"""
        if self._mongo_client:
            self._mongo_client.close()
            self._mongo_client = None

    async def create_collection_async(self, collection_name: str) -> None:
        """Creates a new collection in the data store.

        Arguments:
            collection_name {str} -- The name associated with a collection of embeddings.

        Returns:
            None
        """
        # Defining the search index enforces its creation
        if not await self.does_collection_exist_async(collection_name):
            await self.database.create_collection(collection_name)
            await self.database[collection_name].create_search_index(
                self.search_index_model
            )

    async def get_collections_async(
        self,
    ) -> List[str]:
        """Gets all collection names in the data store.

        Returns:
            List[str] -- A group of collection names.
        """
        return await self.database.list_collection_names()

    async def delete_collection_async(self, collection_name: str) -> None:
        """Deletes a collection from the data store.

        Arguments:
            collection_name {str} -- The name associated with a collection of embeddings.

        Returns:
            None
        """
        await self.database[collection_name].drop()

    async def does_collection_exist_async(self, collection_name: str) -> bool:
        """Determines if a collection exists in the data store.

        Arguments:
            collection_name {str} -- The name associated with a collection of embeddings.

        Returns:
            bool -- True if given collection exists, False if not.
        """
        return collection_name in (await self.get_collections_async())

    async def upsert_async(self, collection_name: str, record: MemoryRecord) -> str:
        """Upserts a memory record into the data store. Does not guarantee that the collection exists.
            If the record already exists, it will be updated.
            If the record does not exist, it will be created.

        Arguments:
            collection_name {str} -- The name associated with a collection of embeddings.
            record {MemoryRecord} -- The memory record to upsert.

        Returns:
            str -- The unique identifier for the memory record.
        """
        await self.create_collection_async(collection_name)

        if not record._id:
            record._id = str(uuid.uuid4())

        document: Mapping[str, Any] = memory_record_to_mongo_document(record)

        update_result: results.UpdateResult = await self.database[
            collection_name
        ].update_one(document, {"$set": document}, upsert=True)

        assert update_result.acknowledged
        return record._id

    async def upsert_batch_async(
        self, collection_name: str, records: List[MemoryRecord]
    ) -> List[str]:
        """Upserts a group of memory records into the data store. Does not guarantee that the collection exists.
            If the record already exists, it will be updated.
            If the record does not exist, it will be created.

        Arguments:
            collection_name {str} -- The name associated with a collection of embeddings.
            records {MemoryRecord} -- The memory records to upsert.

        Returns:
            List[str] -- The unique identifiers for the memory records.
        """
        await self.create_collection_async(collection_name)

        upserts: List[UpdateOne] = []
        for record in records:
            if not record._id:
                record._id = str(uuid.uuid4())
                self._logger.debug(
                    "record id not found for Memory Store, mutated with new uuid {}".format(
                        record._id
                    )
                )
            document = memory_record_to_mongo_document(record)
            upserts.append(UpdateOne(document, {"$set": document}, upsert=True))
        bulk_update_result: results.BulkWriteResult = await self.database[
            collection_name
        ].bulk_write(upserts, ordered=False)

        # Assert the number matched and the number upserted equal the total batch updated
        self._logger.debug(
            "matched_count={}, upserted_count={}".format(
                bulk_update_result.matched_count,
                bulk_update_result.upserted_count,
            )
        )
        assert (
            bulk_update_result.matched_count + bulk_update_result.upserted_count
            == len(records)
        )
        return [record._id for record in records]

    async def get_async(
        self, collection_name: str, key: str, with_embedding: bool
    ) -> MemoryRecord:
        """Gets a memory record from the data store. Does not guarantee that the collection exists.

        Arguments:
            collection_name {str} -- The name associated with a collection of embeddings.
            key {str} -- The unique id associated with the memory record to get.
            with_embedding {bool} -- If true, the embedding will be returned in the memory record.

        Returns:
            MemoryRecord -- The memory record if found
        """
        document = await self.database[collection_name].find_one(
            {MONGODB_FIELD_ID: key}
        )

        return document_to_memory_record(document, with_embedding) if document else None

    async def get_batch_async(
        self, collection_name: str, keys: List[str], with_embeddings: bool
    ) -> List[MemoryRecord]:
        """Gets a batch of memory records from the data store. Does not guarantee that the collection exists.

        Arguments:
            collection_name {str} -- The name associated with a collection of embeddings.
            keys {List[str]} -- The unique ids associated with the memory records to get.
            with_embeddings {bool} -- If true, the embedding will be returned in the memory records.

        Returns:
            List[MemoryRecord] -- The memory records associated with the unique keys provided.
        """
        results = self.database[collection_name].find({MONGODB_FIELD_ID: {"$in": keys}})

        return [
            document_to_memory_record(result, with_embeddings)
            for result in await results.to_list(length=len(keys))
        ]

    async def remove_async(self, collection_name: str, key: str) -> None:
        """Removes a memory record from the data store. Does not guarantee that the collection exists.

        Arguments:
            collection_name {str} -- The name associated with a collection of embeddings.
            key {str} -- The unique id associated with the memory record to remove.

        Returns:
            None
        """
        if not await self.does_collection_exist_async(collection_name):
            raise Exception(f"collection {collection_name} not found")
        await self.database[collection_name].delete_one({MONGODB_FIELD_ID: key})

    async def remove_batch_async(self, collection_name: str, keys: List[str]) -> None:
        """Removes a batch of memory records from the data store. Does not guarantee that the collection exists.

        Arguments:
            collection_name {str} -- The name associated with a collection of embeddings.
            keys {List[str]} -- The unique ids associated with the memory records to remove.

        Returns:
            None
        """
        if not await self.does_collection_exist_async(collection_name):
            raise Exception(f"collection {collection_name} not found")
        deletes: List[DeleteOne] = [DeleteOne({MONGODB_FIELD_ID: key}) for key in keys]
        bulk_write_result = await self.database[collection_name].bulk_write(
            deletes, ordered=False
        )
        self._logger.debug("{} entries deleted".format(bulk_write_result.deleted_count))

    async def get_nearest_matches_async(
        self,
        collection_name: str,
        embedding: ndarray,
        limit: int,
        with_embeddings: bool,
        min_relevance_score: float | None = None,
        pre_filter: dict[str, Any] | None = None,
        post_filter: dict[str, Any] | None = None,
    ) -> List[Tuple[MemoryRecord, float]]:
        """Gets the nearest matches to an embedding of type float. Does not guarantee that the collection exists.

        Arguments:
            collection_name {str} -- The name associated with a collection of embeddings.
            embedding {ndarray} -- The embedding to compare the collection's embeddings with.
            limit {int} -- The maximum number of similarity results to return, defaults to 1.
            min_relevance_score {float} -- The minimum relevance threshold for returned results.
            with_embeddings {bool} -- If true, the embeddings will be returned in the memory records.
            pre_filter {dict[str, Any] | None} -- Mongo Query to filter search information during vector search
                in pipeline, defaults to None
            post_filter {dict[str, Any] | None} -- Additional aggregation pipeline stages to conduct after
                initial vector search, defaults to None

        Returns:
            List[Tuple[MemoryRecord, float]] -- A list of tuples where item1 is a MemoryRecord and item2
                is its similarity score as a float.
        """
        pipeline: list[dict[str, Any]] = []
        vector_search_query: List[Mapping[str, Any]] = {
            "$search": {
                "index": _DEFAULT_SEARCH_INDEX_NAME,
                "knnBeta": {
                    "vector": embedding.tolist(),
                    "k": limit,
                    "path": MONGODB_FIELD_EMBEDDING,
                },
            }
        }

        if pre_filter:
            vector_search_query["$search"]["knnBeta"]["filter"] = pre_filter

        pipeline.append(vector_search_query)
        # add meta search scoring
        pipeline.append({"$set": {"score": {"$meta": "searchScore"}}})

        if min_relevance_score:
            pipeline.append({"$match": {"$gte": ["$score", min_relevance_score]}})

        if post_filter:
            pipeline.append(post_filter)

        cursor: MotorCommandCursor = self.database[collection_name].aggregate(pipeline)

        return [
            (
                document_to_memory_record(doc, with_embeddings=with_embeddings),
                doc["score"],
            )
            for doc in await cursor.to_list(length=limit)
        ]

    async def get_nearest_match_async(
        self,
        collection_name: str,
        embedding: ndarray,
        with_embedding: bool,
        min_relevance_score: float | None = None,
        pre_filter: dict[str, Any] | None = None,
        post_filter: dict[str, Any] | None = None,
    ) -> Tuple[MemoryRecord, float]:
        """Gets the nearest match to an embedding of type float. Does not guarantee that the collection exists.

        Arguments:
            collection_name {str} -- The name associated with a collection of embeddings.
            embedding {ndarray} -- The embedding to compare the collection's embeddings with.
            min_relevance_score {float} -- The minimum relevance threshold for returned result.
            with_embedding {bool} -- If true, the embeddings will be returned in the memory record.
            pre_filter {dict[str, Any] | None} -- Mongo Query to filter search information during vector search
                in pipeline, defaults to None
            post_filter {dict[str, Any] | None} -- Additional aggregation pipeline stages to conduct after
                initial vector search, defaults to None

        Returns:
            Tuple[MemoryRecord, float] -- A tuple consisting of the MemoryRecord and the similarity score as a float.
        """
        matches: List[
            Tuple[MemoryRecord, float]
        ] = await self.get_nearest_matches_async(
            collection_name=collection_name,
            embedding=embedding,
            limit=1,
            min_relevance_score=min_relevance_score,
            with_embeddings=with_embedding,
            pre_filter=pre_filter,
            post_filter=post_filter,
        )

        return matches[0] if matches else None


__all__ = ["MongoDBAtlasVectorSearchMemoryStore"]
