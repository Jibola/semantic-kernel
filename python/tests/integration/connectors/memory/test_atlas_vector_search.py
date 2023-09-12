"""
Test Cases : 
- Test collection create 
- Test collection deletion
- Test client closure
- Test get collections
- Test collection existence
- Test singular upsert
- Test Bulk Upsert
- Test singular deletion
- Test bulk deletion
- Test singular get
- Test knn singular match
- Test knn multitude match

- Constraint Tests
  - Test 2048 limit enforcement
  - Test No Client searchable without index provided
"""
import time
import pytest
import pytest_asyncio
import numpy as np
import random

from pymongo import errors

from semantic_kernel.connectors.memory.mongodb_atlas_vector_search.atlas_vector_search import (
    MongoDBAtlasVectorSearchMemoryStore,
)
from semantic_kernel.memory.memory_record import MemoryRecord

mongodb_atlas_vector_search_installed: bool
try:
    import motor

    mongodb_atlas_vector_search_installed = True
except ImportError:
    mongodb_atlas_vector_search_installed = False

pytestmark = pytest.mark.skipif(
    not mongodb_atlas_vector_search_installed,
    reason="MongoDB Atlas Vector Search not installed; pip install motor",
)
password = "black"
CONNECTION_STRING = f"mongodb+srv://admin:black@buyblack-cluster.duf3c.azure.mongodb.net/?retryWrites=true&w=majority"
DUPLICATE_INDEX = 68


def is_equal_memory_record(mem1: MemoryRecord, mem2: MemoryRecord, with_embeddings):
    def dictify_memory_record(mem):
        return {k: v for k, v in mem.__dict__.items() if k != "_embedding"}

    assert dictify_memory_record(mem1) == dictify_memory_record(mem2)
    if with_embeddings:
        assert mem1._embedding.tolist() == mem2._embedding.tolist()


@pytest.fixture
def memory_record_gen():
    def memory_record(_id):
        return MemoryRecord(
            id=str(_id),
            text=f"{_id} text",
            is_reference=False,
            embedding=np.array([1 / (_id + 1)] * 4),
            description=f"{_id} description",
            external_source_name=f"{_id} external source",
            additional_metadata=f"{_id} additional metadata",
            timestamp=f"{_id} timestamp",
        )

    return memory_record


@pytest.fixture
def test_collection():
    return f"AVSTest-{random.randint(0,1000)}"


@pytest_asyncio.fixture
async def vector_search_store():
    async with MongoDBAtlasVectorSearchMemoryStore(
        connection_string=CONNECTION_STRING, dimensions=4
    ) as memory:
        # Delete all collections before and after
        for cname in await memory.get_collections_async():
            await memory.delete_collection_async(cname)

        try:
            yield memory
        except errors.OperationFailure as e:
            # DuplicateIndex failures in test are due to lag in index deletions
            if e.code != DUPLICATE_INDEX:
                raise
        finally:
            for cname in await memory.get_collections_async():
                await memory.delete_collection_async(cname)
            # TODO Possibly remove
            time.sleep(1)


@pytest.mark.asyncio
async def test_constructor(vector_search_store):
    assert isinstance(vector_search_store, MongoDBAtlasVectorSearchMemoryStore)


@pytest.mark.asyncio
async def test_collection_create_and_delete(vector_search_store, test_collection):
    await vector_search_store.create_collection_async(test_collection)
    assert await vector_search_store.does_collection_exist_async(test_collection)
    await vector_search_store.delete_collection_async(test_collection)
    assert not await vector_search_store.does_collection_exist_async(test_collection)


@pytest.mark.asyncio
async def test_collection_upsert(
    vector_search_store, test_collection, memory_record_gen
):
    mems = [memory_record_gen(i) for i in range(1, 4)]
    mem1 = await vector_search_store.upsert_async(test_collection, mems[0])
    assert mem1 == mems[0]._id


@pytest.mark.asyncio
async def test_collection_batch_upsert(
    vector_search_store, test_collection, memory_record_gen
):
    mems = [memory_record_gen(i) for i in range(1, 4)]
    mems_check = await vector_search_store.upsert_batch_async(test_collection, mems)
    assert [m._id for m in mems] == mems_check


@pytest.mark.asyncio
async def test_collection_deletion(
    vector_search_store, test_collection, memory_record_gen
):
    mem = memory_record_gen(1)
    await vector_search_store.upsert_async(test_collection, mem)
    insertion_val = await vector_search_store.get_async(test_collection, mem._id, True)
    assert mem._id == insertion_val._id
    assert mem._embedding.tolist() == insertion_val._embedding.tolist()
    assert insertion_val is not None
    await vector_search_store.remove_async(test_collection, mem._id)
    val = await vector_search_store.get_async(test_collection, mem._id, False)
    assert val is None


@pytest.mark.asyncio
async def test_collection_batch_deletion(
    vector_search_store, test_collection, memory_record_gen
):
    mems = [memory_record_gen(i) for i in range(1, 4)]
    await vector_search_store.upsert_batch_async(test_collection, mems)
    ids = [mem._id for mem in mems]
    insertion_val = await vector_search_store.get_batch_async(
        test_collection, ids, True
    )
    assert len(insertion_val) == len(mems)
    await vector_search_store.remove_batch_async(test_collection, ids)
    assert not await vector_search_store.get_batch_async(test_collection, ids, False)


@pytest.mark.asyncio
async def test_collection_get(vector_search_store, test_collection, memory_record_gen):
    mem = memory_record_gen(1)
    await vector_search_store.upsert_async(test_collection, mem)
    insertion_val = await vector_search_store.get_async(test_collection, mem._id, False)
    is_equal_memory_record(mem, insertion_val, False)

    refetched_record = await vector_search_store.get_async(
        test_collection, mem._id, True
    )
    is_equal_memory_record(mem, refetched_record, True)


@pytest.mark.asyncio
async def test_collection_batch_get(
    vector_search_store, test_collection, memory_record_gen
):
    mems = {str(i): memory_record_gen(i) for i in range(1, 4)}
    await vector_search_store.upsert_batch_async(test_collection, list(mems.values()))
    insertion_val = await vector_search_store.get_batch_async(
        test_collection, list(mems.keys()), False
    )
    assert len(insertion_val) == len(mems)
    for val in insertion_val:
        is_equal_memory_record(mems[val._id], val, False)

    refetched_vals = await vector_search_store.get_batch_async(
        test_collection, list(mems.keys()), True
    )
    for ref in refetched_vals:
        is_equal_memory_record(mems[ref._id], ref, True)


@pytest.mark.asyncio
async def test_collection_knn_match(
    vector_search_store, test_collection, memory_record_gen
):
    mem = memory_record_gen(3)
    await vector_search_store.upsert_async(test_collection, mem)
    time.sleep(10)
    result, score = await vector_search_store.get_nearest_match_async(
        collection_name=test_collection,
        embedding=mem._embedding,
        with_embedding=True,
    )
    is_equal_memory_record(mem, result, True)
    assert score


@pytest.mark.asyncio
async def test_collection_knn_matches(
    vector_search_store, test_collection, memory_record_gen
):
    mems = {str(i): memory_record_gen(i) for i in range(1, 10)}
    await vector_search_store.upsert_batch_async(test_collection, list(mems.values()))
    time.sleep(10)
    assert await vector_search_store.does_collection_exist_async(test_collection)
    results_and_scores = await vector_search_store.get_nearest_matches_async(
        collection_name=test_collection,
        embedding=mems["3"]._embedding,
        limit=4,
        with_embeddings=True,
    )
    assert len(results_and_scores) == 4
    scores = [score for _, score in results_and_scores]
    assert scores == sorted(scores, reverse=True)
    for result, score in results_and_scores:
        is_equal_memory_record(mems[result._id], result, True)


# TODO: TEST PRE & POST FILTERS
