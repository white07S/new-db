"""
Qdrant Vector Store Provider

This module provides a reusable Qdrant client for vector storage operations.
Supports CRUD operations and advanced search functionality.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import uuid
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams,
    Distance,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    SearchRequest,
    SearchParams,
    UpdateStatus,
    CollectionInfo,
    HasIdCondition,
    MatchAny,
    Range,
    PointIdsList,
    ScoredPoint,
)

try:
    from qdrant_client.models import QueryRequest
except ImportError:  # Older qdrant-client versions
    QueryRequest = None
from qdrant_client.http import models as rest
from common.logger import get_logger

logger = get_logger(__name__)

@dataclass
class QdrantConfig:
    """Configuration for Qdrant connection."""
    host: str = "localhost"
    port: int = 6333
    grpc_port: int = 6334
    prefer_grpc: bool = True
    api_key: Optional[str] = None
    https: bool = False
    timeout: int = 30

    @property
    def url(self) -> str:
        """Get the connection URL."""
        protocol = "https" if self.https else "http"
        return f"{protocol}://{self.host}:{self.port}"


class QdrantProvider:
    """
    Qdrant Vector Store Provider for managing vector collections and operations.

    This class provides a high-level interface for:
    - Collection management (create, delete, list)
    - Document indexing (upsert, delete)
    - Vector search (similarity, hybrid)
    - Metadata filtering
    """

    def __init__(self, config: Optional[QdrantConfig] = None):
        """
        Initialize Qdrant provider.

        Args:
            config: Qdrant configuration. Defaults to localhost connection.
        """
        self.config = config or QdrantConfig()
        self.client = self._initialize_client()
        self._supports_legacy_search = hasattr(self.client, "search")
        self._supports_legacy_batch_search = hasattr(self.client, "search_batch")
        logger.info(f"Initialized Qdrant client", extra={"props": {
            "host": self.config.host,
            "port": self.config.port,
            "grpc": self.config.prefer_grpc
        }})

    def _initialize_client(self) -> QdrantClient:
        """Initialize the Qdrant client."""
        if self.config.prefer_grpc:
            return QdrantClient(
                host=self.config.host,
                grpc_port=self.config.grpc_port,
                api_key=self.config.api_key,
                timeout=self.config.timeout,
                prefer_grpc=True
            )
        else:
            return QdrantClient(
                url=self.config.url,
                api_key=self.config.api_key,
                timeout=self.config.timeout,
                prefer_grpc=False
            )

    # ==================== Collection Management ====================

    def create_collection(
        self,
        collection_name: str,
        vector_size: int,
        distance_metric: str = "cosine",
        on_disk: bool = False,
        recreate_if_exists: bool = False
    ) -> bool:
        """
        Create a new collection in Qdrant.

        Args:
            collection_name: Name of the collection
            vector_size: Dimension of vectors
            distance_metric: Distance metric (cosine, euclidean, dot)
            on_disk: Store vectors on disk instead of RAM
            recreate_if_exists: Delete existing collection if it exists

        Returns:
            bool: True if collection was created successfully
        """
        try:
            # Check if collection exists
            if self.collection_exists(collection_name):
                if recreate_if_exists:
                    logger.warning(f"Collection {collection_name} exists, recreating")
                    self.delete_collection(collection_name)
                else:
                    logger.info(f"Collection {collection_name} already exists")
                    return True

            # Map distance metric
            distance_map = {
                "cosine": Distance.COSINE,
                "euclidean": Distance.EUCLID,
                "dot": Distance.DOT
            }
            distance = distance_map.get(distance_metric, Distance.COSINE)

            # Create collection
            result = self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=distance,
                    on_disk=on_disk
                )
            )

            logger.info(f"Created collection {collection_name}", extra={"props": {
                "vector_size": vector_size,
                "distance": distance_metric,
                "on_disk": on_disk
            }})
            return result

        except Exception as e:
            logger.error(f"Failed to create collection {collection_name}: {e}")
            raise

    def delete_collection(self, collection_name: str) -> bool:
        """
        Delete a collection.

        Args:
            collection_name: Name of the collection to delete

        Returns:
            bool: True if collection was deleted successfully
        """
        try:
            result = self.client.delete_collection(collection_name)
            logger.info(f"Deleted collection {collection_name}")
            return result
        except Exception as e:
            logger.error(f"Failed to delete collection {collection_name}: {e}")
            raise

    def collection_exists(self, collection_name: str) -> bool:
        """
        Check if a collection exists.

        Args:
            collection_name: Name of the collection

        Returns:
            bool: True if collection exists
        """
        try:
            collections = self.client.get_collections()
            return any(col.name == collection_name for col in collections.collections)
        except Exception as e:
            logger.error(f"Failed to check collection existence: {e}")
            raise

    def get_collection_info(self, collection_name: str) -> Optional[CollectionInfo]:
        """
        Get information about a collection.

        Args:
            collection_name: Name of the collection

        Returns:
            CollectionInfo or None if collection doesn't exist
        """
        try:
            return self.client.get_collection(collection_name)
        except Exception as e:
            logger.error(f"Failed to get collection info for {collection_name}: {e}")
            return None

    def list_collections(self) -> List[str]:
        """
        List all collections.

        Returns:
            List of collection names
        """
        try:
            collections = self.client.get_collections()
            return [col.name for col in collections.collections]
        except Exception as e:
            logger.error(f"Failed to list collections: {e}")
            raise

    # ==================== Document Operations ====================

    def upsert_documents(
        self,
        collection_name: str,
        documents: List[Dict[str, Any]],
        embeddings: List[List[float]],
        ids: Optional[List[Union[str, int]]] = None,
        batch_size: int = 100
    ) -> List[Union[str, int]]:
        """
        Insert or update documents in a collection.

        Args:
            collection_name: Name of the collection
            documents: List of document metadata dictionaries
            embeddings: List of embedding vectors
            ids: Optional list of document IDs (auto-generated if not provided)
            batch_size: Number of documents to upsert in one batch

        Returns:
            List of document IDs
        """
        try:
            if len(documents) != len(embeddings):
                raise ValueError(f"Number of documents ({len(documents)}) must match embeddings ({len(embeddings)})")

            # Generate IDs if not provided
            if ids is None:
                ids = [str(uuid.uuid4()) for _ in range(len(documents))]
            elif len(ids) != len(documents):
                raise ValueError(f"Number of IDs ({len(ids)}) must match documents ({len(documents)})")

            # Convert string IDs to UUIDs for Qdrant
            point_ids = []
            for id_val in ids:
                if isinstance(id_val, str):
                    try:
                        # Try to parse as UUID
                        point_ids.append(str(uuid.UUID(id_val)))
                    except ValueError:
                        # If not a valid UUID, create one from the string
                        point_ids.append(str(uuid.uuid5(uuid.NAMESPACE_DNS, id_val)))
                else:
                    point_ids.append(str(id_val))

            # Create points
            points = []
            for idx, (doc_id, embedding, metadata) in enumerate(zip(point_ids, embeddings, documents)):
                points.append(
                    PointStruct(
                        id=doc_id,
                        vector=embedding,
                        payload=metadata
                    )
                )

            # Upsert in batches
            for i in range(0, len(points), batch_size):
                batch = points[i:i + batch_size]
                self.client.upsert(
                    collection_name=collection_name,
                    points=batch
                )
                logger.debug(f"Upserted batch {i//batch_size + 1} of {(len(points)-1)//batch_size + 1}")

            logger.info(f"Upserted {len(points)} documents to {collection_name}")
            return point_ids

        except Exception as e:
            logger.error(f"Failed to upsert documents: {e}")
            raise

    def delete_documents(
        self,
        collection_name: str,
        ids: Optional[List[Union[str, int]]] = None,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Delete documents from a collection.

        Args:
            collection_name: Name of the collection
            ids: List of document IDs to delete
            filter_dict: Filter conditions to select documents to delete

        Returns:
            bool: True if deletion was successful
        """
        try:
            if ids:
                # Convert to proper format
                point_ids = []
                for id_val in ids:
                    if isinstance(id_val, str):
                        try:
                            point_ids.append(str(uuid.UUID(id_val)))
                        except ValueError:
                            point_ids.append(str(uuid.uuid5(uuid.NAMESPACE_DNS, id_val)))
                    else:
                        point_ids.append(str(id_val))

                result = self.client.delete(
                    collection_name=collection_name,
                    points_selector=PointIdsList(points=point_ids)
                )
                logger.info(f"Deleted {len(point_ids)} documents from {collection_name}")

            elif filter_dict:
                filter_obj = self._build_filter(filter_dict)
                result = self.client.delete(
                    collection_name=collection_name,
                    points_selector=filter_obj
                )
                logger.info(f"Deleted documents matching filter from {collection_name}")

            else:
                raise ValueError("Either ids or filter_dict must be provided")

            return result.status == UpdateStatus.COMPLETED

        except Exception as e:
            logger.error(f"Failed to delete documents: {e}")
            raise

    def get_document(
        self,
        collection_name: str,
        doc_id: Union[str, int]
    ) -> Optional[Dict[str, Any]]:
        """
        Retrieve a single document by ID.

        Args:
            collection_name: Name of the collection
            doc_id: Document ID

        Returns:
            Document metadata or None if not found
        """
        try:
            # Convert ID to proper format
            if isinstance(doc_id, str):
                try:
                    point_id = str(uuid.UUID(doc_id))
                except ValueError:
                    point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, doc_id))
            else:
                point_id = str(doc_id)

            result = self.client.retrieve(
                collection_name=collection_name,
                ids=[point_id],
                with_payload=True,
                with_vectors=False
            )

            if result:
                return result[0].payload
            return None

        except Exception as e:
            logger.error(f"Failed to get document {doc_id}: {e}")
            raise

    # ==================== Search Operations ====================

    def search(
        self,
        collection_name: str,
        query_vector: List[float],
        limit: int = 10,
        filter_dict: Optional[Dict[str, Any]] = None,
        score_threshold: Optional[float] = None,
        with_vectors: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Perform similarity search.

        Args:
            collection_name: Name of the collection
            query_vector: Query embedding vector
            limit: Maximum number of results
            filter_dict: Optional metadata filters
            score_threshold: Minimum similarity score
            with_vectors: Include vectors in results

        Returns:
            List of search results with metadata and scores
        """
        try:
            # Build filter if provided
            filter_obj = self._build_filter(filter_dict) if filter_dict else None

            # Perform search using the available client API
            if self._supports_legacy_search:
                raw_points = self.client.search(
                    collection_name=collection_name,
                    query_vector=query_vector,
                    limit=limit,
                    query_filter=filter_obj,
                    score_threshold=score_threshold,
                    with_payload=True,
                    with_vectors=with_vectors
                )
            else:
                response = self.client.query_points(
                    collection_name=collection_name,
                    query=query_vector,
                    query_filter=filter_obj,
                    limit=limit,
                    score_threshold=score_threshold,
                    with_payload=True,
                    with_vectors=with_vectors
                )
                raw_points = response.points if response else []

            # Format results
            formatted_results = []
            for point in raw_points or []:
                result = {
                    "id": point.id,
                    "score": point.score,
                    "metadata": point.payload
                }
                if with_vectors and point.vector:
                    result["vector"] = point.vector
                formatted_results.append(result)

            logger.debug(f"Search returned {len(formatted_results)} results from {collection_name}")
            return formatted_results

        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise

    def batch_search(
        self,
        collection_name: str,
        query_vectors: List[List[float]],
        limit: int = 10,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[List[Dict[str, Any]]]:
        """
        Perform batch similarity search for multiple queries.

        Args:
            collection_name: Name of the collection
            query_vectors: List of query embedding vectors
            limit: Maximum number of results per query
            filter_dict: Optional metadata filters

        Returns:
            List of search results for each query
        """
        try:
            # Build filter if provided
            filter_obj = self._build_filter(filter_dict) if filter_dict else None

            # Build requests for whichever API is available
            if self._supports_legacy_batch_search:
                requests = [
                    SearchRequest(
                        vector=vector,
                        limit=limit,
                        filter=filter_obj,
                        with_payload=True,
                        with_vector=False
                    )
                    for vector in query_vectors
                ]

                batch_results = self.client.search_batch(
                    collection_name=collection_name,
                    requests=requests
                )
            else:
                requests = [
                    QueryRequest(
                        query=vector,
                        limit=limit,
                        filter=filter_obj,
                        with_payload=True,
                        with_vector=False
                    )
                    for vector in query_vectors
                ]

                batch_results = self.client.query_batch_points(
                    collection_name=collection_name,
                    requests=requests
                )

            # Format results
            formatted_results = []
            for batch_result in batch_results:
                query_results = []
                points = getattr(batch_result, "points", batch_result)
                for point in points or []:
                    query_results.append({
                        "id": point.id,
                        "score": point.score,
                        "metadata": point.payload
                    })
                formatted_results.append(query_results)

            logger.debug(f"Batch search for {len(query_vectors)} queries completed")
            return formatted_results

        except Exception as e:
            logger.error(f"Batch search failed: {e}")
            raise

    def scroll(
        self,
        collection_name: str,
        limit: int = 100,
        filter_dict: Optional[Dict[str, Any]] = None,
        offset: Optional[str] = None,
        with_vectors: bool = False
    ) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        """
        Scroll through documents in a collection.

        Args:
            collection_name: Name of the collection
            limit: Number of documents to retrieve
            filter_dict: Optional metadata filters
            offset: Offset ID for pagination
            with_vectors: Include vectors in results

        Returns:
            Tuple of (documents, next_offset)
        """
        try:
            # Build filter if provided
            filter_obj = self._build_filter(filter_dict) if filter_dict else None

            # Perform scroll
            records, next_page = self.client.scroll(
                collection_name=collection_name,
                limit=limit,
                scroll_filter=filter_obj,
                offset=offset,
                with_payload=True,
                with_vectors=with_vectors
            )

            # Format results
            documents = []
            for record in records:
                doc = {
                    "id": record.id,
                    "metadata": record.payload
                }
                if with_vectors and record.vector:
                    doc["vector"] = record.vector
                documents.append(doc)

            logger.debug(f"Scrolled {len(documents)} documents from {collection_name}")
            return documents, next_page

        except Exception as e:
            logger.error(f"Scroll failed: {e}")
            raise

    def count(
        self,
        collection_name: str,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Count documents in a collection.

        Args:
            collection_name: Name of the collection
            filter_dict: Optional metadata filters

        Returns:
            Number of documents
        """
        try:
            if filter_dict:
                filter_obj = self._build_filter(filter_dict)
                result = self.client.count(
                    collection_name=collection_name,
                    count_filter=filter_obj
                )
            else:
                collection_info = self.get_collection_info(collection_name)
                if collection_info:
                    result = collection_info.points_count
                else:
                    result = 0

            logger.debug(f"Collection {collection_name} has {result} documents")
            return result

        except Exception as e:
            logger.error(f"Count failed: {e}")
            raise

    # ==================== Helper Methods ====================

    def _build_filter(self, filter_dict: Dict[str, Any]) -> Filter:
        """
        Build a Qdrant filter from a dictionary.

        Supports:
        - Exact match: {"field": "value"}
        - List match: {"field": ["value1", "value2"]}
        - Range: {"field": {"$gte": 10, "$lte": 20}}
        - Nested: {"must": [...], "should": [...], "must_not": [...]}

        Args:
            filter_dict: Filter conditions dictionary

        Returns:
            Qdrant Filter object
        """
        conditions = []

        for key, value in filter_dict.items():
            # Handle logical operators
            if key in ["must", "should", "must_not"]:
                if isinstance(value, list):
                    nested_conditions = []
                    for item in value:
                        if isinstance(item, dict):
                            nested_conditions.append(self._build_filter(item))

                    if key == "must":
                        conditions.extend(nested_conditions)
                    elif key == "should":
                        conditions.append(Filter(should=nested_conditions))
                    elif key == "must_not":
                        conditions.append(Filter(must_not=nested_conditions))

            # Handle field conditions
            else:
                if isinstance(value, dict):
                    # Range query
                    range_cond = {}
                    if "$gte" in value:
                        range_cond["gte"] = value["$gte"]
                    if "$gt" in value:
                        range_cond["gt"] = value["$gt"]
                    if "$lte" in value:
                        range_cond["lte"] = value["$lte"]
                    if "$lt" in value:
                        range_cond["lt"] = value["$lt"]

                    if range_cond:
                        conditions.append(
                            FieldCondition(
                                key=key,
                                range=Range(**range_cond)
                            )
                        )

                elif isinstance(value, list):
                    # Match any of the values
                    conditions.append(
                        FieldCondition(
                            key=key,
                            match=MatchAny(any=value)
                        )
                    )

                else:
                    # Exact match
                    conditions.append(
                        FieldCondition(
                            key=key,
                            match=MatchValue(value=value)
                        )
                    )

        return Filter(must=conditions) if conditions else None

    def close(self):
        """Close the Qdrant client connection."""
        if hasattr(self.client, 'close'):
            self.client.close()
            logger.info("Closed Qdrant client connection")


# Convenience function for creating a provider instance
def create_qdrant_provider(
    host: str = "localhost",
    port: int = 6333,
    grpc_port: int = 6334,
    prefer_grpc: bool = True,
    api_key: Optional[str] = None
) -> QdrantProvider:
    """
    Create a Qdrant provider instance with the given configuration.

    Args:
        host: Qdrant server host
        port: HTTP port
        grpc_port: gRPC port
        prefer_grpc: Use gRPC for better performance
        api_key: Optional API key for authentication

    Returns:
        QdrantProvider instance
    """
    config = QdrantConfig(
        host=host,
        port=port,
        grpc_port=grpc_port,
        prefer_grpc=prefer_grpc,
        api_key=api_key
    )
    return QdrantProvider(config)
