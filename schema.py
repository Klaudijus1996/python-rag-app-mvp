from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Dict, Any
from enum import Enum


class QueryType(str, Enum):
    """Supported query types for the RAG system."""
    RECOMMENDATION = "recommendation"
    COMPARISON = "comparison"
    COMPLEMENT = "complement"
    INFORMATION = "information"


class ChatRequest(BaseModel):
    """Request model for chat endpoint."""
    session_id: str = Field(
        ..., 
        description="Unique identifier for the user session",
        min_length=1,
        max_length=100
    )
    query: str = Field(
        ..., 
        description="User's question or request",
        min_length=1,
        max_length=1000
    )
    query_type: Optional[QueryType] = Field(
        None,
        description="Optional hint about the type of query"
    )
    max_products: Optional[int] = Field(
        5,
        description="Maximum number of products to return",
        ge=1,
        le=20
    )
    price_range: Optional[Dict[str, float]] = Field(
        None,
        description="Optional price filtering with 'min' and 'max' keys"
    )
    category_filter: Optional[str] = Field(
        None,
        description="Optional category to filter results"
    )

    @field_validator('query')
    @classmethod
    def query_not_empty(cls, v):
        if not v.strip():
            raise ValueError('Query cannot be empty')
        return v.strip()

    @field_validator('price_range')
    @classmethod
    def validate_price_range(cls, v):
        if v is not None:
            if 'min' in v and 'max' in v and v['min'] > v['max']:
                raise ValueError('Minimum price cannot be greater than maximum price')
        return v


class ProductInfo(BaseModel):
    """Structured product information."""
    product_id: str
    name: str
    brand: str
    category: str
    sub_category: Optional[str] = None
    price: float
    type: Optional[str] = None
    rating: Optional[float] = None
    description: str
    url: Optional[str] = None
    image_url: Optional[str] = None
    relevance_score: Optional[float] = None
    reasons: Optional[List[str]] = None


class ChatResponse(BaseModel):
    """Response model for chat endpoint."""
    answer: str = Field(
        ...,
        description="Generated response to the user's query"
    )
    products: Optional[List[ProductInfo]] = Field(
        None,
        description="List of relevant products mentioned in the response"
    )
    query_type: Optional[QueryType] = Field(
        None,
        description="Detected or provided query type"
    )
    sources_count: Optional[int] = Field(
        None,
        description="Number of source documents used for the response"
    )
    session_id: str = Field(
        ...,
        description="Session identifier for context"
    )


class IngestRequest(BaseModel):
    """Request model for ingestion endpoint."""
    force_reindex: Optional[bool] = Field(
        False,
        description="Whether to force reindexing even if index exists"
    )
    chunk_size: Optional[int] = Field(
        1000,
        description="Text chunk size for document splitting",
        ge=100,
        le=2000
    )
    chunk_overlap: Optional[int] = Field(
        200,
        description="Overlap between text chunks",
        ge=0,
        le=500
    )


class IngestResponse(BaseModel):
    """Response model for ingestion endpoint."""
    status: str = Field(
        ...,
        description="Status of the ingestion process"
    )
    chunks_indexed: int = Field(
        ...,
        description="Number of document chunks indexed"
    )
    products_processed: Optional[int] = Field(
        None,
        description="Number of products processed from source data"
    )
    index_path: str = Field(
        ...,
        description="Path where the vector index is stored"
    )
    embedding_model: str = Field(
        ...,
        description="Name of the embedding model used"
    )
    processing_time_seconds: Optional[float] = Field(
        None,
        description="Time taken for the ingestion process"
    )


class HealthResponse(BaseModel):
    """Response model for health endpoint."""
    status: str = Field("healthy", description="Service health status")
    version: str = Field("1.0.0", description="Application version")
    index_ready: bool = Field(..., description="Whether vector index is available")
    embedding_model: str = Field(..., description="Current embedding model")
    total_chunks: Optional[int] = Field(None, description="Total chunks in index")


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str = Field(..., description="Error message")
    error_code: str = Field(..., description="Error code for programmatic handling")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")


class SessionInfo(BaseModel):
    """Information about a user session."""
    session_id: str
    created_at: str
    last_activity: str
    message_count: int
    total_queries: int