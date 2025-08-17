import os
import logging
from typing import Dict, Any, List, Optional
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough, RunnableMap
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.documents import Document
from schema import QueryType, ProductInfo

# Configuration
INDEX_DIR = "store/faiss"
MODEL_CHAT = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
MODEL_EMBED = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
TOP_K = int(os.getenv("RAG_TOP_K_RESULTS", "5"))
SIMILARITY_THRESHOLD = float(os.getenv("RAG_SIMILARITY_THRESHOLD", "0.7"))

# Setup logging
logger = logging.getLogger(__name__)


def load_retriever(k: int = TOP_K, search_type: str = "mmr"):
    """Load and configure the document retriever."""
    
    try:
        embeddings = OpenAIEmbeddings(model=MODEL_EMBED)
        vector_store = FAISS.load_local(
            INDEX_DIR, 
            embeddings, 
            allow_dangerous_deserialization=True
        )
        
        # Configure retrieval strategy
        if search_type == "mmr":
            retriever = vector_store.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": k,
                    "fetch_k": min(20, 4 * k),
                    "lambda_mult": 0.4  # Diversity vs relevance tradeoff
                }
            )
        else:
            retriever = vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": k}
            )
        
        logger.info(f"Retriever loaded successfully with {search_type} search")
        return retriever
        
    except Exception as e:
        logger.error(f"Failed to load retriever: {e}")
        raise


def detect_query_type(query: str) -> QueryType:
    """Detect the type of query based on keywords and patterns."""
    
    query_lower = query.lower()
    
    # Comparison keywords
    comparison_keywords = [
        "compare", "vs", "versus", "difference", "better", "which",
        "between", "against", "comparison"
    ]
    
    # Complement keywords - match patterns more flexibly
    complement_keywords = [
        "complement", "go with", "goes with", "pair", "pairs", "accessories",
        "compatible", "works with", "work with", "bundle", "set"
    ]
    
    # Recommendation keywords
    recommendation_keywords = [
        "recommend", "suggest", "best", "good", "find", "looking for",
        "need", "want", "show me", "help me choose"
    ]
    
    if any(keyword in query_lower for keyword in comparison_keywords):
        return QueryType.COMPARISON
    elif any(keyword in query_lower for keyword in complement_keywords):
        return QueryType.COMPLEMENT
    elif any(keyword in query_lower for keyword in recommendation_keywords):
        return QueryType.RECOMMENDATION
    else:
        return QueryType.INFORMATION


def format_docs(docs: List[Document]) -> str:
    """Format retrieved documents for the language model."""
    
    if not docs:
        return "No relevant products found in the catalog."
    
    formatted_docs = []
    for i, doc in enumerate(docs, 1):
        metadata = doc.metadata or {}
        
        # Format product information
        product_info = f"""[Product {i}] {metadata.get('name', 'Unknown')} 
Brand: {metadata.get('brand', 'Unknown')}
Price: {metadata.get('price', 'Unknown')} {metadata.get('currency', '')}
Category: {metadata.get('category', 'Unknown')}
Product ID: {metadata.get('product_id', 'Unknown')}

{doc.page_content}

URL: {metadata.get('url', 'Not available')}
"""
        formatted_docs.append(product_info)
    
    return "\n" + "="*80 + "\n".join(formatted_docs)


def extract_products_from_docs(docs: List[Document]) -> List[ProductInfo]:
    """Extract structured product information from documents."""
    
    products = []
    for doc in docs:
        metadata = doc.metadata or {}
        
        try:
            product = ProductInfo(
                product_id=metadata.get('product_id', ''),
                name=metadata.get('name', ''),
                brand=metadata.get('brand', ''),
                category=metadata.get('category', ''),
                price=float(metadata.get('price', 0)),
                currency=metadata.get('currency', ''),
                description=doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                url=metadata.get('url'),
                image_url=metadata.get('image_url')
            )
            products.append(product)
        except Exception as e:
            logger.warning(f"Failed to extract product info: {e}")
            continue
    
    return products


# Enhanced system prompt for different query types
SYSTEM_PROMPT = """You are an expert retail shopping assistant specializing in product recommendations and comparisons.

CORE PRINCIPLES:
- Base ALL product facts on the provided catalog context
- If unsure about any detail, state "I don't have that information in the catalog"
- Provide concise, factual answers focused on helping customers make informed decisions
- Always explain price influencers when relevant: brand reputation, materials/quality, capacity/size, advanced features, warranty/support, limited editions

RESPONSE FORMATS by query type:

FOR RECOMMENDATIONS:
- Suggest 3-5 products maximum unless requested otherwise
- For each product include:
  • Product name (brand)
  • 1-2 key reasons why it fits their needs
  • Price in original currency
  • URL if available
- End with a brief "Why these picks" explanation

FOR COMPARISONS:
- Create a clear comparison table or structured format
- Focus on key differentiators: price, features, quality, use cases
- Highlight which option is better for specific needs
- Include objective facts only

FOR COMPLEMENTS/ACCESSORIES:
- Suggest compatible or commonly paired products
- Explain compatibility and usage scenarios
- Consider price range compatibility

FOR GENERAL INFORMATION:
- Provide factual details about specific products
- Explain features, specifications, and use cases
- Cite specific product details from catalog

CONSTRAINTS:
- No medical or financial advice claims
- No hallucinated product features or specifications
- If catalog lacks information, be explicit about limitations
- Keep responses focused and actionable
"""

# Create prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="history"),
    ("human", """Query Type: {query_type}
User Question: {question}

Catalog Context:
{context}

Please provide a helpful response based on the catalog information above."""),
])


def build_rag_chain(session_store: Dict[str, ChatMessageHistory]):
    """Build the complete RAG chain with memory."""
    
    try:
        # Initialize components
        retriever = load_retriever()
        llm = ChatOpenAI(
            model=MODEL_CHAT, 
            temperature=0.1,  # Low temperature for factual responses
            max_tokens=int(os.getenv("RAG_MAX_TOKENS_PER_RESPONSE", "1000"))
        )
        
        # Build the chain using LCEL
        rag_chain = RunnableMap({
            "question": RunnablePassthrough(),
            "docs": retriever,
        }).assign(
            context=lambda x: format_docs(x["docs"]),
            query_type=lambda x: detect_query_type(x["question"]).value
        ) | prompt | llm | StrOutputParser()
        
        # Add session-based memory
        def get_session_history(session_id: str) -> ChatMessageHistory:
            if session_id not in session_store:
                session_store[session_id] = ChatMessageHistory()
            return session_store[session_id]
        
        # Wrap with memory
        rag_with_memory = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="question",
            history_messages_key="history",
        )
        
        logger.info("RAG chain built successfully")
        return rag_with_memory
        
    except Exception as e:
        logger.error(f"Failed to build RAG chain: {e}")
        raise


def build_retrieval_chain():
    """Build a simple retrieval chain for getting relevant documents."""
    
    try:
        retriever = load_retriever()
        
        def retrieve_with_metadata(query: str) -> Dict[str, Any]:
            docs = retriever.invoke(query)
            return {
                "documents": docs,
                "products": extract_products_from_docs(docs),
                "query_type": detect_query_type(query),
                "context": format_docs(docs)
            }
        
        return retrieve_with_metadata
        
    except Exception as e:
        logger.error(f"Failed to build retrieval chain: {e}")
        raise


class RAGSystem:
    """Main RAG system class that orchestrates all components."""
    
    def __init__(self):
        self.session_store: Dict[str, ChatMessageHistory] = {}
        self.rag_chain = None
        self.retrieval_chain = None
        self._initialize()
    
    def _initialize(self):
        """Initialize the RAG system components."""
        try:
            self.rag_chain = build_rag_chain(self.session_store)
            self.retrieval_chain = build_retrieval_chain()
            logger.info("RAG system initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize RAG system: {e}")
            raise
    
    def query(self, question: str, session_id: str) -> str:
        """Process a user query and return a response."""
        if not self.rag_chain:
            raise RuntimeError("RAG chain not initialized")
        
        try:
            response = self.rag_chain.invoke(
                question,
                config={"configurable": {"session_id": session_id}}
            )
            return response
        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            raise
    
    def retrieve(self, query: str) -> Dict[str, Any]:
        """Retrieve relevant documents without generating a response."""
        if not self.retrieval_chain:
            raise RuntimeError("Retrieval chain not initialized")
        
        try:
            return self.retrieval_chain(query)
        except Exception as e:
            logger.error(f"Document retrieval failed: {e}")
            raise
    
    def get_session_info(self, session_id: str) -> Dict[str, Any]:
        """Get information about a user session."""
        if session_id in self.session_store:
            history = self.session_store[session_id]
            return {
                "session_id": session_id,
                "message_count": len(history.messages),
                "exists": True
            }
        return {"session_id": session_id, "message_count": 0, "exists": False}
    
    def clear_session(self, session_id: str) -> bool:
        """Clear a user session."""
        if session_id in self.session_store:
            del self.session_store[session_id]
            return True
        return False