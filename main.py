import streamlit as st
import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv
from openai import OpenAI
import chromadb
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict, Any, Optional, Tuple
import json
import psycopg2
from psycopg2.extras import RealDictCursor
import re
import hashlib
from functools import lru_cache
import time
import uuid
from datetime import datetime

# LangChain imports
from langchain_community.document_loaders import CSVLoader, UnstructuredExcelLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

# Load environment variables
load_dotenv()

# Configure OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class SessionManager:
    """Manages user sessions and stores conversation history in Redis"""
    
    def __init__(self):
        """Initialize Redis connection and session management"""
        try:
            import redis
            # Get Redis configuration from environment variables
            redis_host = os.getenv("REDIS_HOST", "localhost")
            redis_port = int(os.getenv("REDIS_PORT", 6379))
            redis_db = int(os.getenv("REDIS_DB", 0))
            redis_password = os.getenv("REDIS_PASSWORD", None)
            
            self.redis_client = redis.Redis(
                host=redis_host,
                port=redis_port,
                db=redis_db,
                password=redis_password,
                decode_responses=True  # Automatically decode responses to strings
            )
            
            # Test connection
            self.redis_client.ping()
            self.redis_available = True
            print("✅ Redis connection established successfully")
            
        except Exception as e:
            print(f"⚠️ Redis connection failed: {str(e)}")
            self.redis_available = False
            self.redis_client = None
    
    def generate_session_id(self) -> str:
        """Generate a unique session ID"""
        return str(uuid.uuid4())
    
    def get_or_create_session_id(self) -> str:
        """Get existing session ID from Streamlit session state or create new one"""
        if 'session_id' not in st.session_state:
            st.session_state.session_id = self.generate_session_id()
            print(f"🆔 Created new session: {st.session_state.session_id}")
        return st.session_state.session_id
    
    def save_conversation_turn(self, session_id: str, user_query: str, assistant_response: str, 
                              execution_time: float = None, metadata: Dict[str, Any] = None):
        """Save a conversation turn (query + response) to Redis"""
        if not self.redis_available:
            print("⚠️ Redis not available, skipping conversation save")
            return
        
        try:
            # Create conversation entry
            conversation_entry = {
                "timestamp": datetime.now().isoformat(),
                "user_query": user_query,
                "assistant_response": assistant_response,
                "execution_time": execution_time,
                "metadata": metadata or {}
            }
            
            # Use Redis list to store conversation history
            # Key format: session:{session_id}:conversation
            session_key = f"session:{session_id}:conversation"
            
            # Add to the end of the list
            self.redis_client.rpush(session_key, json.dumps(conversation_entry))
            
            # Set expiration for session data (24 hours)
            self.redis_client.expire(session_key, 86400)
            
            # Also save session metadata
            session_meta_key = f"session:{session_id}:metadata"
            session_metadata = {
                "created_at": datetime.now().isoformat(),
                "last_activity": datetime.now().isoformat(),
                "total_turns": self.redis_client.llen(session_key)
            }
            self.redis_client.setex(session_meta_key, 86400, json.dumps(session_metadata))
            
            print(f"💾 Saved conversation turn for session {session_id}")
            
        except Exception as e:
            print(f"❌ Error saving conversation to Redis: {str(e)}")
    
    def get_conversation_history(self, session_id: str) -> List[Dict[str, Any]]:
        """Retrieve conversation history for a session"""
        if not self.redis_available:
            print("⚠️ Redis not available, returning empty conversation history")
            return []
        
        try:
            session_key = f"session:{session_id}:conversation"
            conversation_data = self.redis_client.lrange(session_key, 0, -1)
            
            conversation_history = []
            for entry in conversation_data:
                conversation_history.append(json.loads(entry))
            
            return conversation_history
            
        except Exception as e:
            print(f"❌ Error retrieving conversation history: {str(e)}")
            return []
    
    def get_session_metadata(self, session_id: str) -> Dict[str, Any]:
        """Get metadata for a specific session"""
        if not self.redis_available:
            return {}
        
        try:
            session_meta_key = f"session:{session_id}:metadata"
            metadata_json = self.redis_client.get(session_meta_key)
            
            if metadata_json:
                return json.loads(metadata_json)
            return {}
            
        except Exception as e:
            print(f"❌ Error retrieving session metadata: {str(e)}")
            return {}
    
    def get_all_sessions(self) -> List[Dict[str, Any]]:
        """Get list of all active sessions"""
        if not self.redis_available:
            return []
        
        try:
            # Get all session metadata keys
            session_keys = self.redis_client.keys("session:*:metadata")
            sessions = []
            
            for key in session_keys:
                session_id = key.split(":")[1]  # Extract session ID from key
                metadata = self.get_session_metadata(session_id)
                if metadata:
                    sessions.append({
                        "session_id": session_id,
                        **metadata
                    })
            
            return sessions
            
        except Exception as e:
            print(f"❌ Error retrieving all sessions: {str(e)}")
            return []
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a session and all its data"""
        if not self.redis_available:
            return False
        
        try:
            # Delete conversation history
            conversation_key = f"session:{session_id}:conversation"
            self.redis_client.delete(conversation_key)
            
            # Delete session metadata
            metadata_key = f"session:{session_id}:metadata"
            self.redis_client.delete(metadata_key)
            
            print(f"🗑️ Deleted session {session_id}")
            return True
            
        except Exception as e:
            print(f"❌ Error deleting session: {str(e)}")
            return False

class QueryOptimizer:
    """Optimizes RAG queries for better performance and accuracy"""
    
    def __init__(self):
        self.query_cache = {}
        self.context_cache = {}
        self.search_strategies = {
            'exact_match': self._exact_match_search,
            'semantic_search': self._semantic_search,
            'hybrid_search': self._hybrid_search,
            'aggregation_search': self._aggregation_search
        }
    
    def preprocess_query(self, query: str) -> Dict[str, Any]:
        """Analyze and preprocess the query to determine optimal search strategy"""
        query_lower = query.lower()
        
        # Detect query intent
        intent = self._detect_query_intent(query_lower)
        
        # Extract key terms for search optimization
        key_terms = self._extract_key_terms(query)
        
        # Determine optimal search strategy
        strategy = self._determine_search_strategy(intent, key_terms)
        
        # Estimate query complexity
        complexity = self._estimate_complexity(query, intent)
        
        return {
            'original_query': query,
            'processed_query': self._optimize_query_text(query, key_terms),
            'intent': intent,
            'key_terms': key_terms,
            'strategy': strategy,
            'complexity': complexity,
            'requires_aggregation': intent.get('aggregation', False),
            'requires_cross_dataset': intent.get('cross_dataset', False)
        }
    
    def _detect_query_intent(self, query: str) -> Dict[str, Any]:
        """Detect the intent of the query"""
        intent = {
            'aggregation': False,
            'cross_dataset': False,
            'visualization': False,
            'comparison': False,
            'filtering': False,
            'ranking': False
        }
        
        # Aggregation detection
        agg_keywords = ['sum', 'total', 'average', 'mean', 'count', 'aggregate', 'group by', 'total spend']
        intent['aggregation'] = any(keyword in query for keyword in agg_keywords)
        
        # Cross-dataset detection
        cross_keywords = ['across', 'compare', 'between', 'all datasets', 'multiple', 'sheets']
        intent['cross_dataset'] = any(keyword in query for keyword in cross_keywords)
        
        # Visualization detection
        viz_keywords = ['chart', 'graph', 'plot', 'visualize', 'show me', 'display']
        intent['visualization'] = any(keyword in query for keyword in viz_keywords)
        
        # Comparison detection
        comp_keywords = ['vs', 'versus', 'compare', 'difference', 'better', 'worse']
        intent['comparison'] = any(keyword in query for keyword in comp_keywords)
        
        # Filtering detection
        filter_keywords = ['where', 'filter', 'only', 'show', 'find', 'search']
        intent['filtering'] = any(keyword in query for keyword in filter_keywords)
        
        return intent
    
    def _extract_key_terms(self, query: str) -> List[str]:
        """Extract key terms for search optimization"""
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'}
        
        # Extract words and filter
        words = re.findall(r'\b\w+\b', query.lower())
        key_terms = [word for word in words if word not in stop_words and len(word) > 2]
        
        return key_terms
    
    def _determine_search_strategy(self, intent: Dict[str, Any], key_terms: List[str]) -> str:
        """Determine the optimal search strategy based on intent and terms"""
        if intent['aggregation']:
            return 'aggregation_search'
        elif intent['cross_dataset']:
            return 'hybrid_search'
        elif intent['filtering'] and len(key_terms) > 2:
            return 'exact_match'
        else:
            return 'semantic_search'
    
    def _estimate_complexity(self, query: str, intent: Dict[str, Any]) -> str:
        """Estimate query complexity for adaptive search parameters"""
        word_count = len(query.split())
        
        if intent['aggregation'] or intent['cross_dataset']:
            return 'high'
        elif word_count > 10 or intent['comparison']:
            return 'medium'
        else:
            return 'low'
    
    def _optimize_query_text(self, query: str, key_terms: List[str]) -> str:
        """Optimize query text for better vector search"""
        # Enhance query with key terms if they're not prominent
        enhanced_terms = []
        
        # Add common business terms that might be relevant
        business_terms = ['spend', 'campaign', 'platform', 'channel', 'performance', 'roi', 'conversion']
        for term in business_terms:
            if term in query.lower() and term not in enhanced_terms:
                enhanced_terms.append(term)
        
        # Combine original query with enhanced terms
        if enhanced_terms:
            enhanced_query = f"{query} {' '.join(enhanced_terms)}"
        else:
            enhanced_query = query
            
        return enhanced_query

    def _exact_match_search(self, vectorstore, query: str, k: int = 5) -> List:
        """Perform exact match search for specific terms"""
        # Use a more focused search with higher similarity threshold
        docs = vectorstore.similarity_search_with_score(query, k=k*2)
        # Filter by similarity score (higher is better)
        filtered_docs = [doc for doc, score in docs if score > 0.7]
        return filtered_docs[:k]
    
    def _semantic_search(self, vectorstore, query: str, k: int = 10) -> List:
        """Perform semantic search with optimized parameters"""
        return vectorstore.similarity_search(query, k=k)
    
    def _hybrid_search(self, vectorstore, query: str, k: int = 15) -> List:
        """Perform hybrid search combining multiple strategies"""
        # Get semantic results
        semantic_docs = vectorstore.similarity_search(query, k=k//2)
        
        # Get MMR results for diversity
        mmr_docs = vectorstore.max_marginal_relevance_search(query, k=k//2, fetch_k=k)
        
        # Combine and deduplicate
        all_docs = semantic_docs + mmr_docs
        seen = set()
        unique_docs = []
        for doc in all_docs:
            doc_hash = hash(doc.page_content)
            if doc_hash not in seen:
                seen.add(doc_hash)
                unique_docs.append(doc)
        
        return unique_docs[:k]
    
    def _aggregation_search(self, vectorstore, query: str, k: int = 20) -> List:
        """Perform search optimized for aggregation queries"""
        # For aggregations, we want more diverse results
        return vectorstore.max_marginal_relevance_search(query, k=k, fetch_k=k*2)
    
    def get_adaptive_k(self, complexity: str, intent: Dict[str, Any]) -> int:
        """Get adaptive k value based on query complexity and intent"""
        base_k = {
            'low': 5,
            'medium': 10,
            'high': 15
        }
        
        k = base_k.get(complexity, 10)
        
        # Adjust based on intent
        if intent.get('aggregation'):
            k *= 2  # Need more data for aggregations
        elif intent.get('cross_dataset'):
            k *= 1.5  # Need data from multiple sources
        elif intent.get('filtering'):
            k = min(k, 8)  # More focused results for filtering
        
        # Ensure k is an integer
        return int(k)
    
    def cache_query_result(self, query_hash: str, result: Any, ttl: int = 300):
        """Cache query result with TTL"""
        self.query_cache[query_hash] = {
            'result': result,
            'timestamp': time.time(),
            'ttl': ttl
        }
    
    def get_cached_result(self, query_hash: str) -> Optional[Any]:
        """Get cached result if still valid"""
        if query_hash in self.query_cache:
            cache_entry = self.query_cache[query_hash]
            if time.time() - cache_entry['timestamp'] < cache_entry['ttl']:
                return cache_entry['result']
            else:
                del self.query_cache[query_hash]
        return None

class CSVChatApp:
    def __init__(self):
        self.dfs = {}  # Dictionary to store multiple dataframes
        self.text_files = set()  # Set to track uploaded text files
        self.text_contents = {}  # Dictionary to store original text content
        self.vectorstore = None  # LangChain vectorstore
        self.embeddings = None  # OpenAI embeddings
        self.text_splitter = None  # Text splitter
        self.collection = None  # Direct ChromaDB collection
        self.chat_history = []
        self.processed_files = set()  # Set to track processed file names
        self.query_optimizer = QueryOptimizer()  # Query optimization system
        self.dataset_summaries = {}  # Cache for dataset summaries
        self.query_cache = {}  # Cache for query results
        self.performance_metrics = {}  # Track query performance
        self.query_history = []  # Track query patterns for optimization
        self.session_manager = SessionManager()  # Session management with Redis
        self.agent_insights = {}  # Store agent insights per session

    def setup_langchain(self):
        """Initialize LangChain components for vector storage"""
        try:
            # Ensure chroma_db directory exists
            import os
            os.makedirs("./chroma_db", exist_ok=True)
            
            if self.embeddings is None:
                try:
                    self.embeddings = OpenAIEmbeddings()
                except Exception as embed_error:
                    st.error(f"Error initializing embeddings: {str(embed_error)}")
                    st.error("Please check your OpenAI API key in the .env file")
                    return
            
            if self.text_splitter is None:
                self.text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=500,
                    chunk_overlap=50,
                    length_function=len,
                    separators=["\n\n", "\n", " ", ""],
                    keep_separator=True
                )
            
            # Initialize vectorstore if not exists
            if self.vectorstore is None:
                try:
                    self.vectorstore = Chroma(
                        persist_directory="./chroma_db",
                        embedding_function=self.embeddings,
                        collection_name="csv_data"
                    )
                except Exception as db_error:
                    # Handle read-only database errors gracefully
                    if "readonly" in str(db_error).lower() or "code: 1032" in str(db_error):
                        st.warning("Database is read-only. Using existing data.")
                        # Try to use existing database without clearing
                        try:
                            self.vectorstore = Chroma(
                                persist_directory="./chroma_db",
                                embedding_function=self.embeddings,
                                collection_name="csv_data"
                            )
                        except:
                            st.error("Could not access database. Please restart the app.")
                            self.vectorstore = None
                    elif "Database error" in str(db_error) or "no such table" in str(db_error):
                        st.warning("Database corrupted, clearing and recreating...")
                        import shutil
                        if os.path.exists("./chroma_db"):
                            try:
                                shutil.rmtree("./chroma_db")
                            except:
                                pass
                        os.makedirs("./chroma_db", exist_ok=True)
                        
                        # Try again with fresh database
                        self.vectorstore = Chroma(
                            persist_directory="./chroma_db",
                            embedding_function=self.embeddings,
                            collection_name="csv_data"
                        )
                        st.success("Database recreated successfully!")
                    else:
                        # Re-raise other errors
                        raise db_error
                
        except Exception as e:
            st.error(f"Error setting up LangChain: {str(e)}")
            self.embeddings = None
            self.text_splitter = None
            self.vectorstore = None

    def setup_chroma(self):
        """Initialize ChromaDB collection for direct access"""
        try:
            import chromadb
            from chromadb.config import Settings
            
            # Initialize ChromaDB client
            client = chromadb.PersistentClient(
                path="./chroma_db",
                settings=Settings(
                    anonymized_telemetry=False
                )
            )
            
            # Get or create collection
            self.collection = client.get_or_create_collection(
                name="csv_data",
                metadata={"hnsw:space": "cosine"}
            )
            
            # Don't clear existing data - let it accumulate
            # This prevents read-only database errors
            
        except Exception as e:
            st.error(f"Error setting up ChromaDB: {str(e)}")
            self.collection = None

    def clear_vectorstore(self):
        """Clear the LangChain vectorstore by recreating it"""
        try:
            # Reset vectorstore reference
            self.vectorstore = None
            
            # Try to clean up any existing chroma directories
            import shutil
            import os
            
            for item in os.listdir("."):
                if item.startswith("chroma_db"):
                    try:
                        if os.path.isdir(item):
                            shutil.rmtree(item)
                        else:
                            os.remove(item)
                    except Exception as cleanup_error:
                        # Don't show warning for read-only errors, just continue
                        pass
            
            # Recreate the vectorstore
            self.setup_langchain()
            
        except Exception as e:
            # If we can't clear the vectorstore, just reset the reference
            # This allows the app to continue working
            self.vectorstore = None
            st.warning("Could not clear vector database, but app will continue to work.")

    def reset_session(self):
        """Reset current session while keeping Redis records"""
        try:
            # Clear current session data
            self.dfs.clear()
            self.text_files.clear()
            self.text_contents.clear()
            self.processed_files.clear()
            
            # Clear current session agent insights
            current_session_id = self.session_manager.get_or_create_session_id()
            if current_session_id in self.agent_insights:
                del self.agent_insights[current_session_id]
            
            # Clear vectorstore
            self.clear_vectorstore()
            
            # Generate new session ID
            new_session_id = self.session_manager.generate_session_id()
            st.session_state.session_id = new_session_id
            
            st.success(f"Session reset! New session ID: {new_session_id[:8]}...")
            st.info("Previous session data is preserved in Redis and can be viewed in the session history.")
            
        except Exception as e:
            st.error(f"Error resetting session: {str(e)}")

    def get_session_history_summary(self):
        """Get a summary of all sessions from Redis"""
        try:
            all_sessions = self.session_manager.get_all_sessions()
            
            if not all_sessions:
                return "No previous sessions found."
            
            summary = []
            summary.append(f"📚 **Session History**: {len(all_sessions)} previous sessions")
            
            for session in all_sessions[:5]:  # Show last 5 sessions
                session_id = session['session_id']
                timestamp = session.get('last_activity', 'Unknown')
                interaction_count = session.get('total_turns', 0)
                
                summary.append(f"🆔 **Session {session_id[:8]}...**: {interaction_count} interactions ({timestamp})")
            
            if len(all_sessions) > 5:
                summary.append(f"... and {len(all_sessions) - 5} more sessions")
            
            return "\n".join(summary)
            
        except Exception as e:
            return f"Error retrieving session history: {str(e)}"

    def llm_analyze_data(self, filename: str, df: pd.DataFrame):
        """LLM-powered agent that intelligently analyzes uploaded data"""
        try:
            session_id = self.session_manager.get_or_create_session_id()
            
            # Prepare data summary for LLM
            data_summary = f"Dataset: {filename}, Shape: {df.shape}, Columns: {list(df.columns)}"
            
            # Get sample data (first 5 rows, formatted nicely)
            sample_data = df.head(5).to_string(index=False, max_cols=10, max_colwidth=20)
            
            # Get basic statistics for context
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
            
            stats_summary = ""
            if numeric_cols:
                stats_summary += f"\nNumeric columns: {numeric_cols[:3]}{'...' if len(numeric_cols) > 3 else ''}"
                if len(numeric_cols) > 0:
                    col = numeric_cols[0]
                    stats_summary += f"\nSample stats for {col}: min={df[col].min():.2f}, max={df[col].max():.2f}, mean={df[col].mean():.2f}"
            
            if categorical_cols:
                stats_summary += f"\nCategorical columns: {categorical_cols[:3]}{'...' if len(categorical_cols) > 3 else ''}"
                if len(categorical_cols) > 0:
                    col = categorical_cols[0]
                    unique_count = df[col].nunique()
                    stats_summary += f"\nUnique values in {col}: {unique_count}"
            
            prompt = f"""
            Analyze this dataset and provide intelligent insights. Focus on patterns, anomalies, and interesting findings that would be valuable to a data analyst.

            Dataset Information:
            {data_summary}
            {stats_summary}

            Sample Data (first 5 rows):
            {sample_data}

            Please provide 3-5 intelligent insights about:
            1. Data patterns or trends you notice
            2. Potential anomalies or outliers
            3. Interesting relationships between variables
            4. Data quality observations
            5. Recommendations for further analysis

            Format your response as a clear, professional analysis with bullet points.
            """
            
            # Generate LLM response using OpenAI directly
            try:
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a helpful data analysis assistant. Provide clear, professional insights."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=1000,
                    temperature=0.3
                )
                response = response.choices[0].message.content
            except Exception as e:
                response = f"Error generating LLM response: {str(e)}"
            
            # Store insights in session
            if session_id not in self.agent_insights:
                self.agent_insights[session_id] = {}
            if filename not in self.agent_insights[session_id]:
                self.agent_insights[session_id][filename] = {}
            self.agent_insights[session_id][filename]['analysis'] = response
            
            # Save to Redis
            self.session_manager.save_conversation_turn(
                session_id=session_id,
                user_query=f"LLM analysis of {filename}",
                assistant_response=response,
                metadata={"type": "llm_analysis", "filename": filename}
            )
            
            return response
            
        except Exception as e:
            st.error(f"Error in LLM analysis: {str(e)}")
            return f"Error analyzing {filename}: {str(e)}"

    def llm_quality_check(self, filename: str, df: pd.DataFrame):
        """LLM-powered agent that intelligently assesses data quality"""
        try:
            session_id = self.session_manager.get_or_create_session_id()
            
            # Prepare quality metrics for LLM
            missing_counts = df.isnull().sum()
            total_missing = missing_counts.sum()
            missing_cols = missing_counts[missing_counts > 0]
            
            duplicate_count = df.duplicated().sum()
            
            # Calculate outlier information for numeric columns
            numeric_cols = df.select_dtypes(include=['number']).columns
            outlier_info = []
            for col in numeric_cols[:3]:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = df[(df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)]
                if len(outliers) > 0:
                    outlier_info.append(f"{col}: {len(outliers)} outliers ({len(outliers)/len(df)*100:.1f}%)")
            
            # Data type information
            data_types = df.dtypes.to_dict()
            categorical_cols = df.select_dtypes(include=['object']).columns
            
            quality_summary = f"""
            Dataset: {filename}
            Shape: {df.shape}
            
            Quality Metrics:
            - Missing values: {total_missing} total across {len(missing_cols)} columns
            - Duplicate rows: {duplicate_count} ({duplicate_count/len(df)*100:.1f}%)
            - Outliers detected: {len(outlier_info)} columns with outliers
            - Data types: {dict(data_types)}
            """
            
            if missing_cols.any():
                quality_summary += f"\nMissing value details:\n{missing_cols.to_string()}"
            
            if outlier_info:
                quality_summary += f"\nOutlier details:\n{chr(10).join(outlier_info)}"
            
            prompt = f"""
            Assess the quality of this dataset and provide intelligent insights about data quality issues.

            {quality_summary}

            Please provide a concise quality assessment addressing each of the below in one sentence:
            1. Critical quality issues that need immediate attention
            2. Potential data integrity problems
            3. Recommendations for data cleaning
            4. Impact assessment of quality issues on analysis
            5. Suggestions for handling specific problems

            Focus on actionable insights and prioritize issues by severity.
            Format your response as a professional quality assessment report.
            """
            
            # Generate LLM response using OpenAI directly
            try:
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a data quality assessment expert. Provide clear, actionable quality insights."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=1000,
                    temperature=0.3
                )
                response = response.choices[0].message.content
            except Exception as e:
                response = f"Error generating LLM response: {str(e)}"
            
            # Store quality report in session
            if session_id not in self.agent_insights:
                self.agent_insights[session_id] = {}
            if filename not in self.agent_insights[session_id]:
                self.agent_insights[session_id][filename] = {}
            self.agent_insights[session_id][filename]['quality'] = response
            
            # Save to Redis
            self.session_manager.save_conversation_turn(
                session_id=session_id,
                user_query=f"LLM quality assessment of {filename}",
                assistant_response=response,
                metadata={"type": "llm_quality_check", "filename": filename}
            )
            
            return response
            
        except Exception as e:
            st.error(f"Error in LLM quality check: {str(e)}")
            return f"Error assessing quality of {filename}: {str(e)}"

    def llm_suggest_queries(self, filename: str, df: pd.DataFrame):
        """LLM-powered agent that generates intelligent query suggestions"""
        try:
            session_id = self.session_manager.get_or_create_session_id()
            
            # Prepare data context for LLM
            data_summary = f"Dataset: {filename}, Shape: {df.shape}, Columns: {list(df.columns)}"
            
            # Get sample data
            sample_data = df.head(3).to_string(index=False, max_cols=10, max_colwidth=20)
            
            # Get column information
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
            datetime_cols = df.select_dtypes(include=['datetime']).columns.tolist()
            
            column_info = f"""
            Column Types:
            - Numeric: {numeric_cols[:5]}{'...' if len(numeric_cols) > 5 else ''}
            - Categorical: {categorical_cols[:5]}{'...' if len(categorical_cols) > 5 else ''}
            - DateTime: {datetime_cols[:3]}{'...' if len(datetime_cols) > 3 else ''}
            """
            
            # Add some basic statistics for context
            if numeric_cols:
                col = numeric_cols[0]
                column_info += f"\nSample numeric stats for {col}: min={df[col].min():.2f}, max={df[col].max():.2f}, mean={df[col].mean():.2f}"
            
            if categorical_cols:
                col = categorical_cols[0]
                unique_count = df[col].nunique()
                column_info += f"\nSample categorical info for {col}: {unique_count} unique values"
            
            prompt = f"""
            Based on this dataset, generate 3 intelligent and specific questions that a data analyst would want to ask. 

            Keep the prompts short and concise

            Dataset Information:
            {data_summary}
            {column_info}

            Sample Data:
            {sample_data}

            Generate questions that would provide valuable insights such as:
            1. Statistical analysis questions (correlations, distributions, trends)
            2. Business intelligence questions (patterns, anomalies, performance)
            3. Data exploration questions (relationships, groupings, comparisons)
            4. Predictive analysis questions (trends, forecasting, patterns)
            5. Data quality questions (validation, consistency, completeness)

            Make the questions specific to the actual columns and data types in this dataset.
            Format each question clearly and make them actionable for analysis.
            """
            
            # Generate LLM response using OpenAI directly
            try:
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a data analysis expert. Generate specific, actionable questions for data exploration."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=1000,
                    temperature=0.3
                )
                response = response.choices[0].message.content
            except Exception as e:
                response = f"Error generating LLM response: {str(e)}"
            
            # Store suggestions in session
            if session_id not in self.agent_insights:
                self.agent_insights[session_id] = {}
            if filename not in self.agent_insights[session_id]:
                self.agent_insights[session_id][filename] = {}
            self.agent_insights[session_id][filename]['suggestions'] = response
            
            # Save to Redis
            self.session_manager.save_conversation_turn(
                session_id=session_id,
                user_query=f"LLM query suggestions for {filename}",
                assistant_response=response,
                metadata={"type": "llm_query_suggestions", "filename": filename}
            )
            
            return response
            
        except Exception as e:
            st.error(f"Error generating LLM suggestions: {str(e)}")
            return f"Error generating suggestions for {filename}: {str(e)}"

    def run_automatic_agents(self, filename: str, df: pd.DataFrame):
        """Run all LLM-powered agents for newly uploaded data"""
        # Check if OpenAI API key is available
        if not os.getenv("OPENAI_API_KEY"):
            st.error("❌ OpenAI API key not found. LLM agents cannot run.")
            st.info("Please add your OpenAI API key to the .env file to enable LLM analysis.")
            return
        
        with st.spinner(f"🤖 LLM Agents analyzing {filename}..."):
            # Run all LLM agents
            analysis_insights = self.llm_analyze_data(filename, df)
            quality_report = self.llm_quality_check(filename, df)
            query_suggestions = self.llm_suggest_queries(filename, df)
            
            # Display results automatically
            st.success(f"✅ **LLM Analysis Complete for {filename}**")
            
            # Show insights in expandable sections
            with st.expander(f"🧠 LLM Data Analysis - {filename}", expanded=True):
                st.markdown(analysis_insights)
            
            with st.expander(f"🔍 LLM Quality Assessment - {filename}", expanded=True):
                st.markdown(quality_report)
            
            with st.expander(f"💡 LLM Query Suggestions - {filename}", expanded=True):
                st.markdown(query_suggestions)

    def get_all_vector_data(self) -> str:
        """Retrieve all data from the vector database for analysis"""
        if self.vectorstore is None:
            return "No vector database available. Please upload some data first."
        
        try:
            # Get all documents from the vectorstore
            all_docs = self.vectorstore.similarity_search("", k=1000)  # Large k to get most/all documents
            
            if not all_docs:
                return "No data found in vector database."
            
            # Combine all document content
            combined_content = "\n\n".join([doc.page_content for doc in all_docs])
            
            # Get metadata summary
            metadata_summary = {}
            for doc in all_docs:
                if hasattr(doc, 'metadata') and doc.metadata:
                    filename = doc.metadata.get('filename', 'unknown')
                    if filename not in metadata_summary:
                        metadata_summary[filename] = 0
                    metadata_summary[filename] += 1
            
            summary = f"Total documents in vector database: {len(all_docs)}\n"
            summary += f"Files represented: {list(metadata_summary.keys())}\n"
            summary += f"Document distribution: {metadata_summary}\n\n"
            summary += "Combined content:\n" + combined_content[:5000] + "..." if len(combined_content) > 5000 else combined_content
            
            return summary
            
        except Exception as e:
            return f"Error retrieving vector data: {str(e)}"

    def llm_analyze_all_data(self):
        """LLM-powered agent that analyzes all data in the vector database"""
        try:
            session_id = self.session_manager.get_or_create_session_id()
            
            # Get all data from vector database
            all_data = self.get_all_vector_data()
            
            if "No data found" in all_data or "No vector database" in all_data:
                return "No data available for analysis. Please upload some files first."
            
            prompt = f"""
            Analyze all the data in this session's vector database and provide three short concise insights. 
            Focus on patterns, relationships, and insights across all datasets.

            All Session Data:
            {all_data}

            Please provide three concise intelligent insights from the following list:
            1. Cross-dataset patterns and relationships
            2. Key themes and trends across all data
            3. Potential business insights or opportunities
            4. Recommendations for further analysis
            5. Data integration opportunities
            6. Anomalies or interesting findings

            Format your response as a clear, professional analysis with three bullet points.
            Focus on insights that span across multiple datasets rather than individual file analysis.
            """
            
            # Generate LLM response using OpenAI directly
            try:
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a comprehensive data analysis expert. Provide insights across multiple datasets."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=1500,
                    temperature=0.3
                )
                response = response.choices[0].message.content
            except Exception as e:
                response = f"Error generating LLM response: {str(e)}"
            
            # Save to Redis
            self.session_manager.save_conversation_turn(
                session_id=session_id,
                user_query="LLM analysis of all session data",
                assistant_response=response,
                metadata={"type": "llm_analysis_all_data"}
            )
            
            return response
            
        except Exception as e:
            return f"Error in LLM analysis: {str(e)}"

    def llm_quality_check_all_data(self):
        """LLM-powered agent that assesses quality across all data in the vector database"""
        try:
            session_id = self.session_manager.get_or_create_session_id()
            
            # Get all data from vector database
            all_data = self.get_all_vector_data()
            
            if "No data found" in all_data or "No vector database" in all_data:
                return "No data available for quality assessment. Please upload some files first."
            
            prompt = f"""
            Assess the overall quality of all data in this session's vector database. 
            Focus on cross-dataset quality issues and consistency problems.

            All Session Data:
            {all_data}

            Please provide a comprehensive quality assessment including:
            1. Overall data quality score and summary
            2. Cross-dataset consistency issues
            3. Quality problems that affect analysis
            4. Recommendations for data cleaning and standardization
            5. Impact assessment on business intelligence

            Focus on quality issues that span across multiple datasets and affect overall analysis quality.
            Format your response as a professional quality assessment report.
            """
            
            # Generate LLM response using OpenAI directly
            try:
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a data quality assessment expert. Focus on cross-dataset quality issues."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=1500,
                    temperature=0.3
                )
                response = response.choices[0].message.content
            except Exception as e:
                response = f"Error generating LLM response: {str(e)}"
            
            # Save to Redis
            self.session_manager.save_conversation_turn(
                session_id=session_id,
                user_query="LLM quality assessment of all session data",
                assistant_response=response,
                metadata={"type": "llm_quality_check_all_data"}
            )
            
            return response
            
        except Exception as e:
            return f"Error in LLM quality check: {str(e)}"

    def llm_suggest_queries_all_data(self):
        """LLM-powered agent that generates query suggestions for all data in the vector database"""
        try:
            session_id = self.session_manager.get_or_create_session_id()
            
            # Get all data from vector database
            all_data = self.get_all_vector_data()
            
            if "No data found" in all_data or "No vector database" in all_data:
                return "No data available for query suggestions. Please upload some files first."
            
            prompt = f"""
            Based on all the data in this session's vector database, generate 10-15 intelligent and specific questions 
            that would provide valuable cross-dataset insights.

            All Session Data:
            {all_data}

            Generate questions that would provide valuable insights such as:
            1. Cross-dataset correlation and relationship questions
            2. Data integration and consistency questions
            3. Business intelligence questions spanning multiple datasets
            4. Data quality and validation questions across datasets
            5. Predictive analysis questions using combined data
            6. Comparative analysis questions between datasets
            7. Trend analysis questions across time or categories
            8. Anomaly detection questions across datasets

            Make the questions specific to the actual data available and focus on insights that require 
            analysis across multiple datasets rather than single dataset questions.
            Format each question clearly and make them actionable for analysis.
            """
            
            # Generate LLM response using OpenAI directly
            try:
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a data analysis expert. Generate cross-dataset query suggestions."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=1500,
                    temperature=0.3
                )
                response = response.choices[0].message.content
            except Exception as e:
                response = f"Error generating LLM response: {str(e)}"
            
            # Save to Redis
            self.session_manager.save_conversation_turn(
                session_id=session_id,
                user_query="LLM query suggestions for all session data",
                assistant_response=response,
                metadata={"type": "llm_query_suggestions_all_data"}
            )
            
            return response
            
        except Exception as e:
            return f"Error generating LLM suggestions: {str(e)}"

    def process_csv(self, uploaded_file, filename: str) -> pd.DataFrame:
        """Process uploaded CSV file using LangChain"""
        try:
            # Read CSV with pandas for dataframe storage
            df = pd.read_csv(uploaded_file)
            self.dfs[filename] = df
            self.processed_files.add(filename)
            
            # Setup LangChain components
            self.setup_langchain()
            
            # Process with LangChain CSVLoader
            langchain_success = self.process_with_langchain(uploaded_file, filename, "csv")
            if not langchain_success:
                st.error(f"Failed to process {filename} with LangChain")
                return None
            
            # Note: LLM agents are now available as manual buttons in the main interface
            
            return df
        except Exception as e:
            st.error(f"Error processing CSV: {str(e)}")
            return None

    def process_excel(self, uploaded_file, filename: str) -> pd.DataFrame:
        """Process uploaded Excel file - processes each sheet separately"""
        try:
            # Read all sheets from the Excel file
            excel_file = pd.ExcelFile(uploaded_file)
            sheet_names = excel_file.sheet_names
            
            if len(sheet_names) == 1:
                # Single sheet - read directly
                df = pd.read_excel(uploaded_file)
                self.dfs[filename] = df
                self.processed_files.add(filename)
                self.setup_langchain()
                self.process_excel_sheet_with_langchain(df, filename)
                
                # Note: LLM agents are now available as manual buttons in the main interface
                
                return df
            else:
                # Multiple sheets - process each sheet separately
                st.info(f"Excel file '{filename}' contains {len(sheet_names)} sheets: {sheet_names}")
                
                # Setup LangChain once for all sheets
                self.setup_langchain()
                
                # Process each sheet separately
                processed_sheets = []
                total_rows = 0
                
                for sheet_name in sheet_names:
                    try:
                        df = pd.read_excel(uploaded_file, sheet_name=sheet_name)
                        if len(df) > 0:  # Only process non-empty sheets
                            # Create unique filename for each sheet
                            sheet_filename = f"{filename}_{sheet_name}"
                            self.dfs[sheet_filename] = df
                            self.processed_files.add(sheet_filename)  # Mark as processed
                            
                            # Process this sheet with LangChain
                            self.process_excel_sheet_with_langchain(df, sheet_filename)
                            
                            # Note: LLM agents are now available as manual buttons in the main interface
                            
                            processed_sheets.append(sheet_name)
                            total_rows += len(df)
                            st.success(f"✓ Processed sheet: {sheet_name} ({len(df)} rows)")
                        else:
                            st.warning(f"⚠ Skipped empty sheet: {sheet_name}")
                    except Exception as sheet_error:
                        st.warning(f"⚠ Could not process sheet '{sheet_name}': {str(sheet_error)}")
                        continue
                
                if processed_sheets:
                    st.success(f"✅ Processed {len(processed_sheets)} sheets separately: {total_rows} total rows")
                    
                    # Return a summary dataframe for display purposes
                    summary_data = {
                        'Sheet_Name': processed_sheets,
                        'Rows': [len(self.dfs[f"{filename}_{sheet}"]) for sheet in processed_sheets],
                        'Status': ['Processed'] * len(processed_sheets)
                    }
                    summary_df = pd.DataFrame(summary_data)
                    return summary_df
                else:
                    st.error("No sheets could be processed successfully")
                    return None
        except Exception as e:
            st.error(f"Error processing Excel file: {str(e)}")
            return None

    def process_with_langchain(self, uploaded_file, filename: str, file_type: str):
        """Process files using LangChain loaders"""
        try:
            if file_type == "csv":
                # Save uploaded file temporarily
                temp_path = f"temp_{filename}"
                with open(temp_path, 'wb') as f:
                    f.write(uploaded_file.read())
                
                # Load with LangChain CSVLoader
                loader = CSVLoader(temp_path)
                documents = loader.load()
                
                # Add metadata to all documents (don't filter)
                for doc in documents:
                    doc.metadata.update({
                        "filename": filename,
                        "file_type": "csv",
                        "content_type": "data_row"
                    })
                valid_documents = documents
                
                if not valid_documents:
                    st.warning(f"No valid content found in {filename} to process")
                    st.info("This might be due to empty rows or formatting issues. Check your CSV file.")
                    return False
                
                # Ensure text_splitter is initialized
                if self.text_splitter is None:
                    st.error("Text splitter not initialized. Please check your OpenAI API key.")
                    return
                
                # Split documents
                chunks = self.text_splitter.split_documents(valid_documents)
                
                # Filter out empty chunks
                valid_chunks = []
                for chunk in chunks:
                    if chunk.page_content and chunk.page_content.strip():
                        valid_chunks.append(chunk)
                
                if not valid_chunks:
                    st.warning(f"No valid content found in {filename} after processing")
                    st.info("This might be due to very small content that gets filtered out during chunking.")
                    return False
                
                # Debug information
                st.info(f"Original documents: {len(valid_documents)}, Chunks after splitting: {len(chunks)}, Valid chunks: {len(valid_chunks)}")
                
                # Verify embeddings are properly initialized
                if not self.embeddings:
                    st.error("Embeddings not properly initialized. Please check your OpenAI API key.")
                    return
                
                # Add to vectorstore
                try:
                    self.vectorstore.add_documents(valid_chunks)
                except Exception as embedding_error:
                    st.error(f"Error adding documents to vectorstore: {str(embedding_error)}")
                    st.error("This might be due to empty embeddings. Check if your OpenAI API key is valid and the content is meaningful.")
                    return False
                
                # Clean up temp file
                os.remove(temp_path)
                
                st.success(f"Processed {len(valid_chunks)} chunks from {filename} with LangChain")
                return True
                
        except Exception as e:
            st.error(f"Error processing {filename} with LangChain: {str(e)}")
            return False

    def process_excel_sheet_with_langchain(self, df: pd.DataFrame, filename: str):
        """Process Excel sheet data with LangChain"""
        try:
            # Convert dataframe to documents
            documents = []
            for idx, row in df.iterrows():
                # Create document from row
                row_text = " ".join([f"{col}: {val}" for col, val in row.items()])
                
                # Only add documents with meaningful content - more lenient
                if row_text and row_text.strip():
                    doc = type('Document', (), {
                        'page_content': row_text,
                        'metadata': {
                            'filename': filename,
                            'row_index': idx,
                            'file_type': 'excel',
                            'content_type': 'data_row'
                        }
                    })()
                    documents.append(doc)
            
            # Check if we have any documents to process
            if not documents:
                st.warning(f"No valid content found in {filename} to process")
                return
            
            # Ensure text_splitter is initialized
            if self.text_splitter is None:
                st.error("Text splitter not initialized. Please check your OpenAI API key.")
                return
            
            # Ensure text_splitter is initialized
            if self.text_splitter is None:
                st.error("Text splitter not initialized. Please check your OpenAI API key.")
                return
            
            # Split documents
            chunks = self.text_splitter.split_documents(documents)
            
            # Filter out empty chunks
            valid_chunks = []
            for chunk in chunks:
                if chunk.page_content and chunk.page_content.strip():
                    valid_chunks.append(chunk)
            
            if not valid_chunks:
                st.warning(f"No valid content found in {filename} after processing")
                return
            
            # Debug information
            st.info(f"Original documents: {len(documents)}, Chunks after splitting: {len(chunks)}, Valid chunks: {len(valid_chunks)}")
            
            # Verify embeddings are properly initialized
            if not self.embeddings:
                st.error("Embeddings not properly initialized. Please check your OpenAI API key.")
                return
            
            # Add to vectorstore
            try:
                self.vectorstore.add_documents(valid_chunks)
            except Exception as embedding_error:
                st.error(f"Error adding documents to vectorstore: {str(embedding_error)}")
                st.error("This might be due to empty embeddings. Check if your OpenAI API key is valid and the content is meaningful.")
                return
            
            st.success(f"Processed {len(valid_chunks)} chunks from {filename} with LangChain")
            
        except Exception as e:
            st.error(f"Error processing Excel sheet {filename} with LangChain: {str(e)}")

    def process_text_with_langchain(self, text_content: str, filename: str):
        """Process text content with LangChain"""
        try:
            # Validate text content
            if not text_content or not text_content.strip() or len(text_content.strip()) < 10:
                st.warning(f"No valid content found in {filename} to process")
                return
            
            # Create document from text content
            doc = type('Document', (), {
                'page_content': text_content,
                'metadata': {
                    'filename': filename,
                    'file_type': 'text',
                    'content_type': 'text'
                }
            })()
            
            # Ensure text_splitter is initialized
            if self.text_splitter is None:
                st.error("Text splitter not initialized. Please check your OpenAI API key.")
                return
            
            # Split document
            chunks = self.text_splitter.split_documents([doc])
            
            # Add metadata to chunks
            for chunk in chunks:
                chunk.metadata.update({
                    'filename': filename,
                    'file_type': 'text',
                    'content_type': 'text_chunk'
                })
            
            # Filter out empty chunks
            valid_chunks = []
            for chunk in chunks:
                if chunk.page_content and chunk.page_content.strip():
                    valid_chunks.append(chunk)
            
            if not valid_chunks:
                st.warning(f"No valid content found in {filename} after processing")
                return
            
            # Debug information
            st.info(f"Original document length: {len(text_content)}, Chunks after splitting: {len(chunks)}, Valid chunks: {len(valid_chunks)}")
            
            # Verify embeddings are properly initialized
            if not self.embeddings:
                st.error("Embeddings not properly initialized. Please check your OpenAI API key.")
                return
            
            # Add to vectorstore
            try:
                self.vectorstore.add_documents(valid_chunks)
            except Exception as embedding_error:
                st.error(f"Error adding documents to vectorstore: {str(embedding_error)}")
                st.error("This might be due to empty embeddings. Check if your OpenAI API key is valid and the content is meaningful.")
                return
            
            st.success(f"Processed {len(valid_chunks)} text chunks from {filename} with LangChain")
            
        except Exception as e:
            st.error(f"Error processing text {filename} with LangChain: {str(e)}")

    def process_text_file(self, uploaded_file, filename: str) -> bool:
        """Process uploaded text file and add to vector database"""
        try:
            # Read the text content
            text_content = uploaded_file.read().decode('utf-8')
            
            # Store original content for preview
            self.text_contents[filename] = text_content
            
            # Setup LangChain if not already done
            self.setup_langchain()
            
            # Process text with LangChain
            self.process_text_with_langchain(text_content, filename)
            
            # Track the uploaded text file
            self.text_files.add(filename)
            self.processed_files.add(filename)  # Mark as processed
            
            # AUTOMATIC: Run AI agents on the text content
            if os.getenv("OPENAI_API_KEY"):
                # Create a simple dataframe representation for analysis
                text_df = pd.DataFrame({
                    'content': [text_content],
                    'length': [len(text_content)],
                    'word_count': [len(text_content.split())],
                    'line_count': [len(text_content.split('\n'))]
                })
                # Note: LLM agents are now available as manual buttons in the main interface
            
            return True
        except Exception as e:
            st.error(f"Error processing text file {filename}: {str(e)}")
            return False

    def index_text_data(self, filename: str, text_content: str):
        """Index text data for semantic search"""
        if self.collection is None:
            st.error("ChromaDB collection not initialized")
            return

        try:
            # Split text into chunks (simple approach - can be improved)
            chunk_size = 1000  # characters per chunk
            overlap = 200  # characters overlap between chunks
            
            chunks = []
            start = 0
            
            while start < len(text_content):
                end = start + chunk_size
                chunk = text_content[start:end]
                chunks.append(chunk)
                start = end - overlap
                
                # Don't go beyond the end
                if start >= len(text_content):
                    break

            # Create documents and metadata for each chunk
            documents = []
            metadatas = []
            ids = []

            for idx, chunk in enumerate(chunks):
                # Clean the chunk (remove extra whitespace)
                clean_chunk = ' '.join(chunk.split())
                if len(clean_chunk.strip()) > 50:  # Only add chunks with meaningful content
                    documents.append(clean_chunk)
                    metadatas.append({
                        "chunk_index": idx,
                        "filename": filename,
                        "content_type": "text",
                        "total_chunks": len(chunks)
                    })
                    ids.append(f"{filename}_text_chunk_{idx}")

            # Add to ChromaDB
            if documents:
                self.collection.add(
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids
                )
                st.success(f"Indexed {len(documents)} text chunks from {filename} in ChromaDB")
            else:
                st.warning(f"No meaningful content found in {filename}")

        except Exception as e:
            st.error(f"Error indexing text data from {filename}: {str(e)}")

    def index_data(self, filename: str, df: pd.DataFrame):
        """Index data for semantic search"""
        if df is None:
            return

        if self.collection is None:
            st.error("ChromaDB collection not initialized")
            return

        try:
            # Create text representations of each row
            documents = []
            metadatas = []
            ids = []

            # Create semantic chunks instead of just individual rows
            chunk_size = 5  # Number of rows per chunk
            for chunk_start in range(0, len(df), chunk_size):
                chunk_end = min(chunk_start + chunk_size, len(df))
                chunk_df = df.iloc[chunk_start:chunk_end]
                
                # Create semantic representation of the chunk
                chunk_parts = []
                chunk_parts.append(f"Data chunk {chunk_start//chunk_size + 1} (rows {chunk_start}-{chunk_end-1}):")
                
                # Add summary stats for the chunk
                numeric_cols = chunk_df.select_dtypes(include=['number']).columns.tolist()
                if numeric_cols:
                    chunk_parts.append("Numeric summaries:")
                    for col in numeric_cols:
                        chunk_parts.append(f"  {col}: sum={chunk_df[col].sum():.2f}, avg={chunk_df[col].mean():.2f}")
                
                # Add individual rows with context
                for idx, row in chunk_df.iterrows():
                    row_text = " ".join([f"{col}: {val}" for col, val in row.items()])
                    chunk_parts.append(f"Row {idx}: {row_text}")
                
                chunk_text = "\n".join(chunk_parts)
                documents.append(chunk_text)
                
                # Enhanced metadata for chunks
                metadata = {
                    "chunk_index": chunk_start // chunk_size,
                    "start_row": chunk_start,
                    "end_row": chunk_end - 1,
                    "filename": filename,
                    "content_type": "data_chunk",
                    "row_count": len(chunk_df)
                }
                
                # Extract sheet name from filename if it contains sheet info
                if "_" in filename and not filename.endswith('.csv') and not filename.endswith('.xlsx'):
                    parts = filename.split("_", 1)
                    if len(parts) > 1:
                        metadata["original_file"] = parts[0]
                        metadata["sheet_name"] = parts[1]
                
                metadatas.append(metadata)
                ids.append(f"{filename}_chunk_{chunk_start//chunk_size}")

            # Validate that all arrays have the same length
            if len(documents) != len(metadatas) or len(documents) != len(ids):
                st.error(f"Length mismatch: documents={len(documents)}, metadatas={len(metadatas)}, ids={len(ids)}")
                return
            
            # Add to ChromaDB
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            
            # Add dataset-level summary document
            self.add_dataset_summary(filename, df)
            
            st.success(f"Indexed {len(documents)} chunks from {filename} in ChromaDB")
        except Exception as e:
            st.error(f"Error indexing data from {filename}: {str(e)}")

    def query_data(self, user_query: str) -> str:
        """Optimized query method using query preprocessing and adaptive search"""
        start_time = time.time()
        
        # Get or create session ID
        session_id = self.session_manager.get_or_create_session_id()
        
        if not self.dfs and not self.text_files:
            return "Please upload at least one CSV file or text document first."

        if self.vectorstore is None:
            st.error("LangChain vectorstore is None. Please try uploading your files again.")
            return "Please upload a CSV file or text document first to initialize the search index."

        # Generate query hash for caching
        query_hash = hashlib.md5(user_query.encode()).hexdigest()
        
        # Check cache first
        cached_result = self.query_optimizer.get_cached_result(query_hash)
        if cached_result:
            cache_time = time.time() - start_time
            self._record_performance_metrics(query_hash, 'cache_hit', cache_time)
            st.info(f"Returning cached result (retrieved in {cache_time:.2f}s)")
            
            # Save conversation turn to Redis even for cached results
            self.session_manager.save_conversation_turn(
                session_id=session_id,
                user_query=user_query,
                assistant_response=cached_result,
                execution_time=cache_time,
                metadata={'cache_hit': True, 'query_hash': query_hash}
            )
            
            return cached_result

        # Preprocess and optimize the query
        preprocessing_start = time.time()
        query_analysis = self.query_optimizer.preprocess_query(user_query)
        preprocessing_time = time.time() - preprocessing_start
        
        # Get adaptive search parameters
        k = self.query_optimizer.get_adaptive_k(query_analysis['complexity'], query_analysis['intent'])
        
        # Perform optimized search
        try:
            search_start = time.time()
            search_strategy = query_analysis['strategy']
            search_method = self.query_optimizer.search_strategies[search_strategy]
            docs = search_method(self.vectorstore, query_analysis['processed_query'], k)
            search_time = time.time() - search_start
            
            if not docs:
                response = "No relevant data found. Please make sure your files were processed successfully."
                
                # Save conversation turn to Redis
                total_time = time.time() - start_time
                self.session_manager.save_conversation_turn(
                    session_id=session_id,
                    user_query=user_query,
                    assistant_response=response,
                    execution_time=total_time,
                    metadata={'no_data_found': True, 'query_hash': query_hash}
                )
                
                return response

            # Process documents with optimized context building
            processing_start = time.time()
            relevant_data = self._process_search_results(docs)
            processing_time = time.time() - processing_start
            
            # Build optimized context
            context_start = time.time()
            context = self._build_optimized_context(query_analysis, relevant_data)
            context_time = time.time() - context_start
            
            # Generate response
            response_start = time.time()
            response = self._generate_response(context, user_query)
            response_time = time.time() - response_start
            
            # Cache the result
            self.query_optimizer.cache_query_result(query_hash, response)
            
            # Record performance metrics
            total_time = time.time() - start_time
            self._record_performance_metrics(query_hash, 'full_query', total_time, {
                'preprocessing_time': preprocessing_time,
                'search_time': search_time,
                'processing_time': processing_time,
                'context_time': context_time,
                'response_time': response_time,
                'strategy': search_strategy,
                'k': k,
                'docs_found': len(docs),
                'complexity': query_analysis['complexity']
            })
            
            # Add to query history for pattern analysis
            self.query_history.append({
                'query': user_query,
                'strategy': search_strategy,
                'complexity': query_analysis['complexity'],
                'execution_time': total_time,
                'timestamp': time.time()
            })
            
            # Save conversation turn to Redis
            self.session_manager.save_conversation_turn(
                session_id=session_id,
                user_query=user_query,
                assistant_response=response,
                execution_time=total_time,
                metadata={
                    'query_hash': query_hash,
                    'strategy': search_strategy,
                    'complexity': query_analysis['complexity'],
                    'docs_found': len(docs),
                    'k': k,
                    'cache_hit': False
                }
            )
            
            return response
            
        except Exception as e:
            error_time = time.time() - start_time
            self._record_performance_metrics(query_hash, 'error', error_time)
            error_response = f"Error processing query: {str(e)}"
            st.error(f"Error in optimized query: {str(e)}")
            
            # Save error conversation turn to Redis
            self.session_manager.save_conversation_turn(
                session_id=session_id,
                user_query=user_query,
                assistant_response=error_response,
                execution_time=error_time,
                metadata={'error': True, 'error_message': str(e), 'query_hash': query_hash}
            )
            
            return error_response
    
    def _record_performance_metrics(self, query_hash: str, query_type: str, execution_time: float, details: Dict[str, Any] = None):
        """Record performance metrics for analysis"""
        if query_hash not in self.performance_metrics:
            self.performance_metrics[query_hash] = []
        
        metric = {
            'type': query_type,
            'execution_time': execution_time,
            'timestamp': time.time()
        }
        
        if details:
            metric.update(details)
        
        self.performance_metrics[query_hash].append(metric)
    
    def get_performance_insights(self) -> Dict[str, Any]:
        """Get insights about query performance for optimization"""
        if not self.performance_metrics:
            return {"message": "No performance data available yet"}
        
        insights = {
            'total_queries': len(self.performance_metrics),
            'cache_hit_rate': 0,
            'average_execution_time': 0,
            'strategy_performance': {},
            'complexity_distribution': {},
            'slowest_queries': []
        }
        
        total_queries = 0
        cache_hits = 0
        total_time = 0
        strategy_times = {}
        complexity_counts = {}
        
        for query_hash, metrics in self.performance_metrics.items():
            for metric in metrics:
                total_queries += 1
                total_time += metric['execution_time']
                
                if metric['type'] == 'cache_hit':
                    cache_hits += 1
                
                if 'strategy' in metric:
                    strategy = metric['strategy']
                    if strategy not in strategy_times:
                        strategy_times[strategy] = []
                    strategy_times[strategy].append(metric['execution_time'])
                
                if 'complexity' in metric:
                    complexity = metric['complexity']
                    complexity_counts[complexity] = complexity_counts.get(complexity, 0) + 1
        
        if total_queries > 0:
            insights['cache_hit_rate'] = cache_hits / total_queries
            insights['average_execution_time'] = total_time / total_queries
        
        # Strategy performance
        for strategy, times in strategy_times.items():
            insights['strategy_performance'][strategy] = {
                'count': len(times),
                'average_time': sum(times) / len(times),
                'min_time': min(times),
                'max_time': max(times)
            }
        
        # Complexity distribution
        insights['complexity_distribution'] = complexity_counts
        
        # Find slowest queries
        all_queries = []
        for query_hash, metrics in self.performance_metrics.items():
            for metric in metrics:
                if metric['type'] == 'full_query':
                    all_queries.append((query_hash, metric['execution_time']))
        
        all_queries.sort(key=lambda x: x[1], reverse=True)
        insights['slowest_queries'] = all_queries[:5]
        
        return insights
    
    def display_performance_insights(self):
        """Display performance insights in Streamlit"""
        insights = self.get_performance_insights()
        
        if 'message' in insights:
            st.info(insights['message'])
            return
        
        st.subheader("🚀 Query Performance Insights")
        
        # Overall metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Queries", insights['total_queries'])
        with col2:
            st.metric("Cache Hit Rate", f"{insights['cache_hit_rate']:.1%}")
        with col3:
            st.metric("Avg Execution Time", f"{insights['average_execution_time']:.2f}s")
        
        # Strategy performance
        if insights['strategy_performance']:
            st.subheader("📊 Search Strategy Performance")
            strategy_data = []
            for strategy, perf in insights['strategy_performance'].items():
                strategy_data.append({
                    'Strategy': strategy,
                    'Count': perf['count'],
                    'Avg Time (s)': f"{perf['average_time']:.2f}",
                    'Min Time (s)': f"{perf['min_time']:.2f}",
                    'Max Time (s)': f"{perf['max_time']:.2f}"
                })
            
            strategy_df = pd.DataFrame(strategy_data)
            st.dataframe(strategy_df, use_container_width=True)
        
        # Complexity distribution
        if insights['complexity_distribution']:
            st.subheader("🎯 Query Complexity Distribution")
            complexity_df = pd.DataFrame([
                {'Complexity': k, 'Count': v} 
                for k, v in insights['complexity_distribution'].items()
            ])
            st.dataframe(complexity_df, use_container_width=True)
        
        # Slowest queries
        if insights['slowest_queries']:
            st.subheader("🐌 Slowest Queries")
            slow_queries_df = pd.DataFrame([
                {'Query Hash': qh[:8] + '...', 'Execution Time (s)': f"{time:.2f}"}
                for qh, time in insights['slowest_queries']
            ])
            st.dataframe(slow_queries_df, use_container_width=True)
    
    def get_query_optimization_recommendations(self) -> List[str]:
        """Get recommendations for further query optimization"""
        insights = self.get_performance_insights()
        recommendations = []
        
        if 'message' in insights:
            return ["Run some queries first to get optimization recommendations"]
        
        # Cache hit rate recommendations
        if insights['cache_hit_rate'] < 0.2:
            recommendations.append("🔧 Consider increasing cache TTL or implementing query similarity caching")
        
        # Strategy recommendations
        if insights['strategy_performance']:
            slowest_strategy = min(insights['strategy_performance'].items(), 
                                 key=lambda x: x[1]['average_time'])
            fastest_strategy = max(insights['strategy_performance'].items(), 
                                 key=lambda x: x[1]['average_time'])
            
            if slowest_strategy[1]['average_time'] > fastest_strategy[1]['average_time'] * 2:
                recommendations.append(f"⚡ {slowest_strategy[0]} strategy is significantly slower than {fastest_strategy[0]}. Consider optimizing query routing.")
        
        # Complexity recommendations
        if insights['complexity_distribution'].get('high', 0) > insights['complexity_distribution'].get('low', 0):
            recommendations.append("🎯 High complexity queries are common. Consider query simplification or better preprocessing.")
        
        # General recommendations
        if insights['average_execution_time'] > 5.0:
            recommendations.append("⏱️ Average query time is high. Consider implementing more aggressive caching or query optimization.")
        
        if not recommendations:
            recommendations.append("✅ Query performance looks good! No immediate optimizations needed.")
        
        return recommendations

    def _process_search_results(self, docs: List) -> List[Dict[str, Any]]:
        """Process search results with error handling"""
        relevant_data = []
        
        for doc in docs:
            metadata = doc.metadata
            if metadata and 'filename' in metadata:
                filename = metadata['filename']
                
                # Handle data rows (CSV or Excel)
                if 'row_index' in metadata and filename in self.dfs:
                    try:
                        row_idx = metadata['row_index']
                        row_data = self.dfs[filename].iloc[row_idx]
                        
                        data_type = 'csv_row' if 'sheet_name' not in metadata else 'excel_row'
                        relevant_data.append({
                            'filename': filename,
                            'type': data_type,
                            'data': row_data,
                            'sheet_name': metadata.get('sheet_name', None),
                            'original_file': metadata.get('original_file', filename),
                            'content': doc.page_content
                        })
                    except Exception as row_error:
                        st.warning(f"Error accessing row {row_idx} from {filename}: {str(row_error)}")
                        continue
                
                # Handle text data
                elif 'content_type' in metadata and metadata['content_type'] == 'text_chunk':
                    relevant_data.append({
                        'filename': filename,
                        'type': 'text_chunk',
                        'data': doc.page_content,
                        'content': doc.page_content
                    })
        
        return relevant_data
    
    def _build_optimized_context(self, query_analysis: Dict[str, Any], relevant_data: List[Dict[str, Any]]) -> str:
        """Build optimized context based on query analysis"""
        context_parts = []
        
        # Get conversation history for context
        session_id = self.session_manager.get_or_create_session_id()
        conversation_history = self.session_manager.get_conversation_history(session_id)
        
        # Add conversation history context if available
        if conversation_history:
            recent_context = self._build_conversation_context(conversation_history, query_analysis['original_query'])
            if recent_context:
                context_parts.append(f"Previous Conversation Context:\n{recent_context}")
        
        # Add aggregation results if needed
        if query_analysis['requires_aggregation']:
            aggregation_results = self.perform_aggregation(query_analysis['original_query'])
            if aggregation_results:
                context_parts.append(f"Aggregation Results: {aggregation_results}")
        
        # Add cross-dataset insights if needed
        if query_analysis['requires_cross_dataset']:
            cross_dataset_insights = self.generate_cross_dataset_insights(query_analysis['original_query'])
            if cross_dataset_insights:
                context_parts.append(f"Cross-Dataset Insights: {cross_dataset_insights}")
        
        # Add dataset summaries (cached)
        dataset_summaries = self._get_dataset_summaries()
        if dataset_summaries:
            context_parts.append(f"Available Datasets:\n{dataset_summaries}")
        
        # Add relevant data found
        if relevant_data:
            relevant_data_text = self._format_relevant_data(relevant_data)
            context_parts.append(f"Relevant Data for Query:\n{relevant_data_text}")
        else:
            context_parts.append("No relevant data found")
        
        return "\n\n".join(context_parts)
    
    def _get_dataset_summaries(self) -> str:
        """Get cached dataset summaries"""
        if not self.dataset_summaries:
            summaries = []
            for filename, df in self.dfs.items():
                if "_" in filename and not filename.endswith(('.csv', '.xlsx')):
                    # Excel sheet
                    parts = filename.split("_", 1)
                    original_file = parts[0]
                    sheet_name = parts[1]
                    summaries.append(f"""
                    Dataset: {filename}
                    Type: Excel Sheet
                    Original File: {original_file}
                    Sheet Name: {sheet_name}
                    Total Rows: {len(df)}
                    Columns: {list(df.columns)}
                    """)
                else:
                    # Regular file
                    file_type = "Excel" if filename.endswith(('.xlsx', '.xls')) else "CSV"
                    summaries.append(f"""
                    Dataset: {filename}
                    Type: {file_type}
                    Columns: {list(df.columns)}
                    Total Rows: {len(df)}
                    """)
            
            # Add text files
            for filename in self.text_files:
                summaries.append(f"""
                Document: {filename}
                Type: Text Document
                Status: Indexed in vector database
                """)
            
            self.dataset_summaries = "\n".join(summaries)
        
        return self.dataset_summaries
    
    def _format_relevant_data(self, relevant_data: List[Dict[str, Any]]) -> str:
        """Format relevant data for context"""
        formatted_parts = []
        
        for item in relevant_data:
            if item['type'] == 'csv_row':
                formatted_parts.append(f"""
                From CSV file {item['filename']}:
                {item['data'].to_string()}
                """)
            elif item['type'] == 'excel_row':
                sheet_info = f" (Sheet: {item['sheet_name']})" if item['sheet_name'] else ""
                original_file = item.get('original_file', item['filename'])
                formatted_parts.append(f"""
                From Excel file {original_file}{sheet_info}:
                {item['data'].to_string()}
                """)
            elif item['type'] == 'text_chunk':
                formatted_parts.append(f"""
                From text document {item['filename']}:
                {item['data']}
                """)
        
        return "\n".join(formatted_parts)
    
    def _build_conversation_context(self, conversation_history: List[Dict[str, Any]], current_query: str) -> str:
        """Build context from previous conversation turns that are relevant to current query"""
        if not conversation_history:
            return ""
        
        # Get the last 3 conversation turns for context
        recent_turns = conversation_history[-3:]
        
        # Simple keyword matching to find relevant previous answers
        current_query_lower = current_query.lower()
        relevant_contexts = []
        
        for turn in recent_turns:
            previous_query = turn['user_query'].lower()
            previous_response = turn['assistant_response']
            
            # Check if there's semantic similarity or keyword overlap
            similarity_score = self._calculate_query_similarity(current_query_lower, previous_query)
            
            if similarity_score > 0.3:  # Threshold for relevance
                relevant_contexts.append(f"""
                Previous Q: {turn['user_query']}
                Previous A: {previous_response[:200]}{'...' if len(previous_response) > 200 else ''}
                """)
        
        if relevant_contexts:
            return "Based on our previous conversation:\n" + "\n".join(relevant_contexts)
        
        return ""
    
    def _calculate_query_similarity(self, query1: str, query2: str) -> float:
        """Calculate simple similarity between two queries"""
        # Simple word overlap similarity
        words1 = set(query1.split())
        words2 = set(query2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    def _generate_response(self, context: str, user_query: str) -> str:
        """Generate response using OpenAI with optimized context"""
        try:
            # Get conversation history for better context
            session_id = self.session_manager.get_or_create_session_id()
            conversation_history = self.session_manager.get_conversation_history(session_id)
            
            # Build conversation history messages
            messages = [
                {"role": "system", "content": "You are a helpful data analyst assistant. The data you are looking at is advertising spend data related to marketing campaigns. Sometimes the data is spread across multiple sheets. Your job is to identify helpful information and insights across all of the sheets. When there are multiple sheets, consider information from all of the sheets before giving a response. Be concise and focused on the specific query. If the user asks a similar question to what you've answered before, reference your previous response and build upon it."}
            ]
            
            # Add recent conversation history (last 2 turns) for context
            if conversation_history:
                recent_turns = conversation_history[-2:]  # Last 2 turns
                for turn in recent_turns:
                    messages.append({"role": "user", "content": turn['user_query']})
                    messages.append({"role": "assistant", "content": turn['assistant_response']})
            
            # Add current context and query
            messages.append({"role": "user", "content": f"{context}\n\nUser Query: {user_query}"})
            
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=1500,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating response: {str(e)}"

    def add_dataset_summary(self, filename: str, df: pd.DataFrame):
        """Add a comprehensive dataset summary to the vector database"""
        try:
            # Create comprehensive dataset summary
            summary_parts = []
            
            # Basic info
            summary_parts.append(f"Dataset: {filename}")
            summary_parts.append(f"Total rows: {len(df)}")
            summary_parts.append(f"Total columns: {len(df.columns)}")
            
            # Column analysis
            summary_parts.append("Columns:")
            for col in df.columns:
                col_type = str(df[col].dtype)
                unique_count = df[col].nunique()
                null_count = df[col].isnull().sum()
                
                if df[col].dtype in ['int64', 'float64']:
                    # Numeric column stats
                    min_val = df[col].min()
                    max_val = df[col].max()
                    mean_val = df[col].mean()
                    summary_parts.append(f"  - {col} ({col_type}): {unique_count} unique values, range {min_val:.2f} to {max_val:.2f}, mean {mean_val:.2f}")
                else:
                    # Categorical column stats
                    top_values = df[col].value_counts().head(3).to_dict()
                    summary_parts.append(f"  - {col} ({col_type}): {unique_count} unique values, top values: {top_values}")
            
            # Data patterns and insights
            summary_parts.append("Data Patterns:")
            
            # Find relationships between columns
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            
            if len(numeric_cols) >= 2:
                # Correlation insights
                corr_matrix = df[numeric_cols].corr()
                high_corr_pairs = []
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        corr_val = corr_matrix.iloc[i, j]
                        if abs(corr_val) > 0.7:  # High correlation threshold
                            high_corr_pairs.append(f"{corr_matrix.columns[i]} and {corr_matrix.columns[j]} (correlation: {corr_val:.2f})")
                
                if high_corr_pairs:
                    summary_parts.append(f"  - High correlations: {', '.join(high_corr_pairs)}")
            
            # Grouping insights
            if categorical_cols and numeric_cols:
                for cat_col in categorical_cols[:3]:  # Limit to first 3 categorical columns
                    for num_col in numeric_cols[:3]:  # Limit to first 3 numeric columns
                        grouped_stats = df.groupby(cat_col)[num_col].agg(['mean', 'count']).reset_index()
                        top_group = grouped_stats.loc[grouped_stats['mean'].idxmax()]
                        summary_parts.append(f"  - {cat_col} '{top_group[cat_col]}' has highest average {num_col}: {top_group['mean']:.2f}")
            
            # Sample data patterns
            summary_parts.append("Sample Data Patterns:")
            for i in range(min(3, len(df))):
                row = df.iloc[i]
                pattern = f"  Row {i}: " + ", ".join([f"{col}={val}" for col, val in row.items() if pd.notna(val)][:5])
                summary_parts.append(pattern)
            
            # Create the summary document
            summary_text = "\n".join(summary_parts)
            
            # Add to ChromaDB
            self.collection.add(
                documents=[summary_text],
                metadatas=[{
                    "filename": filename,
                    "content_type": "dataset_summary",
                    "row_count": len(df),
                    "column_count": len(df.columns)
                }],
                ids=[f"{filename}_summary"]
            )
            
        except Exception as e:
            st.warning(f"Error creating dataset summary for {filename}: {str(e)}")

    def perform_aggregation(self, query: str) -> str:
        """Perform data aggregations based on user query"""
        query_lower = query.lower()
        results = []
        
        for filename, df in self.dfs.items():
            try:
                # Find numeric columns
                numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                
                # Find categorical columns for grouping
                categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
                
                # Check for aggregation requests
                if 'sum' in query_lower:
                    # Find what to sum
                    for col in numeric_cols:
                        if col.lower() in query_lower or any(word in col.lower() for word in ['spend', 'amount', 'value', 'cost']):
                            # Check if grouping is requested
                            if 'by' in query_lower:
                                # Find group by column
                                for cat_col in categorical_cols:
                                    if cat_col.lower() in query_lower or any(word in cat_col.lower() for word in ['platform', 'channel', 'category', 'type']):
                                        grouped = df.groupby(cat_col)[col].sum().reset_index()
                                        results.append(f"Sum of {col} by {cat_col} in {filename}:\n{grouped.to_string()}")
                                        break
                            else:
                                total = df[col].sum()
                                results.append(f"Total sum of {col} in {filename}: {total:,.2f}")
                
                elif 'average' in query_lower or 'mean' in query_lower:
                    for col in numeric_cols:
                        if col.lower() in query_lower or any(word in col.lower() for word in ['spend', 'amount', 'value', 'cost']):
                            if 'by' in query_lower:
                                for cat_col in categorical_cols:
                                    if cat_col.lower() in query_lower or any(word in cat_col.lower() for word in ['platform', 'channel', 'category', 'type']):
                                        grouped = df.groupby(cat_col)[col].mean().reset_index()
                                        results.append(f"Average of {col} by {cat_col} in {filename}:\n{grouped.to_string()}")
                                        break
                            else:
                                avg = df[col].mean()
                                results.append(f"Average of {col} in {filename}: {avg:,.2f}")
                
                elif 'count' in query_lower:
                    if 'by' in query_lower:
                        for cat_col in categorical_cols:
                            if cat_col.lower() in query_lower:
                                counts = df[cat_col].value_counts().reset_index()
                                counts.columns = [cat_col, 'count']
                                results.append(f"Count by {cat_col} in {filename}:\n{counts.to_string()}")
                                break
                    else:
                        results.append(f"Total count in {filename}: {len(df)} rows")
                        
            except Exception as e:
                results.append(f"Error processing {filename}: {str(e)}")
        
        return "\n\n".join(results) if results else ""

    def generate_cross_dataset_insights(self, query: str) -> str:
        """Generate insights across multiple datasets"""
        if len(self.dfs) < 2:
            return ""
        
        insights = []
        query_lower = query.lower()
        
        try:
            # Find common columns across datasets
            all_columns = {}
            for filename, df in self.dfs.items():
                for col in df.columns:
                    if col not in all_columns:
                        all_columns[col] = []
                    all_columns[col].append(filename)
            
            # Find columns that appear in multiple datasets
            common_columns = {col: files for col, files in all_columns.items() if len(files) > 1}
            
            if common_columns:
                insights.append("Cross-Dataset Analysis:")
                
                for col, files in common_columns.items():
                    if col.lower() in query_lower or any(word in col.lower() for word in ['spend', 'amount', 'value', 'platform', 'channel']):
                        insights.append(f"\nColumn '{col}' appears in {len(files)} datasets: {', '.join(files)}")
                        
                        # Compare values across datasets
                        col_values = {}
                        for filename in files:
                            df = self.dfs[filename]
                            if col in df.columns:
                                if df[col].dtype in ['int64', 'float64']:
                                    col_values[filename] = {
                                        'sum': df[col].sum(),
                                        'mean': df[col].mean(),
                                        'count': len(df)
                                    }
                                else:
                                    col_values[filename] = {
                                        'unique': df[col].nunique(),
                                        'top_values': df[col].value_counts().head(3).to_dict()
                                    }
                        
                        # Add comparison insights
                        if col_values:
                            insights.append(f"  Comparison for '{col}':")
                            for filename, stats in col_values.items():
                                if 'sum' in stats:
                                    insights.append(f"    {filename}: sum={stats['sum']:.2f}, avg={stats['mean']:.2f}, count={stats['count']}")
                                else:
                                    insights.append(f"    {filename}: {stats['unique']} unique values, top: {stats['top_values']}")
            
            # Add overall dataset comparison
            total_rows = sum(len(df) for df in self.dfs.values())
            insights.append(f"\nOverall: {len(self.dfs)} datasets with {total_rows} total rows")
            
        except Exception as e:
            insights.append(f"Error in cross-dataset analysis: {str(e)}")
        
        return "\n".join(insights)

    def generate_visualization(self, query: str):
        """Generate visualizations based on user query"""
        if not self.dfs:
            return None

        # Parse the visualization intent
        intent = self.parse_visualization_intent(query)
        
        if not intent['chart_type']:
            return None
            
        # Generate the appropriate chart
        try:
            if intent['chart_type'] == 'bar':
                return self.create_bar_chart(intent)
            elif intent['chart_type'] == 'line':
                return self.create_line_chart(intent)
            elif intent['chart_type'] == 'scatter':
                return self.create_scatter_chart(intent)
            elif intent['chart_type'] == 'pie':
                return self.create_pie_chart(intent)
            elif intent['chart_type'] == 'histogram':
                return self.create_histogram_chart(intent)
            elif intent['chart_type'] == 'box':
                return self.create_box_chart(intent)
        except Exception as e:
            st.error(f"Error creating chart: {str(e)}")
            return None

    def create_bar_chart(self, intent):
        """Create a bar chart based on parsed intent"""
        if not intent['columns']:
            return None
            
        # Get the first dataset and columns
        df_name = intent['datasets'][0] if intent['datasets'] else list(self.dfs.keys())[0]
        df = self.dfs[df_name]
        
        # Validate columns exist
        for col in intent['columns']:
            if col not in df.columns:
                st.error(f"Column '{col}' not found in dataset '{df_name}'. Available columns: {list(df.columns)}")
                return None
        
        # Handle grouping with LLM-extracted parameters
        if intent['parameters'].get('group_by') and intent['parameters']['group_by'] != 'null':
            group_col = intent['parameters']['group_by']
            
            # Use the exact column name from LLM parsing
            if group_col in df.columns:
                value_col = intent['columns'][0] if intent['columns'] else df.columns[0]
                
                try:
                    # Aggregate based on LLM-extracted aggregation type
                    aggregation = intent['parameters'].get('aggregation', 'count')
                    if aggregation == 'mean':
                        data = df.groupby(group_col)[value_col].mean().reset_index()
                    elif aggregation == 'sum':
                        data = df.groupby(group_col)[value_col].sum().reset_index()
                    elif aggregation == 'count':
                        data = df.groupby(group_col)[value_col].count().reset_index()
                    else:
                        data = df.groupby(group_col)[value_col].count().reset_index()
                    
                    fig = px.bar(data, x=group_col, y=value_col,
                               title=f"{aggregation.title()} of {value_col} by {group_col}")
                    return fig
                except Exception as e:
                    st.error(f"Error creating grouped bar chart: {str(e)}")
                    return None
        
        # Simple bar chart - use first column as x-axis
        if len(intent['columns']) >= 1:
            col = intent['columns'][0]
            fig = px.bar(df, x=col, title=f"Distribution of {col}")
            return fig
        
        return None

    def create_line_chart(self, intent):
        """Create a line chart based on parsed intent"""
        if not intent['columns']:
            return None
            
        df_name = intent['datasets'][0] if intent['datasets'] else list(self.dfs.keys())[0]
        df = self.dfs[df_name]
        
        # Look for date columns for time series
        date_cols = [col for col in df.columns if 'date' in col.lower()]
        
        if date_cols and intent['columns']:
            date_col = date_cols[0]
            value_col = intent['columns'][0]
            
            # Convert to datetime if needed
            df_copy = df.copy()
            df_copy[date_col] = pd.to_datetime(df_copy[date_col])
            df_copy = df_copy.sort_values(date_col)
            
            fig = px.line(df_copy, x=date_col, y=value_col,
                         title=f"{value_col} over time")
            return fig
        
        return None

    def create_scatter_chart(self, intent):
        """Create a scatter plot based on parsed intent"""
        if len(intent['columns']) < 2:
            return None
            
        df_name = intent['datasets'][0] if intent['datasets'] else list(self.dfs.keys())[0]
        df = self.dfs[df_name]
        
        x_col = intent['columns'][0]
        y_col = intent['columns'][1]
        
        fig = px.scatter(df, x=x_col, y=y_col,
                        title=f"{y_col} vs {x_col}")
        return fig

    def create_pie_chart(self, intent):
        """Create a pie chart based on parsed intent"""
        if not intent['columns']:
            return None
            
        df_name = intent['datasets'][0] if intent['datasets'] else list(self.dfs.keys())[0]
        df = self.dfs[df_name]
        
        col = intent['columns'][0]
        
        # Count values for pie chart
        value_counts = df[col].value_counts()
        
        fig = px.pie(values=value_counts.values, names=value_counts.index,
                    title=f"Distribution of {col}")
        return fig

    def create_histogram_chart(self, intent):
        """Create a histogram based on parsed intent"""
        if not intent['columns']:
            return None
            
        df_name = intent['datasets'][0] if intent['datasets'] else list(self.dfs.keys())[0]
        df = self.dfs[df_name]
        
        col = intent['columns'][0]
        
        fig = px.histogram(df, x=col, title=f"Distribution of {col}")
        return fig

    def create_box_chart(self, intent):
        """Create a box plot based on parsed intent"""
        if not intent['columns']:
            return None
            
        df_name = intent['datasets'][0] if intent['datasets'] else list(self.dfs.keys())[0]
        df = self.dfs[df_name]
        
        col = intent['columns'][0]
        
        fig = px.box(df, y=col, title=f"Box plot of {col}")
        return fig

    def parse_visualization_intent(self, query: str):
        """Parse visualization intent using LLM for better accuracy"""
        try:
            # Get available columns from all datasets
            available_columns = {}
            for df_name, df in self.dfs.items():
                available_columns[df_name] = list(df.columns)
            
            # Use LLM to parse the entire visualization intent
            prompt = f"""
            Parse this visualization query: "{query}"
            
            Available datasets and columns:
            {json.dumps(available_columns, indent=2)}
            
            Return ONLY a JSON object with:
            {{
                "chart_type": "bar|line|scatter|pie|histogram|box",
                "columns": ["column_name1", "column_name2"],
                "datasets": ["dataset_name"],
                "parameters": {{
                    "aggregation": "sum|mean|count|null",
                    "group_by": "column_name|null"
                }}
            }}
            
            Rules:
            - chart_type: Choose the most appropriate chart type based on the query
            - columns: Match exact column names from available columns
            - datasets: Choose the most relevant dataset(s)
            - aggregation: Extract if query mentions sum/average/count
            - group_by: Extract grouping column if query mentions "by" or "group by"
            - Return only the JSON, no other text
            """
            
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.1
            )
            
            # Parse the JSON response
            response_text = response.choices[0].message.content.strip()
            
            # Extract JSON from response
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            if json_start != -1 and json_end != 0:
                json_str = response_text[json_start:json_end]
                parsed = json.loads(json_str)
                
                return {
                    'chart_type': parsed.get('chart_type'),
                    'columns': parsed.get('columns', []),
                    'datasets': parsed.get('datasets', []),
                    'parameters': parsed.get('parameters', {})
                }
            else:
                # Fallback to original method
                return self._fallback_parse_visualization_intent(query)
                
        except Exception as e:
            st.warning(f"LLM visualization parsing failed: {str(e)}. Using fallback method.")
            return self._fallback_parse_visualization_intent(query)
    
    def _fallback_parse_visualization_intent(self, query: str):
        """Fallback visualization intent parsing"""
        query_lower = query.lower()
        
        # Step 1: Detect chart type
        chart_type = self.detect_chart_type(query_lower)
        
        # Step 2: Extract columns and datasets
        columns, datasets = self._fallback_column_extraction(query_lower)
        
        # Step 3: Extract additional parameters
        params = self.extract_chart_parameters(query_lower)
        
        return {
            'chart_type': chart_type,
            'columns': columns,
            'datasets': datasets,
            'parameters': params
        }

    def detect_chart_type(self, query: str):
        chart_patterns = {
            'bar': ['bar', 'bars', 'bar chart', 'bar graph'],
            'line': ['line', 'line chart', 'trend', 'over time', 'timeline'],
            'scatter': ['scatter', 'scatter plot', 'correlation', 'relationship'],
            'pie': ['pie', 'pie chart', 'proportion', 'percentage', 'share'],
            'histogram': ['histogram', 'distribution', 'frequency'],
            'box': ['box', 'box plot', 'boxplot', 'quartile']
        }
        
        for chart_type, keywords in chart_patterns.items():
            if any(keyword in query for keyword in keywords):
                return chart_type
        return None

    def extract_columns_and_datasets(self, query: str):
        """Use LLM to intelligently match columns"""
        try:
            # Get available columns from all datasets
            available_columns = {}
            for df_name, df in self.dfs.items():
                available_columns[df_name] = list(df.columns)
            
            # Use LLM to parse the query
            prompt = f"""
            Parse this visualization query: "{query}"
            
            Available datasets and columns:
            {json.dumps(available_columns, indent=2)}
            
            Return ONLY a JSON object with:
            {{
                "columns": ["column_name1", "column_name2"],
                "datasets": ["dataset_name"],
                "chart_type": "bar|line|scatter|pie|histogram|box"
            }}
            
            Rules:
            - Match column names exactly as they appear in the available columns
            - Choose the most relevant columns for the query
            - If query mentions specific columns, use those
            - If query is vague, choose numeric columns for y-axis and categorical for x-axis
            - Return only the JSON, no other text
            """
            
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.1
            )
            
            # Parse the JSON response
            response_text = response.choices[0].message.content.strip()
            
            # Extract JSON from response (in case there's extra text)
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            if json_start != -1 and json_end != 0:
                json_str = response_text[json_start:json_end]
                parsed = json.loads(json_str)
                
                return parsed.get('columns', []), parsed.get('datasets', [])
            else:
                # Fallback if JSON parsing fails
                return self._fallback_column_extraction(query)
                
        except Exception as e:
            st.warning(f"LLM column extraction failed: {str(e)}. Using fallback method.")
            return self._fallback_column_extraction(query)
    
    def _fallback_column_extraction(self, query: str):
        """Fallback column extraction method"""
        columns = []
        datasets = []
        
        # Check each dataset and its columns
        for df_name, df in self.dfs.items():
            for col in df.columns:
                col_lower = col.lower().replace('_', ' ')
                # More flexible matching
                if (col_lower in query or 
                    col in query or 
                    col_lower.replace(' ', '') in query.replace(' ', '') or
                    any(word in col_lower for word in query.split())):
                    columns.append(col)
                    datasets.append(df_name)
        
        return columns, datasets

    def extract_chart_parameters(self, query: str):
        params = {}
        
        # Extract aggregation functions
        if 'average' in query or 'mean' in query:
            params['aggregation'] = 'mean'
        elif 'sum' in query:
            params['aggregation'] = 'sum'
        elif 'count' in query:
            params['aggregation'] = 'count'
        
        # Extract grouping
        if 'by' in query:
            # Find what comes after "by"
            by_index = query.find('by')
            after_by = query[by_index:].split()[1] if len(query[by_index:].split()) > 1 else None
            if after_by:
                params['group_by'] = after_by
        
        return params

def main():
    st.set_page_config(
        page_title="ABI - Agentic Business Intelligence",
        page_icon="assets/abi_lime.png",
        layout="wide"
    )

    # Compact header with smaller logo and inline description
    col1 , col2= st.columns([1, 1])
    with col1:
        st.image("assets/abi_horiizontal_lime.png", width=300)
    # with col2:
    #     st.markdown("### Upload documents are start gathering insights")
  

    # Initialize app
    if 'app' not in st.session_state:
        st.session_state.app = CSVChatApp()

    app = st.session_state.app
    
    # Initialize session management
    session_id = app.session_manager.get_or_create_session_id()
    
    # Combined sidebar for all sidebar content
    with st.sidebar:
        st.header("🆔 Session Info")
        st.info(f"**Session ID:** {session_id[:8]}...")
        
        # Show session metadata
        session_metadata = app.session_manager.get_session_metadata(session_id)
        if session_metadata:
            st.write(f"**Created:** {session_metadata.get('created_at', 'Unknown')[:19]}")
            st.write(f"**Last Activity:** {session_metadata.get('last_activity', 'Unknown')[:19]}")
            st.write(f"**Total Turns:** {session_metadata.get('total_turns', 0)}")
        
        # Show conversation history
        conversation_history = app.session_manager.get_conversation_history(session_id)
        if conversation_history:
            with st.expander(f"📝 Conversation History ({len(conversation_history)} turns)"):
                for i, turn in enumerate(conversation_history[-5:], 1):  # Show last 5 turns
                    st.write(f"**Turn {i}:**")
                    st.write(f"**Q:** {turn['user_query'][:100]}{'...' if len(turn['user_query']) > 100 else ''}")
                    st.write(f"**A:** {turn['assistant_response'][:100]}{'...' if len(turn['assistant_response']) > 100 else ''}")
                    st.write(f"**Time:** {turn['timestamp'][:19]}")
                    st.write("---")
        
        # Clear session button
        if st.button("🗑️ Clear Session"):
            if app.session_manager.delete_session(session_id):
                st.success("Session cleared successfully!")
                # Generate new session ID
                st.session_state.session_id = app.session_manager.generate_session_id()
                st.rerun()
            else:
                st.error("Failed to clear session")


        # Upload more data toggle (only show when data is already loaded)
        if app.dfs or app.text_files:
            with st.expander("📁 Upload More Data", expanded=False):
                st.markdown("Upload additional CSV, Excel, or text files.")
                
                additional_files = st.file_uploader(
                    "Choose additional files",
                    type=['csv', 'xlsx', 'xls', 'txt'],
                    accept_multiple_files=True,
                    help="Upload additional CSV, Excel, or text files.",
                    key="additional_uploader"
                )

                if additional_files:
                    # Process files one by one with better error handling
                    for uploaded_file in additional_files:
                        try:
                            # Check if this file has already been processed
                            file_already_processed = uploaded_file.name in app.processed_files
                            
                            if not file_already_processed:
                                st.info(f"Processing {uploaded_file.name}...")
                                
                                # Determine file type and process accordingly
                                if uploaded_file.name.lower().endswith('.csv'):
                                    df = app.process_csv(uploaded_file, uploaded_file.name)
                                    if df is not None:
                                        st.success(f"✅ {uploaded_file.name} processed!")
                                    else:
                                        st.error(f"❌ Failed to process {uploaded_file.name}")
                                        
                                elif uploaded_file.name.lower().endswith(('.xlsx', '.xls')):
                                    df = app.process_excel(uploaded_file, uploaded_file.name)
                                    if df is not None:
                                        st.success(f"✅ {uploaded_file.name} processed!")
                                    else:
                                        st.error(f"❌ Failed to process {uploaded_file.name}")
                                        
                                elif uploaded_file.name.lower().endswith('.txt'):
                                    success = app.process_text_file(uploaded_file, uploaded_file.name)
                                    if success:
                                        st.success(f"✅ Text file '{uploaded_file.name}' processed!")
                                    else:
                                        st.error(f"❌ Failed to process text file '{uploaded_file.name}'")
                                else:
                                    st.error(f"❌ Unsupported file type: {uploaded_file.name}")
                            else:
                                st.info(f"✅ {uploaded_file.name} already processed")
                                
                        except Exception as e:
                            st.error(f"❌ Error processing {uploaded_file.name}: {str(e)}")
                    
                    # Show completion message without rerun
                    st.success("🎉 File processing complete!")
        
        # Show loaded datasets
        if app.dfs or app.text_files:
            st.header("📊 Loaded Datasets")
            
            # Add button to clear all data
            if st.button("🗑️ Clear All Data"):
                app.dfs.clear()
                app.text_files.clear()
                app.text_contents.clear()
                app.processed_files.clear()  # Clear processed files tracking
                
                # Clear the vectorstore
                app.clear_vectorstore()
                
                st.success("All data cleared!")
                st.rerun()
            
            # Show CSV datasets
            for filename, df in app.dfs.items():
                with st.expander(f"📊 {filename} ({len(df)} rows)"):
                    st.dataframe(df.head(10000))
                    st.write(f"**Columns:** {list(df.columns)}")
                    
                    # Add remove button for individual datasets
                    if st.button(f"❌ Remove {filename}", key=f"remove_csv_{filename}"):
                        # Remove from memory
                        del app.dfs[filename]
                        
                        # Remove from processed files tracking
                        if filename in app.processed_files:
                            app.processed_files.remove(filename)
                        
                        # Note: LangChain Chroma doesn't support direct deletion by metadata
                        # The document will remain in the vectorstore but won't be accessible
                        # through the app interface
                        st.info(f"Removed {filename} from memory. Note: Vector embeddings remain in database.")
                        
                        st.rerun()
            
            # Show text files
            for filename in app.text_files:
                with st.expander(f"📄 {filename}"):
                    st.write(f"**Type:** Text Document")
                    st.write(f"**Status:** Indexed in vector database")
                    
                    # Show text preview
                    if filename in app.text_contents:
                        text_content = app.text_contents[filename]
                        # Count chunks (approximate)
                        chunk_count = len(text_content) // 1000 + 1
                        st.write(f"**Chunks:** {chunk_count} chunks indexed")
                        
                        # Show preview (first 500 characters)
                        preview_length = 500
                        preview_text = text_content[:preview_length]
                        if len(text_content) > preview_length:
                            preview_text += "..."
                        
                        st.write("**Preview:**")
                        st.text_area("", value=preview_text, height=150, disabled=True, key=f"preview_{filename}")
                        
                        # Show full document button
                        if st.button(f"📖 View Full Document", key=f"view_full_{filename}"):
                            st.session_state[f"show_full_{filename}"] = not st.session_state.get(f"show_full_{filename}", False)
                        
                        # Show full document if requested
                        if st.session_state.get(f"show_full_{filename}", False):
                            st.markdown("---")
                            st.markdown("### 📖 Full Document Content")
                            st.text_area("", value=text_content, height=400, disabled=True, key=f"full_{filename}")
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write(f"**Document Length:** {len(text_content)} characters")
                            with col2:
                                st.write(f"**Estimated Chunks:** {len(text_content) // 1000 + 1}")
                            st.markdown("---")
                    
                    # Add remove button for text files
                    if st.button(f"❌ Remove {filename}", key=f"remove_text_{filename}"):
                        # Remove from tracking
                        app.text_files.remove(filename)
                        if filename in app.text_contents:
                            del app.text_contents[filename]
                        
                        # Remove from processed files tracking
                        if filename in app.processed_files:
                            app.processed_files.remove(filename)
                        
                        # Note: LangChain Chroma doesn't support direct deletion by metadata
                        # The document will remain in the vectorstore but won't be accessible
                        # through the app interface
                        st.info(f"Removed {filename} from memory. Note: Vector embeddings remain in database.")
                        
                        st.rerun()

        # Session Management
        st.header("🆔 Session Management")
        
        # Show current session info
        current_session_id = app.session_manager.get_or_create_session_id()
        st.info(f"**Current Session**: {current_session_id[:8]}...")
        
        # Session history
        with st.expander("📚 Session History"):
            history_summary = app.get_session_history_summary()
            st.write(history_summary)
            
            # Show detailed history
            all_sessions = app.session_manager.get_all_sessions()
            if all_sessions:
                st.subheader("Detailed History")
                for session in all_sessions[:3]:  # Show last 3 sessions
                    # Get interaction count safely
                    interaction_count = session.get('total_turns', 0)
                    
                    # Use a container instead of nested expander
                    with st.container():
                        st.markdown(f"**Session {session['session_id'][:8]}... ({interaction_count} interactions)**")
                        st.write(f"**Last Activity**: {session.get('last_activity', 'Unknown')}")
                        st.write(f"**Interactions**: {interaction_count}")
                        
                        # Show conversation history for this session
                        conversation = app.session_manager.get_conversation_history(session['session_id'])
                        if conversation:
                            st.write("**Recent Interactions**:")
                            for turn in conversation[-3:]:  # Show last 3 interactions
                                timestamp = turn.get('timestamp', 'Unknown')[:19] if turn.get('timestamp') else 'Unknown'
                                user_query = turn.get('user_query', 'No query')[:50] + '...' if turn.get('user_query') else 'No query'
                                st.write(f"- **{timestamp}**: {user_query}")
        
        # Reset session button
        if st.button("🔄 Reset Session"):
            app.reset_session()
            st.rerun()
        
        # Configuration info
        st.header("🔑 Configuration")
        if os.getenv("OPENAI_API_KEY"):
            st.success("✅ OpenAI API Key configured")
        else:
            st.error("❌ OpenAI API Key not found in .env file")
            st.info("Please add OPENAI_API_KEY to your .env file")

    # Main content area
    if not (app.dfs or app.text_files):
        # Upload section when no data is loaded
        st.header("📁 Add Your Data")
        st.markdown("Choose Excel or text files to start analyzing with AI.")
        
        # Unified file uploader
        uploaded_files = st.file_uploader(
            "Select files",
            type=['xlsx', 'xls', 'txt'],
            accept_multiple_files=True,
            help="Excel and text files will be processed automatically."
        )

        if uploaded_files:
            for uploaded_file in uploaded_files:
                # Check if this file has already been processed
                file_already_processed = uploaded_file.name in app.processed_files
                
                # Also check for Excel sheets that might have been processed
                if not file_already_processed:
                    for existing_filename in app.processed_files:
                        if existing_filename.startswith(uploaded_file.name + "_"):
                            file_already_processed = True
                            break
                
                if not file_already_processed or st.button(f"Reload {uploaded_file.name}"):
                    with st.spinner(f"Processing {uploaded_file.name}..."):
                        # Determine file type and process accordingly
                        if uploaded_file.name.lower().endswith('.csv'):
                            df = app.process_csv(uploaded_file, uploaded_file.name)
                            if df is not None:
                                st.success(f"✅ {uploaded_file.name} processed successfully!")
                                st.write(f"**Shape:** {df.shape} | **Columns:** {list(df.columns)}")
                        elif uploaded_file.name.lower().endswith(('.xlsx', '.xls')):
                            df = app.process_excel(uploaded_file, uploaded_file.name)
                            if df is not None:
                                st.success(f"✅ {uploaded_file.name} processed successfully!")
                                if hasattr(df, 'shape'):
                                    st.write(f"**Shape:** {df.shape} | **Columns:** {list(df.columns)}")
                                else:
                                    st.write("**Processed Sheets:**")
                                    st.dataframe(df)
                        elif uploaded_file.name.lower().endswith('.txt'):
                            success = app.process_text_file(uploaded_file, uploaded_file.name)
                            if success:
                                st.success(f"✅ Text file '{uploaded_file.name}' processed and added to vector database!")
                            else:
                                st.error(f"❌ Failed to process text file '{uploaded_file.name}'")
                        else:
                            st.error(f"❌ Unsupported file type: {uploaded_file.name}")
                            continue
                else:
                    st.info(f"✅ {uploaded_file.name} already processed")
            
            # Show a success message and prompt to start chatting
            if uploaded_files:
                st.success("🎉 Files processed! You can now start chatting with your data.")
                st.markdown("---")
    
    # Main chat interface
    if app.dfs or app.text_files:
        st.header("💬 Chat with Your Data")
        
        # AI Agents Section
        st.subheader("🤖 AI Analysis Agents")
        st.markdown("Run intelligent analysis on all your uploaded data:")
        
        # Check if OpenAI API key is available
        if not os.getenv("OPENAI_API_KEY"):
            st.warning("⚠️ OpenAI API key not found. AI agents require an OpenAI API key to function.")
            st.info("Please add your OpenAI API key to the .env file to enable AI analysis.")
        else:
            # Single column layout for AI agent buttons
            if st.button("🧠 Analyze All Data", help="Get comprehensive insights across all datasets"):
                # Add user message to chat history
                user_msg = "🧠 Analyze All Data"
                st.session_state.messages.append({"role": "user", "content": user_msg})
                with st.chat_message("user"):
                    st.markdown(user_msg)
                
                # Generate and display response in chat
                with st.chat_message("assistant"):
                    with st.spinner("🤖 AI Agent analyzing all data..."):
                        analysis_result = app.llm_analyze_all_data()
                        st.markdown(analysis_result)
                
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": analysis_result})
                st.rerun()
            
            if st.button("🔍 Quality Assessment", help="Assess data quality across all datasets"):
                # Add user message to chat history
                user_msg = "🔍 Quality Assessment"
                st.session_state.messages.append({"role": "user", "content": user_msg})
                with st.chat_message("user"):
                    st.markdown(user_msg)
                
                # Generate and display response in chat
                with st.chat_message("assistant"):
                    with st.spinner("🤖 AI Agent checking data quality..."):
                        quality_result = app.llm_quality_check_all_data()
                        st.markdown(quality_result)
                
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": quality_result})
                st.rerun()
            
            if st.button("💡 Query Suggestions", help="Get intelligent query suggestions for all data"):
                # Add user message to chat history
                user_msg = "💡 Query Suggestions"
                st.session_state.messages.append({"role": "user", "content": user_msg})
                with st.chat_message("user"):
                    st.markdown(user_msg)
                
                # Generate and display response in chat
                with st.chat_message("assistant"):
                    with st.spinner("🤖 AI Agent generating query suggestions..."):
                        suggestions_result = app.llm_suggest_queries_all_data()
                        st.markdown(suggestions_result)
                
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": suggestions_result})
                st.rerun()
        
        st.markdown("---")

        # Display data preview
        with st.expander("📋 Data Preview"):
            # Show CSV datasets
            for filename, df in app.dfs.items():
                st.subheader(f"📊 {filename}")
                st.dataframe(df.head(100))
                st.write("---")
            
            # Show text documents
            for filename in app.text_files:
                st.subheader(f"📄 {filename}")
                if filename in app.text_contents:
                    text_content = app.text_contents[filename]
                    # Show preview (first 1000 characters)
                    preview_length = 1000
                    preview_text = text_content[:preview_length]
                    if len(text_content) > preview_length:
                        preview_text += "..."
                    
                    st.text(preview_text)
                    st.write(f"**Document Length:** {len(text_content)} characters | **Chunks:** {len(text_content) // 1000 + 1}")
                st.write("---")

        # Chat interface
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Chat input
        if prompt := st.chat_input("Ask a question about your data..."):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = app.query_data(prompt)
                    st.markdown(response)

                    # Generate visualization only if explicitly requested
                    viz_keywords = ['create visualization', 'create chart', 'create graph', 'show chart', 'show graph', 'plot', 'visualize', 'chart', 'graph']
                    if any(keyword in prompt.lower() for keyword in viz_keywords):
                        viz = app.generate_visualization(prompt)
                        if viz:
                            st.subheader("📊 Generated Visualization")
                            st.plotly_chart(viz, use_container_width=True)
                            st.caption(f"Chart type: {app.parse_visualization_intent(prompt)['chart_type']}")
                        else:
                            st.info("Could not generate visualization. Please check your data and try again.")

            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})

        # Clear chat button
        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.rerun()
        
        # Reset database button (for troubleshooting)
        if st.button("🔄 Reset Database (Troubleshoot)"):
            with st.spinner("Resetting database..."):
                app.clear_vectorstore()
                st.success("Database reset complete! You may need to re-upload your files.")
                st.rerun()

if __name__ == "__main__":
    main()
