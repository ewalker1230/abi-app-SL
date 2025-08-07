import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv
from openai import OpenAI
import chromadb
import plotly.express as px
from typing import List, Dict, Any, Optional
import json
import hashlib
import time
import uuid
from datetime import datetime

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

# Load environment variables
load_dotenv()

# Configure OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class QueryOptimizer:
    def __init__(self):
        self.cache = {}
        self.cache_ttl = {}
        self.performance_metrics = {}
        self.query_patterns = []
        self.optimization_history = []

    def preprocess_query(self, query: str) -> Dict[str, Any]:
        """Preprocess and analyze a query for optimization"""
        try:
            # Detect query intent
            intent = self._detect_query_intent(query)
            
            # Extract key terms
            key_terms = self._extract_key_terms(query)
            
            # Determine search strategy
            search_strategy = self._determine_search_strategy(intent, key_terms)
            
            # Estimate complexity
            complexity = self._estimate_complexity(query, intent)
            
            # Optimize query text
            optimized_query = self._optimize_query_text(query, key_terms)
            
            return {
                'original_query': query,
                'optimized_query': optimized_query,
                'intent': intent,
                'key_terms': key_terms,
                'search_strategy': search_strategy,
                'complexity': complexity
            }
        except Exception as e:
            # Return basic analysis if optimization fails
            return {
                'original_query': query,
                'optimized_query': query,
                'intent': {'type': 'general', 'confidence': 0.5},
                'key_terms': query.lower().split(),
                'search_strategy': 'semantic',
                'complexity': 'medium'
            }

    def _detect_query_intent(self, query: str) -> Dict[str, Any]:
        """Detect the intent of a query"""
        query_lower = query.lower()
        
        # Define intent patterns
        intent_patterns = {
            'aggregation': ['sum', 'total', 'average', 'mean', 'count', 'max', 'min', 'aggregate'],
            'comparison': ['compare', 'vs', 'versus', 'difference', 'higher', 'lower', 'better'],
            'trend': ['trend', 'over time', 'growth', 'decline', 'increase', 'decrease'],
            'filter': ['filter', 'where', 'only', 'just', 'specific', 'particular'],
            'sort': ['sort', 'order', 'rank', 'top', 'bottom', 'highest', 'lowest'],
            'visualization': ['chart', 'graph', 'plot', 'visualize', 'show', 'display']
        }
        
        # Check for intent patterns
        detected_intents = []
        for intent_type, patterns in intent_patterns.items():
            for pattern in patterns:
                if pattern in query_lower:
                    detected_intents.append(intent_type)
                    break
        
        # Default to general if no specific intent detected
        if not detected_intents:
            detected_intents = ['general']
        
        return {
            'type': detected_intents[0],
            'confidence': 0.8 if len(detected_intents) == 1 else 0.6,
            'all_intents': detected_intents
        }

    def _extract_key_terms(self, query: str) -> List[str]:
        """Extract key terms from a query"""
        # Simple key term extraction - split and filter
        words = query.lower().split()
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'}
        key_terms = [word for word in words if word not in stop_words and len(word) > 2]
        return key_terms

    def _determine_search_strategy(self, intent: Dict[str, Any], key_terms: List[str]) -> str:
        """Determine the best search strategy based on intent and key terms"""
        intent_type = intent.get('type', 'general')
        
        if intent_type in ['aggregation', 'comparison']:
            return 'hybrid'
        elif intent_type == 'filter':
            return 'exact_match'
        elif intent_type == 'trend':
            return 'semantic'
        else:
            return 'semantic'

    def _estimate_complexity(self, query: str, intent: Dict[str, Any]) -> str:
        """Estimate query complexity"""
        word_count = len(query.split())
        intent_type = intent.get('type', 'general')
        
        if word_count > 15 or intent_type in ['aggregation', 'comparison']:
            return 'high'
        elif word_count > 8:
            return 'medium'
        else:
            return 'low'

    def _optimize_query_text(self, query: str, key_terms: List[str]) -> str:
        """Optimize query text for better search results"""
        # Simple optimization - use key terms
        if key_terms:
            return ' '.join(key_terms)
        return query

    def get_adaptive_k(self, complexity: str, intent: Dict[str, Any]) -> int:
        """Get adaptive k value based on complexity and intent"""
        base_k = {
            'low': 5,
            'medium': 10,
            'high': 15
        }.get(complexity, 10)
        
        # Adjust based on intent
        intent_type = intent.get('type', 'general')
        if intent_type in ['aggregation', 'comparison']:
            base_k = min(base_k * 2, 20)
        elif intent_type == 'filter':
            base_k = max(base_k // 2, 3)
        
        return base_k

    def cache_query_result(self, query_hash: str, result: Any, ttl: int = 300):
        """Cache a query result"""
        self.cache[query_hash] = result
        self.cache_ttl[query_hash] = time.time() + ttl

    def get_cached_result(self, query_hash: str) -> Optional[Any]:
        """Get a cached query result"""
        if query_hash in self.cache:
            if time.time() < self.cache_ttl.get(query_hash, 0):
                return self.cache[query_hash]
            else:
                # Remove expired cache entry
                del self.cache[query_hash]
                del self.cache_ttl[query_hash]
        return None

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

class CSVChatApp:
    def __init__(self):
        self.dfs = {}  # Dictionary to store multiple dataframes
        self.excel_sheets = {}  # Dictionary to store Excel sheets for each file
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
        
        # Initialize text splitter (doesn't require API key)
        self.setup_langchain()

    def setup_langchain(self):
        """Initialize LangChain components for vector storage"""
        try:
            # Ensure chroma_db directory exists
            import os
            os.makedirs("./chroma_db", exist_ok=True)
            
            # Initialize text splitter first (doesn't require API key)
            if self.text_splitter is None:
                self.text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=500,
                    chunk_overlap=50,
                    length_function=len,
                    separators=["\n\n", "\n", " ", ""],
                    keep_separator=True
                )
            
            # Initialize embeddings (requires API key)
            if self.embeddings is None:
                try:
                    self.embeddings = OpenAIEmbeddings()
                except Exception as embed_error:
                    st.error(f"Error initializing embeddings: {str(embed_error)}")
                    st.error("Please check your OpenAI API key in the .env file")
                    # Don't return here - continue without embeddings
            
            # Initialize vectorstore if embeddings are available
            if self.vectorstore is None and self.embeddings is not None:
                try:
                    self.vectorstore = Chroma(
                        persist_directory="./chroma_db",
                        embedding_function=self.embeddings,
                        collection_name="csv_data"
                    )
                except Exception as db_error:
                    # Handle read-only database errors gracefully
                    if "readonly" in str(db_error).lower() or "code: 1032" in str(db_error):
                        st.warning("Database is read-only. Clearing and recreating...")
                        # Clear the corrupted database
                        import shutil
                        if os.path.exists("./chroma_db"):
                            try:
                                shutil.rmtree("./chroma_db")
                            except:
                                pass
                        os.makedirs("./chroma_db", exist_ok=True)
                        
                        # Try again with fresh database
                        try:
                            self.vectorstore = Chroma(
                                persist_directory="./chroma_db",
                                embedding_function=self.embeddings,
                                collection_name="csv_data"
                            )
                            st.success("✅ Database recreated successfully!")
                        except Exception as recreate_error:
                            st.error(f"❌ Could not recreate database: {str(recreate_error)}")
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

    def llm_analyze_all_data(self):
        """LLM-powered agent that intelligently analyzes all uploaded data"""
        try:
            session_id = self.session_manager.get_or_create_session_id()
            
            # Get all data summaries
            data_summaries = []
            
            # Add CSV datasets
            for filename, df in self.dfs.items():
                data_summaries.append(f"📊 {filename}: {df.shape[0]} rows, {df.shape[1]} columns")
                data_summaries.append(f"   Columns: {list(df.columns)}")
                data_summaries.append(f"   Sample data: {df.head(3).to_string()}")
                data_summaries.append("")
            
            # Add text documents
            for filename in self.text_files:
                if filename in self.text_contents:
                    text_content = self.text_contents[filename]
                    data_summaries.append(f"📄 {filename}: {len(text_content)} characters")
                    data_summaries.append(f"   Preview: {text_content[:200]}...")
                    data_summaries.append("")
            
            if not data_summaries:
                return "No data found to analyze. Please upload some files first."
            
            # Create comprehensive analysis prompt
            data_summary_text = '\n'.join(data_summaries)
            analysis_prompt = f"""
            Analyze the following uploaded data and provide comprehensive insights:
            
            {data_summary_text}
            
            Please provide:
            1. **Data Overview**: Summary of what data is available
            2. **Key Patterns**: Notable patterns or trends in the data
            3. **Data Quality**: Assessment of data quality and completeness
            4. **Business Insights**: Potential business value and insights
            5. **Recommendations**: Suggested next steps for analysis
            
            Focus on actionable insights that would be valuable for business decision-making.
            """
            
            # Get response from LLM
            response = self.query_data(analysis_prompt)
            
            # Save to session
            self.session_manager.save_conversation_turn(
                session_id=session_id,
                user_query="🧠 Analyze All Data",
                assistant_response=response,
                metadata={"analysis_type": "comprehensive_data_analysis"}
            )
            
            return response
            
        except Exception as e:
            error_msg = f"Error analyzing data: {str(e)}"
            st.error(error_msg)
            return error_msg

    def llm_quality_check_all_data(self):
        """LLM-powered agent that checks data quality across all datasets"""
        try:
            session_id = self.session_manager.get_or_create_session_id()
            
            # Get all data summaries
            data_summaries = []
            
            # Add CSV datasets
            for filename, df in self.dfs.items():
                data_summaries.append(f"📊 {filename}: {df.shape[0]} rows, {df.shape[1]} columns")
                data_summaries.append(f"   Columns: {list(df.columns)}")
                data_summaries.append(f"   Missing values: {df.isnull().sum().sum()}")
                data_summaries.append(f"   Duplicates: {df.duplicated().sum()}")
                data_summaries.append("")
            
            # Add text documents
            for filename in self.text_files:
                if filename in self.text_contents:
                    text_content = self.text_contents[filename]
                    data_summaries.append(f"📄 {filename}: {len(text_content)} characters")
                    data_summaries.append(f"   Word count: {len(text_content.split())}")
                    data_summaries.append("")
            
            if not data_summaries:
                return "No data found to check. Please upload some files first."
            
            # Create quality check prompt
            data_summary_text = '\n'.join(data_summaries)
            quality_prompt = f"""
            Perform a comprehensive data quality assessment on the following data:
            
            {data_summary_text}
            
            Please provide:
            1. **Data Completeness**: Assessment of missing values and gaps
            2. **Data Consistency**: Check for inconsistencies and anomalies
            3. **Data Accuracy**: Identify potential accuracy issues
            4. **Data Validity**: Check if data meets expected formats and ranges
            5. **Recommendations**: Suggestions for data cleaning and improvement
            
            Focus on identifying data quality issues that could impact analysis reliability.
            """
            
            # Get response from LLM
            response = self.query_data(quality_prompt)
            
            # Save to session
            self.session_manager.save_conversation_turn(
                session_id=session_id,
                user_query="🔍 Quality Assessment",
                assistant_response=response,
                metadata={"analysis_type": "data_quality_assessment"}
            )
            
            return response
            
        except Exception as e:
            error_msg = f"Error checking data quality: {str(e)}"
            st.error(error_msg)
            return error_msg

    def llm_suggest_queries_all_data(self):
        """LLM-powered agent that suggests intelligent queries for all data"""
        try:
            session_id = self.session_manager.get_or_create_session_id()
            
            # Get all data summaries
            data_summaries = []
            
            # Add CSV datasets
            for filename, df in self.dfs.items():
                data_summaries.append(f"📊 {filename}: {df.shape[0]} rows, {df.shape[1]} columns")
                data_summaries.append(f"   Columns: {list(df.columns)}")
                data_summaries.append("")
            
            # Add text documents
            for filename in self.text_files:
                if filename in self.text_contents:
                    text_content = self.text_contents[filename]
                    data_summaries.append(f"📄 {filename}: {len(text_content)} characters")
                    data_summaries.append("")
            
            if not data_summaries:
                return "No data found to suggest queries for. Please upload some files first."
            
            # Create query suggestion prompt
            data_summary_text = '\n'.join(data_summaries)
            suggestion_prompt = f"""
            Based on the following data, suggest 10 intelligent and useful queries that users could ask:
            
            {data_summary_text}
            
            Please provide:
            1. **Exploratory Queries**: Questions to understand the data better
            2. **Analytical Queries**: Questions for deeper analysis and insights
            3. **Business Queries**: Questions relevant to business decision-making
            4. **Comparative Queries**: Questions that compare different aspects
            5. **Trend Queries**: Questions about patterns and trends over time
            
            Format each query as a clear, natural language question that users can ask directly.
            """
            
            # Get response from LLM
            response = self.query_data(suggestion_prompt)
            
            # Save to session
            self.session_manager.save_conversation_turn(
                session_id=session_id,
                user_query="💡 Query Suggestions",
                assistant_response=response,
                metadata={"analysis_type": "query_suggestions"}
            )
            
            return response
            
        except Exception as e:
            error_msg = f"Error generating query suggestions: {str(e)}"
            st.error(error_msg)
            return error_msg

    def process_csv(self, uploaded_file, filename: str) -> pd.DataFrame:
        """Process uploaded CSV file and return DataFrame"""
        try:
            # Read CSV file
            df = pd.read_csv(uploaded_file)
            
            # Store in memory
            self.dfs[filename] = df
            
            # Process with LangChain for vector storage
            self.process_with_langchain(uploaded_file, filename, "csv")
            
            # Track the uploaded file
            self.processed_files.add(filename)
            
            return df
            
        except Exception as e:
            st.error(f"Error processing CSV file {filename}: {str(e)}")
            return None

    def process_excel(self, uploaded_file, filename: str):
        """Process uploaded Excel file and return DataFrame"""
        try:
            # Read Excel file
            df = pd.read_excel(uploaded_file, sheet_name=None)
            
            # If multiple sheets, process each one
            if isinstance(df, dict):
                processed_sheets = {}
                for sheet_name, sheet_df in df.items():
                    sheet_filename = f"{filename}_{sheet_name}"
                    processed_sheets[sheet_name] = sheet_df
                    
                    # Process each sheet with LangChain
                    self.process_excel_sheet_with_langchain(sheet_df, sheet_filename)
                    
                    # Track the processed sheet
                    self.processed_files.add(sheet_filename)
                
                # Store all sheets in excel_sheets
                self.excel_sheets[filename] = processed_sheets
                
                # Store the first sheet as the main dataframe
                first_sheet_name = list(df.keys())[0]
                self.dfs[filename] = df[first_sheet_name]
                
                return processed_sheets
            else:
                # Single sheet
                self.dfs[filename] = df
                
                # Store as single sheet in excel_sheets
                self.excel_sheets[filename] = {filename: df}
                
                # Process with LangChain
                self.process_excel_sheet_with_langchain(df, filename)
                
                # Track the uploaded file
                self.processed_files.add(filename)
                
                return df
                
        except Exception as e:
            st.error(f"Error processing Excel file {filename}: {str(e)}")
            return None

    def process_with_langchain(self, uploaded_file, filename: str, file_type: str):
        """Process uploaded file with LangChain for vector storage"""
        try:
            # Reset file pointer
            uploaded_file.seek(0)
            
            # Create temporary file for processing
            import tempfile
            import os
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_type}") as temp_file:
                temp_path = temp_file.name
                temp_file.write(uploaded_file.read())
            
            # Read the file content
            if file_type == "csv":
                df = pd.read_csv(temp_path)
            elif file_type in ["xlsx", "xls"]:
                df = pd.read_excel(temp_path)
            else:
                st.error(f"Unsupported file type: {file_type}")
                os.remove(temp_path)
                return False
            
            # Convert dataframe to documents
            documents = []
            for idx, row in df.iterrows():
                # Create document from row
                row_text = " ".join([f"{col}: {val}" for col, val in row.items()])
                
                # Only add documents with meaningful content
                if row_text and row_text.strip():
                    doc = type('Document', (), {
                        'page_content': row_text,
                        'metadata': {
                            'filename': filename,
                            'row_index': idx,
                            'file_type': file_type,
                            'content_type': 'data_row'
                        }
                    })()
                    documents.append(doc)
            
            # Check if we have any documents to process
            if not documents:
                st.warning(f"No valid content found in {filename} to process")
                os.remove(temp_path)
                return False
            
            # Ensure text_splitter is initialized
            if self.text_splitter is None:
                self.setup_langchain()
                if self.text_splitter is None:
                    st.error("Text splitter not initialized. Please check your OpenAI API key.")
                    os.remove(temp_path)
                    return False
            
            # Split documents
            chunks = self.text_splitter.split_documents(documents)
            
            # Filter out empty chunks
            valid_chunks = []
            for chunk in chunks:
                if chunk.page_content and chunk.page_content.strip():
                    valid_chunks.append(chunk)
            
            if not valid_chunks:
                st.warning(f"No valid content found in {filename} after processing")
                os.remove(temp_path)
                return False
            
            # Debug information
            st.info(f"Original documents: {len(documents)}, Chunks after splitting: {len(chunks)}, Valid chunks: {len(valid_chunks)}")
            
            # Verify embeddings are properly initialized
            if not self.embeddings:
                st.error("Embeddings not properly initialized. Please check your OpenAI API key.")
                os.remove(temp_path)
                return False
            
            # Add to vectorstore
            try:
                self.vectorstore.add_documents(valid_chunks)
            except Exception as embedding_error:
                error_msg = str(embedding_error).lower()
                if "readonly" in error_msg or "code: 1032" in error_msg or "attempt to write a readonly database" in error_msg:
                    st.warning("Database is read-only. Clearing and recreating...")
                    # Clear the corrupted database
                    import shutil
                    if os.path.exists("./chroma_db"):
                        try:
                            shutil.rmtree("./chroma_db")
                        except:
                            pass
                    os.makedirs("./chroma_db", exist_ok=True)
                    
                    # Recreate vectorstore
                    try:
                        self.vectorstore = Chroma(
                            persist_directory="./chroma_db",
                            embedding_function=self.embeddings,
                            collection_name="csv_data"
                        )
                        # Try adding documents again
                        self.vectorstore.add_documents(valid_chunks)
                        st.success("Database recreated and documents added successfully!")
                    except Exception as recreate_error:
                        st.error(f"Could not recreate database: {str(recreate_error)}")
                        os.remove(temp_path)
                        return False
                else:
                    st.error(f"Error adding documents to vectorstore: {str(embedding_error)}")
                    st.error("This might be due to empty embeddings. Check if your OpenAI API key is valid and the content is meaningful.")
                    os.remove(temp_path)
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
                self.setup_langchain()
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
                error_msg = str(embedding_error).lower()
                if "readonly" in error_msg or "code: 1032" in error_msg or "attempt to write a readonly database" in error_msg:
                    st.warning("Database is read-only. Clearing and recreating...")
                    # Clear the corrupted database
                    import shutil
                    if os.path.exists("./chroma_db"):
                        try:
                            shutil.rmtree("./chroma_db")
                        except:
                            pass
                    os.makedirs("./chroma_db", exist_ok=True)
                    
                    # Recreate vectorstore
                    try:
                        self.vectorstore = Chroma(
                            persist_directory="./chroma_db",
                            embedding_function=self.embeddings,
                            collection_name="csv_data"
                        )
                        # Try adding documents again
                        self.vectorstore.add_documents(valid_chunks)
                        st.success("Database recreated and documents added successfully!")
                    except Exception as recreate_error:
                        st.error(f"Could not recreate database: {str(recreate_error)}")
                        return
                else:
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
                self.setup_langchain()
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
                error_msg = str(embedding_error).lower()
                if "readonly" in error_msg or "code: 1032" in error_msg or "attempt to write a readonly database" in error_msg:
                    st.warning("Database is read-only. Clearing and recreating...")
                    # Clear the corrupted database
                    import shutil
                    if os.path.exists("./chroma_db"):
                        try:
                            shutil.rmtree("./chroma_db")
                        except:
                            pass
                    os.makedirs("./chroma_db", exist_ok=True)
                    
                    # Recreate vectorstore
                    try:
                        self.vectorstore = Chroma(
                            persist_directory="./chroma_db",
                            embedding_function=self.embeddings,
                            collection_name="csv_data"
                        )
                        # Try adding documents again
                        self.vectorstore.add_documents(valid_chunks)
                        st.success("Database recreated and documents added successfully!")
                    except Exception as recreate_error:
                        st.error(f"Could not recreate database: {str(recreate_error)}")
                        return
                else:
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
            
            return True
            
        except Exception as e:
            st.error(f"Error processing text file {filename}: {str(e)}")
            return False

    def query_data(self, user_query: str) -> str:
        """Query the vector database and return a response"""
        try:
            session_id = self.session_manager.get_or_create_session_id()
            
            # Check cache first
            query_hash = hashlib.md5(user_query.encode()).hexdigest()
            cached_result = self.query_optimizer.get_cached_result(query_hash)
            if cached_result:
                return cached_result
            
            # Preprocess query
            query_analysis = self.query_optimizer.preprocess_query(user_query)
            
            # Get adaptive k value
            k = self.query_optimizer.get_adaptive_k(query_analysis['complexity'], query_analysis['intent'])
            
            # Search vectorstore
            if self.vectorstore:
                try:
                    # Use similarity search
                    docs = self.vectorstore.similarity_search(user_query, k=k)
                    
                    # Build context from documents
                    context_parts = []
                    for doc in docs:
                        context_parts.append(doc.page_content)
                    
                    context = "\n\n".join(context_parts)
                    
                    # Generate response using OpenAI
                    response = self._generate_llm_response(user_query, context)
                    
                    # Cache the result
                    self.query_optimizer.cache_query_result(query_hash, response)
                    
                    # Save to session
                    self.session_manager.save_conversation_turn(
                        session_id=session_id,
                        user_query=user_query,
                        assistant_response=response
                    )
                    
                    return response
                    
                except Exception as search_error:
                    st.error(f"Error searching vector database: {str(search_error)}")
                    return f"I encountered an error while searching the data: {str(search_error)}"
            else:
                return "No data has been loaded yet. Please upload some files first."
                
        except Exception as e:
            error_msg = f"Error querying data: {str(e)}"
            st.error(error_msg)
            return error_msg

    def _generate_llm_response(self, user_query: str, context: str) -> str:
        """Generate LLM response based on user query and context"""
        try:
            # Create prompt
            prompt = f"""
            Based on the following context, please answer the user's question. 
            If the answer cannot be found in the context, say so clearly.
            
            Context:
            {context}
            
            User Question: {user_query}
            
            Please provide a clear, helpful, and accurate response based on the context provided.
            """
            
            # Use OpenAI to generate response
            from openai import OpenAI
            client = OpenAI()
            
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful AI assistant that analyzes data and provides insights. Always be accurate and helpful."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.3
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"Error generating response: {str(e)}"

    def generate_visualization(self, query: str):
        """Generate visualization based on query"""
        try:
            # Parse visualization intent
            intent = self.parse_visualization_intent(query)
            
            if not intent:
                return None
            
            # Generate chart based on intent
            chart_type = intent.get('chart_type', 'bar')
            
            if chart_type == 'bar':
                return self.create_bar_chart(intent)
            elif chart_type == 'line':
                return self.create_line_chart(intent)
            elif chart_type == 'scatter':
                return self.create_scatter_chart(intent)
            elif chart_type == 'pie':
                return self.create_pie_chart(intent)
            elif chart_type == 'histogram':
                return self.create_histogram_chart(intent)
            elif chart_type == 'box':
                return self.create_box_chart(intent)
            else:
                return self.create_bar_chart(intent)  # Default to bar chart
                
        except Exception as e:
            st.error(f"Error generating visualization: {str(e)}")
            return None

    def parse_visualization_intent(self, query: str):
        """Parse visualization intent from query"""
        try:
            query_lower = query.lower()
            
            # Detect chart type
            chart_type = 'bar'  # Default
            
            if any(word in query_lower for word in ['line', 'trend', 'over time']):
                chart_type = 'line'
            elif any(word in query_lower for word in ['scatter', 'correlation']):
                chart_type = 'scatter'
            elif any(word in query_lower for word in ['pie', 'percentage', 'proportion']):
                chart_type = 'pie'
            elif any(word in query_lower for word in ['histogram', 'distribution']):
                chart_type = 'histogram'
            elif any(word in query_lower for word in ['box', 'boxplot']):
                chart_type = 'box'
            
            return {
                'chart_type': chart_type,
                'query': query
            }
            
        except Exception as e:
            return None

    def create_bar_chart(self, intent):
        """Create a bar chart"""
        try:
            # This is a placeholder - you would implement actual chart creation
            import plotly.express as px
            import plotly.graph_objects as go
            
            # Create sample data for demonstration
            data = {'Category': ['A', 'B', 'C', 'D'], 'Value': [10, 20, 15, 25]}
            df = pd.DataFrame(data)
            
            fig = px.bar(df, x='Category', y='Value', title='Sample Bar Chart')
            return fig
            
        except Exception as e:
            st.error(f"Error creating bar chart: {str(e)}")
            return None

    def create_line_chart(self, intent):
        """Create a line chart"""
        try:
            import plotly.express as px
            
            # Create sample data for demonstration
            data = {'Date': pd.date_range('2023-01-01', periods=10), 'Value': range(10)}
            df = pd.DataFrame(data)
            
            fig = px.line(df, x='Date', y='Value', title='Sample Line Chart')
            return fig
            
        except Exception as e:
            st.error(f"Error creating line chart: {str(e)}")
            return None

    def create_scatter_chart(self, intent):
        """Create a scatter chart"""
        try:
            import plotly.express as px
            
            # Create sample data for demonstration
            data = {'X': range(10), 'Y': [i**2 for i in range(10)]}
            df = pd.DataFrame(data)
            
            fig = px.scatter(df, x='X', y='Y', title='Sample Scatter Chart')
            return fig
            
        except Exception as e:
            st.error(f"Error creating scatter chart: {str(e)}")
            return None

    def create_pie_chart(self, intent):
        """Create a pie chart"""
        try:
            import plotly.express as px
            
            # Create sample data for demonstration
            data = {'Category': ['A', 'B', 'C', 'D'], 'Value': [30, 25, 20, 25]}
            df = pd.DataFrame(data)
            
            fig = px.pie(df, values='Value', names='Category', title='Sample Pie Chart')
            return fig
            
        except Exception as e:
            st.error(f"Error creating pie chart: {str(e)}")
            return None

    def create_histogram_chart(self, intent):
        """Create a histogram chart"""
        try:
            import plotly.express as px
            import numpy as np
            
            # Create sample data for demonstration
            data = np.random.normal(0, 1, 1000)
            df = pd.DataFrame({'Value': data})
            
            fig = px.histogram(df, x='Value', title='Sample Histogram')
            return fig
            
        except Exception as e:
            st.error(f"Error creating histogram chart: {str(e)}")
            return None

    def create_box_chart(self, intent):
        """Create a box chart"""
        try:
            import plotly.express as px
            import numpy as np
            
            # Create sample data for demonstration
            data = {'Group': ['A']*50 + ['B']*50, 'Value': list(np.random.normal(0, 1, 50)) + list(np.random.normal(2, 1, 50))}
            df = pd.DataFrame(data)
            
            fig = px.box(df, x='Group', y='Value', title='Sample Box Chart')
            return fig
            
        except Exception as e:
            st.error(f"Error creating box chart: {str(e)}")
            return None

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
        
        # Show loaded datasets (collapsible, not auto-expanded)
        if app.dfs or app.text_files:
            with st.expander("📊 View Data", expanded=False):
                st.header("📊 Loaded Datasets")
                
                # Add button to clear all data
                if st.button("🗑️ Clear All Data"):
                    app.dfs.clear()
                    app.excel_sheets.clear()  # Clear Excel sheets
                    app.text_files.clear()
                    app.text_contents.clear()
                    app.processed_files.clear()  # Clear processed files tracking
                    
                    # Clear the vectorstore
                    app.clear_vectorstore()
                    
                    st.success("All data cleared!")
                    st.rerun()
                
                # Show CSV datasets and Excel files
                for filename, df in app.dfs.items():
                    # Check if this is an Excel file with multiple sheets
                    if filename in app.excel_sheets and len(app.excel_sheets[filename]) > 1:
                        # This is an Excel file with multiple sheets
                        with st.expander(f"📊 {filename} (Excel - {len(app.excel_sheets[filename])} sheets)"):
                            # Show all sheets
                            for sheet_name, sheet_df in app.excel_sheets[filename].items():
                                st.write(f"**Sheet: {sheet_name}**")
                                st.dataframe(sheet_df.head(10000))
                                st.write("---")
                            
                            # Add remove button for the entire Excel file
                            if st.button(f"❌ Remove {filename}", key=f"remove_excel_{filename}"):
                                # Remove from memory
                                del app.dfs[filename]
                                if filename in app.excel_sheets:
                                    del app.excel_sheets[filename]
                                
                                # Remove from processed files tracking
                                for processed_file in list(app.processed_files):
                                    if processed_file.startswith(filename):
                                        app.processed_files.remove(processed_file)
                                
                                # Note: LangChain Chroma doesn't support direct deletion by metadata
                                # The document will remain in the vectorstore but won't be accessible
                                # through the app interface
                                st.info(f"Removed {filename} from memory. Note: Vector embeddings remain in database.")
                                
                                st.rerun()
                    else:
                        # This is a CSV or single-sheet Excel file
                        with st.expander(f"📊 {filename} ({len(df)} rows)"):
                            st.dataframe(df.head(10000))
                            
                            # Add remove button for individual datasets
                            if st.button(f"❌ Remove {filename}", key=f"remove_csv_{filename}"):
                                # Remove from memory
                                del app.dfs[filename]
                                if filename in app.excel_sheets:
                                    del app.excel_sheets[filename]
                                
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
                        elif uploaded_file.name.lower().endswith(('.xlsx', '.xls')):
                            df = app.process_excel(uploaded_file, uploaded_file.name)
                            if df is not None:
                                st.success(f"✅ {uploaded_file.name} processed successfully!")
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

        # Display data preview (collapsible, not auto-expanded)
        with st.expander("📋 Data Preview", expanded=False):
            # Show CSV datasets and Excel files
            for filename, df in app.dfs.items():
                # Check if this is an Excel file with multiple sheets
                if filename in app.excel_sheets and len(app.excel_sheets[filename]) > 1:
                    # This is an Excel file with multiple sheets
                    st.subheader(f"📊 {filename} (Excel - {len(app.excel_sheets[filename])} sheets)")
                    
                    # Show all sheets
                    for sheet_name, sheet_df in app.excel_sheets[filename].items():
                        st.write(f"**Sheet: {sheet_name}**")
                        st.dataframe(sheet_df.head(100))
                        st.write("---")
                else:
                    # This is a CSV or single-sheet Excel file
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
