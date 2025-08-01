import streamlit as st
import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv
from openai import OpenAI
import chromadb
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict, Any
import json
import psycopg2
from psycopg2.extras import RealDictCursor

# Load environment variables
load_dotenv()

# Configure OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


class PostgresVectorCSVChatApp:
    def __init__(self):
        self.df = None
        self.pg_conn = None
        self.setup_postgres()

    def setup_postgres(self):
        """Initialize PostgreSQL connection with pgvector"""
        try:
            self.pg_conn = psycopg2.connect(
                host=os.getenv("POSTGRES_HOST", "localhost"),
                database=os.getenv("POSTGRES_DB", "csv_chat"),
                user=os.getenv("POSTGRES_USER", "postgres"),
                password=os.getenv("POSTGRES_PASSWORD", "password"),
                port=os.getenv("POSTGRES_PORT", "5432")
            )

            # Enable pgvector extension
            with self.pg_conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                self.pg_conn.commit()

        except Exception as e:
            st.error(f"PostgreSQL connection failed: {str(e)}")
            st.info("Make sure PostgreSQL is running and pgvector extension is installed")

    def process_csv(self, uploaded_file) -> pd.DataFrame:
        """Process uploaded CSV file and store in PostgreSQL"""
        try:
            df = pd.read_csv(uploaded_file)
            self.df = df
            self.store_data_in_postgres(df)
            return df
        except Exception as e:
            st.error(f"Error processing CSV: {str(e)}")
            return None

    def store_data_in_postgres(self, df: pd.DataFrame):
        """Store CSV data in PostgreSQL with vector embeddings"""
        if self.pg_conn is None:
            return

        # Create table for CSV data
        columns = []
        for col in df.columns:
            # Determine column type based on data
            if df[col].dtype in ['int64', 'float64']:
                columns.append(f"{col} NUMERIC")
            else:
                columns.append(f"{col} TEXT")

        create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS csv_data (
            id SERIAL PRIMARY KEY,
            {', '.join(columns)},
            embedding vector(1536)  -- OpenAI embedding dimension
        )
        """

        with self.pg_conn.cursor() as cur:
            cur.execute(create_table_sql)
            self.pg_conn.commit()

        # Generate embeddings and store data
        self.generate_and_store_embeddings(df)

    def generate_and_store_embeddings(self, df: pd.DataFrame):
        """Generate embeddings for each row and store in PostgreSQL"""
        if self.pg_conn is None:
            return

        # Clear existing data
        with self.pg_conn.cursor() as cur:
            cur.execute("DELETE FROM csv_data")
            self.pg_conn.commit()

        # Generate embeddings for each row
        for idx, row in df.iterrows():
            # Create text representation of the row
            row_text = " ".join([f"{col}: {val}" for col, val in row.items()])

            # Generate embedding using OpenAI
            try:
                embedding_response = client.embeddings.create(input=row_text,
                model="text-embedding-ada-002")
                embedding = embedding_response.data[0].embedding

                # Insert data with embedding
                columns = list(df.columns)
                placeholders = ', '.join(['%s'] * len(columns))
                insert_sql = f"""
                INSERT INTO csv_data ({', '.join(columns)}, embedding)
                VALUES ({placeholders}, %s)
                """

                values = list(row.values) + [embedding]

                with self.pg_conn.cursor() as cur:
                    cur.execute(insert_sql, values)

            except Exception as e:
                st.warning(f"Could not generate embedding for row {idx}: {str(e)}")

        self.pg_conn.commit()
        st.success(f"Stored {len(df)} rows with embeddings in PostgreSQL")

    def query_data(self, user_query: str) -> str:
        """Query the data using OpenAI and PostgreSQL vector search"""
        if self.df is None or self.pg_conn is None:
            return "Please upload a CSV file first."

        # Generate embedding for user query
        try:
            query_embedding_response = client.embeddings.create(input=user_query,
            model="text-embedding-ada-002")
            query_embedding = query_embedding_response.data[0].embedding

            # Search for similar vectors using cosine similarity
            search_sql = """
            SELECT *, 
                   1 - (embedding <=> %s) as similarity
            FROM csv_data 
            ORDER BY embedding <=> %s
            LIMIT 10
            """

            with self.pg_conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(search_sql, (query_embedding, query_embedding))
                results = cur.fetchall()

            # Convert results to DataFrame
            relevant_data = []
            for row in results:
                row_dict = dict(row)
                # Remove embedding and similarity from display
                row_dict.pop('embedding', None)
                row_dict.pop('similarity', None)
                relevant_data.append(row_dict)

            relevant_df = pd.DataFrame(relevant_data)

        except Exception as e:
            st.error(f"Error in vector search: {str(e)}")
            relevant_df = self.df.head(5)  # Fallback to first 5 rows

        # Create context for OpenAI
        context = f"""
        CSV Data Schema:
        Columns: {list(self.df.columns)}
        Total Rows: {len(self.df)}
        
        Sample Data (first 5 rows):
        {self.df.head().to_string()}
        
        Relevant Data for Query (vector similarity search):
        {relevant_df.to_string() if not relevant_df.empty else "No relevant data found"}
        
        User Query: {user_query}
        
        Please provide a helpful response based on this data. If the query asks for analysis, 
        provide insights. If it asks for specific data, provide the relevant information.
        Be conversational and helpful.
        """

        try:
            response = client.chat.completions.create(model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful data analyst assistant. You help users understand and analyze CSV data."},
                {"role": "user", "content": context}
            ],
            max_tokens=500,
            temperature=0.7)
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating response: {str(e)}"

    def run_complex_query(self, query: str):
        """Run complex SQL queries on the data"""
        if self.pg_conn is None:
            return None

        try:
            with self.pg_conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query)
                results = cur.fetchall()
                return pd.DataFrame(results)
        except Exception as e:
            st.error(f"SQL Query Error: {str(e)}")
            return None

    def generate_visualization(self, query: str):
        """Generate visualizations based on user query"""
        if self.df is None:
            return None

        # Simple visualization logic based on query keywords
        query_lower = query.lower()

        if "chart" in query_lower or "graph" in query_lower or "plot" in query_lower:
            # Try to create a basic chart
            numeric_cols = self.df.select_dtypes(include=['number']).columns

            if len(numeric_cols) >= 2:
                fig = px.scatter(self.df, x=numeric_cols[0], y=numeric_cols[1], 
                               title=f"{numeric_cols[0]} vs {numeric_cols[1]}")
                return fig
            elif len(numeric_cols) == 1:
                fig = px.histogram(self.df, x=numeric_cols[0], 
                                 title=f"Distribution of {numeric_cols[0]}")
                return fig

        return None

def main():
    st.set_page_config(
        page_title="ABI - PostgreSQL Vector Edition",
        page_icon="🐘",
        layout="wide"
    )

    st.title("🐘 ABI - PostgreSQL Vector Edition")
    st.markdown("Upload a CSV file and chat with your data using PostgreSQL + pgvector!")

    # Initialize app
    if 'postgres_app' not in st.session_state:
        st.session_state.postgres_app = PostgresVectorCSVChatApp()

    app = st.session_state.postgres_app

    # Sidebar for file upload
    with st.sidebar:
        st.header("📁 Upload Data")
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help="Upload your CSV file to start chatting with it"
        )

        if uploaded_file is not None:
            if app.df is None or st.button("Reload Data"):
                with st.spinner("Processing CSV and generating embeddings..."):
                    df = app.process_csv(uploaded_file)
                    if df is not None:
                        st.success("CSV processed and stored in PostgreSQL!")
                        st.write(f"**Shape:** {df.shape}")
                        st.write(f"**Columns:** {list(df.columns)}")

        # Database connection status
        st.header("🔗 Database Status")
        if app.pg_conn:
            st.success("✅ PostgreSQL Connected")
        else:
            st.error("❌ PostgreSQL Not Connected")

        # Configuration info
        st.header("🔑 Configuration")
        if os.getenv("OPENAI_API_KEY"):
            st.success("✅ OpenAI API Key configured")
        else:
            st.error("❌ OpenAI API Key not found in .env file")
            st.info("Please add OPENAI_API_KEY to your .env file")

    # Main chat interface
    if app.df is not None:
        st.header("💬 Chat with Your Data")

        # Display data preview
        with st.expander("📋 Data Preview"):
            st.dataframe(app.df.head(10))

        # Chat interface
        if "postgres_messages" not in st.session_state:
            st.session_state.postgres_messages = []

        # Display chat history
        for message in st.session_state.postgres_messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Chat input
        if prompt := st.chat_input("Ask a question about your data..."):
            # Add user message to chat history
            st.session_state.postgres_messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("Searching PostgreSQL vectors..."):
                    response = app.query_data(prompt)
                    st.markdown(response)

                    # Generate visualization if appropriate
                    viz = app.generate_visualization(prompt)
                    if viz:
                        st.plotly_chart(viz, use_container_width=True)

            # Add assistant response to chat history
            st.session_state.postgres_messages.append({"role": "assistant", "content": response})

        # Clear chat button
        if st.button("Clear Chat"):
            st.session_state.postgres_messages = []
            st.rerun()

    else:
        st.info("👆 Please upload a CSV file in the sidebar to get started!")

        # PostgreSQL advantages
        st.header("🐘 Why PostgreSQL + pgvector?")
        st.markdown("""
        **Advantages over ChromaDB:**
        - ✅ **Unified Database**: One system for structured data AND vectors
        - ✅ **ACID Compliance**: Transactional vector operations
        - ✅ **SQL Integration**: Use familiar SQL for complex queries
        - ✅ **Production Ready**: Enterprise-grade reliability
        - ✅ **Cost Effective**: No separate vector database needed
        
        **Example PostgreSQL vector queries:**
        ```sql
        -- Find similar data using cosine similarity
        SELECT *, 1 - (embedding <=> '[0.1, 0.2, ...]') as similarity
        FROM csv_data 
        ORDER BY embedding <=> '[0.1, 0.2, ...]'
        LIMIT 10;
        
        -- Combine vector search with SQL analysis
        SELECT category, AVG(sales) as avg_sales
        FROM csv_data 
        WHERE 1 - (embedding <=> '[0.1, 0.2, ...]') > 0.8
        GROUP BY category;
        ```
        """)

if __name__ == "__main__":
    main() 