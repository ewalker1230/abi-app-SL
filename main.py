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

class CSVChatApp:
    def __init__(self):
        self.df = None
        self.chroma_client = None
        self.collection = None
        self.chat_history = []

    def setup_chroma(self):
        """Initialize ChromaDB for vector storage"""
        try:
            if self.chroma_client is None:
                # Use persistent storage to avoid context leaks
                self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
                
            # Try to get existing collection, create if it doesn't exist
            try:
                self.collection = self.chroma_client.get_collection(name="csv_data")
            except:
                self.collection = self.chroma_client.create_collection(
                    name="csv_data",
                    metadata={"description": "CSV data for semantic search"}
                )
        except Exception as e:
            st.error(f"Error setting up ChromaDB: {str(e)}")
            self.chroma_client = None
            self.collection = None

    def process_csv(self, uploaded_file) -> pd.DataFrame:
        """Process uploaded CSV file"""
        try:
            df = pd.read_csv(uploaded_file)
            self.df = df
            self.setup_chroma()
            self.index_data()
            return df
        except Exception as e:
            st.error(f"Error processing CSV: {str(e)}")
            return None

    def index_data(self):
        """Index CSV data for semantic search"""
        if self.df is None:
            return

        if self.collection is None:
            st.error("ChromaDB collection not initialized")
            return

        try:
            # Create text representations of each row
            documents = []
            metadatas = []
            ids = []

            for idx, row in self.df.iterrows():
                # Create a text representation of the row
                row_text = " ".join([f"{col}: {val}" for col, val in row.items()])
                documents.append(row_text)
                metadatas.append({"row_index": idx})
                ids.append(f"row_{idx}")

            # Add to ChromaDB
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            st.success(f"Indexed {len(documents)} rows in ChromaDB")
        except Exception as e:
            st.error(f"Error indexing data: {str(e)}")

    def query_data(self, user_query: str) -> str:
        """Query the data using OpenAI and ChromaDB"""
        if self.df is None:
            return "Please upload a CSV file first."

        if self.collection is None:
            return "Please upload a CSV file first to initialize the search index."

        try:
            # Search for relevant data
            results = self.collection.query(
                query_texts=[user_query],
                n_results=min(10, len(self.df))
            )

            # Get relevant rows
            relevant_rows = []
            if results['metadatas'] and results['metadatas'][0]:
                for metadata in results['metadatas'][0]:
                    row_idx = metadata['row_index']
                    relevant_rows.append(self.df.iloc[row_idx])
        except Exception as e:
            return f"Error searching data: {str(e)}"

        # Create context for OpenAI
        context = f"""
        CSV Data Schema:
        Columns: {list(self.df.columns)}
        Total Rows: {len(self.df)}
        
        Sample Data (first 5 rows):
        {self.df.head().to_string()}
        
        Relevant Data for Query:
        {pd.DataFrame(relevant_rows).to_string() if relevant_rows else "No relevant data found"}
        
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
        page_title="CSV Chat Assistant",
        page_icon="📊",
        layout="wide"
    )

    st.title("ABI - Automated Business Intelligence")
    st.markdown("Upload a CSV file and chat with your data in natural language!")

    # Initialize app
    if 'app' not in st.session_state:
        st.session_state.app = CSVChatApp()

    app = st.session_state.app

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
                with st.spinner("Processing CSV..."):
                    df = app.process_csv(uploaded_file)
                    if df is not None:
                        st.success("CSV processed successfully!")
                        st.write(f"**Shape:** {df.shape}")
                        st.write(f"**Columns:** {list(df.columns)}")

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

                    # Generate visualization if appropriate
                    viz = app.generate_visualization(prompt)
                    if viz:
                        st.plotly_chart(viz, use_container_width=True)

            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})

        # Clear chat button
        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.rerun()

    else:
        st.info("👆 Please upload a CSV file in the sidebar to get started!")

        # Example queries
        st.header("💡 Example Questions")
        st.markdown("""
        Once you upload your data, you can ask questions like:
        - "What are the main trends in this data?"
        - "Show me the top 5 values in column X"
        - "What's the average of column Y?"
        - "Create a chart showing the relationship between X and Y"
        - "Are there any outliers in the data?"
        - "What insights can you find in this dataset?"
        """)

if __name__ == "__main__":
    main()
