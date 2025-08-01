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
        self.dfs = {}  # Dictionary to store multiple dataframes
        self.chroma_client = None
        self.collection = None
        self.chat_history = []

    def setup_chroma(self):
        """Initialize ChromaDB for vector storage"""
        try:
            if self.chroma_client is None:
                # Use persistent storage to avoid context leaks
                self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
                
            # Delete existing collection if it exists (to avoid metadata format issues)
            try:
                self.chroma_client.delete_collection(name="csv_data")
            except:
                pass  # Collection doesn't exist, which is fine
                
            # Create new collection
            self.collection = self.chroma_client.create_collection(
                name="csv_data",
                metadata={"description": "CSV data for semantic search"}
            )
        except Exception as e:
            st.error(f"Error setting up ChromaDB: {str(e)}")
            self.chroma_client = None
            self.collection = None

    def process_csv(self, uploaded_file, filename: str) -> pd.DataFrame:
        """Process uploaded CSV file"""
        try:
            df = pd.read_csv(uploaded_file)
            self.dfs[filename] = df
            self.setup_chroma()
            self.index_data(filename, df)
            return df
        except Exception as e:
            st.error(f"Error processing CSV: {str(e)}")
            return None

    def index_data(self, filename: str, df: pd.DataFrame):
        """Index CSV data for semantic search"""
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

            for idx, row in df.iterrows():
                # Create a text representation of the row
                row_text = " ".join([f"{col}: {val}" for col, val in row.items()])
                documents.append(row_text)
                metadatas.append({"row_index": idx, "filename": filename})
                ids.append(f"{filename}_row_{idx}")

            # Add to ChromaDB
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            st.success(f"Indexed {len(documents)} rows from {filename} in ChromaDB")
        except Exception as e:
            st.error(f"Error indexing data from {filename}: {str(e)}")

    def query_data(self, user_query: str) -> str:
        """Query the data using OpenAI and ChromaDB"""
        if not self.dfs:
            return "Please upload at least one CSV file first."

        if self.collection is None:
            st.error("ChromaDB collection is None. Please try uploading your CSV files again.")
            return "Please upload a CSV file first to initialize the search index."

        try:
            # Calculate total rows across all dataframes
            total_rows = sum(len(df) for df in self.dfs.values())
            
            # Check if collection has any data
            collection_count = self.collection.count()
            if collection_count == 0:
                return "No data has been indexed yet. Please make sure your CSV files were processed successfully."
            
            # Search for relevant data
            results = self.collection.query(
                query_texts=[user_query],
                n_results=min(10, total_rows)
            )

            # Get relevant rows from all dataframes
            relevant_rows = []
            if results and 'metadatas' in results and results['metadatas'] and results['metadatas'][0]:
                for metadata in results['metadatas'][0]:
                    if metadata and 'row_index' in metadata and 'filename' in metadata:
                        row_idx = metadata['row_index']
                        filename = metadata['filename']
                        if filename in self.dfs:
                            try:
                                relevant_rows.append({
                                    'filename': filename,
                                    'data': self.dfs[filename].iloc[row_idx]
                                })
                            except Exception as row_error:
                                st.warning(f"Error accessing row {row_idx} from {filename}: {str(row_error)}")
                                continue
        except Exception as e:
            st.error(f"Error searching data: {str(e)}")
            return f"Error searching data: {str(e)}"

        # Create context for OpenAI
        context_parts = []
        
        # Add information about all loaded datasets
        for filename, df in self.dfs.items():
            context_parts.append(f"""
            Dataset: {filename}
            Columns: {list(df.columns)}
            Total Rows: {len(df)}
            Sample Data (first 3 rows):
            {df.head(3).to_string()}
            """)
        
        # Add relevant data found
        if relevant_rows:
            relevant_data_parts = []
            for item in relevant_rows:
                relevant_data_parts.append(f"""
                From {item['filename']}:
                {item['data'].to_string()}
                """)
            relevant_data_text = "\n".join(relevant_data_parts)
        else:
            relevant_data_text = "No relevant data found"
        
        context = f"""
        Available Datasets:
        {chr(10).join(context_parts)}
        
        Relevant Data for Query:
        {relevant_data_text}
        
        User Query: {user_query}
        
        Please provide a helpful response based on this data. If the query asks for analysis, 
        provide insights. If it asks for specific data, provide the relevant information.
        Be conversational and helpful. When referencing data, mention which dataset it comes from.
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
        if not self.dfs:
            return None

        # Simple visualization logic based on query keywords
        query_lower = query.lower()

        if "chart" in query_lower or "graph" in query_lower or "plot" in query_lower:
            # For now, use the first dataframe for visualization
            # You could enhance this to work with multiple dataframes
            first_df = list(self.dfs.values())[0]
            numeric_cols = first_df.select_dtypes(include=['number']).columns

            if len(numeric_cols) >= 2:
                fig = px.scatter(first_df, x=numeric_cols[0], y=numeric_cols[1], 
                               title=f"{numeric_cols[0]} vs {numeric_cols[1]}")
                return fig
            elif len(numeric_cols) == 1:
                fig = px.histogram(first_df, x=numeric_cols[0], 
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
        uploaded_files = st.file_uploader(
            "Choose CSV files",
            type=['csv'],
            accept_multiple_files=True,
            help="Upload one or more CSV files to start chatting with them"
        )

        if uploaded_files:
            for uploaded_file in uploaded_files:
                if uploaded_file.name not in app.dfs or st.button(f"Reload {uploaded_file.name}"):
                    with st.spinner(f"Processing {uploaded_file.name}..."):
                        df = app.process_csv(uploaded_file, uploaded_file.name)
                        if df is not None:
                            st.success(f"{uploaded_file.name} processed successfully!")
                            st.write(f"**Shape:** {df.shape}")
                            st.write(f"**Columns:** {list(df.columns)}")
        
        # Show loaded datasets
        if app.dfs:
            st.header("📊 Loaded Datasets")
            
            # Add button to re-index all data
            if st.button("🔄 Re-index All Data"):
                with st.spinner("Re-indexing all data..."):
                    app.setup_chroma()  # This will clear and recreate the collection
                    for filename, df in app.dfs.items():
                        app.index_data(filename, df)
                st.success("All data re-indexed successfully!")
            
            # Add button to clear all data
            if st.button("🗑️ Clear All Data"):
                app.dfs.clear()
                app.setup_chroma()  # This will clear the ChromaDB collection
                st.success("All data cleared!")
                st.rerun()
            
            for filename, df in app.dfs.items():
                with st.expander(f"{filename} ({len(df)} rows)"):
                    st.dataframe(df.head(5))
                    st.write(f"**Columns:** {list(df.columns)}")
                    
                    # Add remove button for individual datasets
                    if st.button(f"❌ Remove {filename}", key=f"remove_{filename}"):
                        # Remove from memory
                        del app.dfs[filename]
                        
                        # Remove from ChromaDB
                        if app.collection:
                            try:
                                # Get all IDs for this filename
                                results = app.collection.get(
                                    where={"filename": filename}
                                )
                                if results['ids']:
                                    app.collection.delete(ids=results['ids'])
                                    st.success(f"Removed {filename} from database")
                            except Exception as e:
                                st.error(f"Error removing from database: {str(e)}")
                        
                        st.rerun()

        # Configuration info
        st.header("🔑 Configuration")
        if os.getenv("OPENAI_API_KEY"):
            st.success("✅ OpenAI API Key configured")
        else:
            st.error("❌ OpenAI API Key not found in .env file")
            st.info("Please add OPENAI_API_KEY to your .env file")

    # Main chat interface
    if app.dfs:
        st.header("💬 Chat with Your Data")

        # Display data preview
        with st.expander("📋 Data Preview"):
            for filename, df in app.dfs.items():
                st.subheader(f"📄 {filename}")
                st.dataframe(df.head(5))
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
