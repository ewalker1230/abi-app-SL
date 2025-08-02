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
import re

# Load environment variables
load_dotenv()

# Configure OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class CSVChatApp:
    def __init__(self):
        self.dfs = {}  # Dictionary to store multiple dataframes
        self.text_files = set()  # Set to track uploaded text files
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

    def process_text_file(self, uploaded_file, filename: str) -> bool:
        """Process uploaded text file and add to vector database"""
        try:
            # Read the text content
            text_content = uploaded_file.read().decode('utf-8')
            
            # Setup ChromaDB if not already done
            self.setup_chroma()
            
            # Index the text content
            self.index_text_data(filename, text_content)
            
            # Track the uploaded text file
            self.text_files.add(filename)
            
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
        
        # Handle grouping
        if 'group_by' in intent['parameters']:
            group_col = intent['parameters']['group_by']
            # Find the actual column name that matches
            group_col_actual = None
            for col in df.columns:
                if group_col in col.lower() or col.lower() in group_col:
                    group_col_actual = col
                    break
            
            if group_col_actual and intent['columns']:
                value_col = intent['columns'][0]
                # Aggregate if needed
                if intent['parameters'].get('aggregation') == 'mean':
                    data = df.groupby(group_col_actual)[value_col].mean().reset_index()
                elif intent['parameters'].get('aggregation') == 'sum':
                    data = df.groupby(group_col_actual)[value_col].sum().reset_index()
                else:
                    data = df.groupby(group_col_actual)[value_col].count().reset_index()
                
                fig = px.bar(data, x=group_col_actual, y=value_col,
                           title=f"{value_col} by {group_col_actual}")
                return fig
        
        # Simple bar chart of first column
        if intent['columns']:
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
        query_lower = query.lower()
        
        # Step 1: Detect chart type
        chart_type = self.detect_chart_type(query_lower)
        
        # Step 2: Extract columns and datasets
        columns, datasets = self.extract_columns_and_datasets(query_lower)
        
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
        columns = []
        datasets = []
        
        # Check each dataset and its columns
        for df_name, df in self.dfs.items():
            for col in df.columns:
                col_lower = col.lower().replace('_', ' ')
                if col_lower in query or col in query:
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
        page_title="CSV Chat Assistant",
        page_icon="📊",
        layout="wide"
    )

    st.title("ABI - Agentic Business Intelligence")
    st.markdown("Upload CSV files and text documents to chat with your data in natural language!")

    # Initialize app
    if 'app' not in st.session_state:
        st.session_state.app = CSVChatApp()

    app = st.session_state.app

    # Sidebar for file upload
    with st.sidebar:
        st.header("📁 Upload Data")
        
        # CSV file upload
        st.subheader("📊 CSV Files")
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
        
        # Text file upload
        st.subheader("📄 Text Files")
        uploaded_text_files = st.file_uploader(
            "Choose text files",
            type=['txt'],
            accept_multiple_files=True,
            help="Upload one or more text files to add to the knowledge base"
        )
        
        if uploaded_text_files:
            for uploaded_file in uploaded_text_files:
                with st.spinner(f"Processing text file {uploaded_file.name}..."):
                    success = app.process_text_file(uploaded_file, uploaded_file.name)
                    if success:
                        st.success(f"Text file '{uploaded_file.name}' processed and added to vector database!")
                    else:
                        st.error(f"Failed to process text file '{uploaded_file.name}'")
        
        # Show loaded datasets
        if app.dfs or app.text_files:
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
                app.text_files.clear()
                app.setup_chroma()  # This will clear the ChromaDB collection
                st.success("All data cleared!")
                st.rerun()
            
            # Show CSV datasets
            for filename, df in app.dfs.items():
                with st.expander(f"📊 {filename} ({len(df)} rows)"):
                    st.dataframe(df.head(5))
                    st.write(f"**Columns:** {list(df.columns)}")
                    
                    # Add remove button for individual datasets
                    if st.button(f"❌ Remove {filename}", key=f"remove_csv_{filename}"):
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
            
            # Show text files
            for filename in app.text_files:
                with st.expander(f"📄 {filename}"):
                    st.write(f"**Type:** Text Document")
                    st.write(f"**Status:** Indexed in vector database")
                    
                    # Add remove button for text files
                    if st.button(f"❌ Remove {filename}", key=f"remove_text_{filename}"):
                        # Remove from tracking
                        app.text_files.remove(filename)
                        
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
    if app.dfs or app.text_files:
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
                        st.subheader("📊 Generated Visualization")
                        st.plotly_chart(viz, use_container_width=True)
                        st.caption(f"Chart type: {app.parse_visualization_intent(prompt)['chart_type']}")

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
