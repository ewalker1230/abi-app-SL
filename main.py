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

# LangChain imports
from langchain_community.document_loaders import CSVLoader, UnstructuredExcelLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

# Load environment variables
load_dotenv()

# Configure OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class CSVChatApp:
    def __init__(self):
        self.dfs = {}  # Dictionary to store multiple dataframes
        self.text_files = set()  # Set to track uploaded text files
        self.text_contents = {}  # Dictionary to store original text content
        self.vectorstore = None  # LangChain vectorstore
        self.embeddings = None  # OpenAI embeddings
        self.text_splitter = None  # Text splitter
        self.chat_history = []
        self.processed_files = set()  # Set to track processed file names

    def setup_langchain(self):
        """Initialize LangChain components for vector storage"""
        try:
            if self.embeddings is None:
                self.embeddings = OpenAIEmbeddings()
            
            if self.text_splitter is None:
                self.text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200,
                    length_function=len,
                    separators=["\n\n", "\n", " ", ""]
                )
            
            # Initialize vectorstore if not exists
            if self.vectorstore is None:
                # Use persistent ChromaDB with LangChain
                self.vectorstore = Chroma(
                    persist_directory="./chroma_db",
                    embedding_function=self.embeddings,
                    collection_name="csv_data"
                )
                
        except Exception as e:
            st.error(f"Error setting up LangChain: {str(e)}")
            self.embeddings = None
            self.text_splitter = None
            self.vectorstore = None

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
            self.process_with_langchain(uploaded_file, filename, "csv")
            
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
                
                # Add metadata to documents
                for doc in documents:
                    doc.metadata.update({
                        "filename": filename,
                        "file_type": "csv",
                        "content_type": "data_row"
                    })
                
                # Split documents
                chunks = self.text_splitter.split_documents(documents)
                
                # Add to vectorstore
                self.vectorstore.add_documents(chunks)
                
                # Clean up temp file
                os.remove(temp_path)
                
                st.success(f"Processed {len(chunks)} chunks from {filename} with LangChain")
                
        except Exception as e:
            st.error(f"Error processing {filename} with LangChain: {str(e)}")

    def process_excel_sheet_with_langchain(self, df: pd.DataFrame, filename: str):
        """Process Excel sheet data with LangChain"""
        try:
            # Convert dataframe to documents
            documents = []
            for idx, row in df.iterrows():
                # Create document from row
                row_text = " ".join([f"{col}: {val}" for col, val in row.items()])
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
            
            # Split documents
            chunks = self.text_splitter.split_documents(documents)
            
            # Add to vectorstore
            self.vectorstore.add_documents(chunks)
            
            st.success(f"Processed {len(chunks)} chunks from {filename} with LangChain")
            
        except Exception as e:
            st.error(f"Error processing Excel sheet {filename} with LangChain: {str(e)}")

    def process_text_with_langchain(self, text_content: str, filename: str):
        """Process text content with LangChain"""
        try:
            # Create document from text content
            doc = type('Document', (), {
                'page_content': text_content,
                'metadata': {
                    'filename': filename,
                    'file_type': 'text',
                    'content_type': 'text'
                }
            })()
            
            # Split document
            chunks = self.text_splitter.split_documents([doc])
            
            # Add metadata to chunks
            for chunk in chunks:
                chunk.metadata.update({
                    'filename': filename,
                    'file_type': 'text',
                    'content_type': 'text_chunk'
                })
            
            # Add to vectorstore
            self.vectorstore.add_documents(chunks)
            
            st.success(f"Processed {len(chunks)} text chunks from {filename} with LangChain")
            
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
        """Query the data using OpenAI and LangChain"""
        if not self.dfs and not self.text_files:
            return "Please upload at least one CSV file or text document first."

        if self.vectorstore is None:
            st.error("LangChain vectorstore is None. Please try uploading your files again.")
            return "Please upload a CSV file or text document first to initialize the search index."

        # Check if query requires aggregation
        query_lower = user_query.lower()
        aggregation_keywords = ['sum', 'total', 'average', 'mean', 'count', 'aggregate', 'group by']
        requires_aggregation = any(keyword in query_lower for keyword in aggregation_keywords)

        try:
            # Search for relevant data using LangChain
            docs = self.vectorstore.similarity_search(user_query, k=10)
            
            if not docs:
                return "No relevant data found. Please make sure your files were processed successfully."

            # Process LangChain documents
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
                            
                            # Determine data type and add sheet info if available
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
        except Exception as e:
            st.error(f"Error searching data: {str(e)}")
            return f"Error searching data: {str(e)}"

        # Create context for OpenAI
        context_parts = []
        
        # If aggregation is needed, perform it first
        aggregation_results = ""
        if requires_aggregation:
            aggregation_results = self.perform_aggregation(user_query)
        
        # Add cross-dataset insights
        cross_dataset_insights = self.generate_cross_dataset_insights(user_query)
        
        # Add information about all loaded datasets
        for filename, df in self.dfs.items():
            # Check if this is a sheet-specific filename (contains underscore and not a file extension)
            if "_" in filename and not filename.endswith('.csv') and not filename.endswith('.xlsx'):
                # This is a sheet from an Excel file
                parts = filename.split("_", 1)
                original_file = parts[0]
                sheet_name = parts[1]
                
                context_parts.append(f"""
                Dataset: {filename}
                Type: Excel Sheet
                Original File: {original_file}
                Sheet Name: {sheet_name}
                Total Rows: {len(df)}
                Columns: {list(df.columns)}
                Sample Data (first 3 rows):
                {df.head(3).to_string()}
                """)
            else:
                # This is a regular CSV or single-sheet Excel file
                file_type = "Excel" if filename.endswith(('.xlsx', '.xls')) else "CSV"
                context_parts.append(f"""
                Dataset: {filename}
                Type: {file_type}
                Columns: {list(df.columns)}
                Total Rows: {len(df)}
                Sample Data (first 3 rows):
                {df.head(3).to_string()}
                """)
        
        # Add information about text files
        for filename in self.text_files:
            context_parts.append(f"""
            Document: {filename}
            Type: Text Document
            Status: Indexed in vector database
            """)
        
        # Add relevant data found
        if relevant_data:
            relevant_data_parts = []
            for item in relevant_data:
                if item['type'] == 'csv_row':
                    relevant_data_parts.append(f"""
                From CSV file {item['filename']}:
                {item['data'].to_string()}
                LangChain content: {item.get('content', 'N/A')}
                """)
                elif item['type'] == 'excel_row':
                    sheet_info = f" (Sheet: {item['sheet_name']})" if item['sheet_name'] else ""
                    original_file = item.get('original_file', item['filename'])
                    relevant_data_parts.append(f"""
                From Excel file {original_file}{sheet_info}:
                {item['data'].to_string()}
                LangChain content: {item.get('content', 'N/A')}
                """)
                elif item['type'] == 'text_chunk':
                    relevant_data_parts.append(f"""
                From text document {item['filename']}:
                {item['data']}
                """)
            relevant_data_text = "\n".join(relevant_data_parts)
        else:
            relevant_data_text = "No relevant data found"
        
        context = f"""
        Available Datasets:
        {chr(10).join(context_parts)}
        
        {f"Aggregation Results: {aggregation_results}" if aggregation_results else ""}
        
        {f"Cross-Dataset Insights: {cross_dataset_insights}" if cross_dataset_insights else ""}
        
        Relevant Data for Query:
        {relevant_data_text}
        
        User Query: {user_query}
        
        Please provide a helpful response based on this data. Consider the holistic view of all datasets,
        their relationships, and patterns. If the query asks for analysis, provide comprehensive insights
        that span across the entire dataset. If it asks for specific data, provide the relevant information
        with context about how it fits into the broader picture. Be conversational and helpful.
        """

        try:
            response = client.chat.completions.create(model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful data analyst assistant. The data you are looking at is advertising spend data related to marketing campaigns. Sometimes the data is spread across multiple sheets. Your job is to indentify helpful information and insights across all of the sheets. When there are multiple sheets, consider information from all of the sheets before giving a response"},
                {"role": "user", "content": context}
            ],
            max_tokens=1500,
            temperature=0.7)
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
                
                # Check if columns exist
                if value_col not in df.columns:
                    st.error(f"Column '{value_col}' not found in dataset. Available columns: {list(df.columns)}")
                    return None
                
                if group_col_actual not in df.columns:
                    st.error(f"Column '{group_col_actual}' not found in dataset. Available columns: {list(df.columns)}")
                    return None
                
                try:
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
                except Exception as e:
                    st.error(f"Error creating chart: {str(e)}")
                    return None
        
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

    # Just the logo as title
    st.image("assets/abi_horiizontal_lime.png", width=450)
    st.markdown("Upload CSV, Excel files and text documents to chat with your data in natural language!")

    # Initialize app
    if 'app' not in st.session_state:
        st.session_state.app = CSVChatApp()

    app = st.session_state.app

    # Sidebar for file upload
    with st.sidebar:
        st.header("📁 Upload Data")
        
        # Data file upload (CSV and Excel)
        #st.subheader("📊 Data Files")
        uploaded_files = st.file_uploader(
            "Choose CSV or Excel files",
            type=['csv', 'xlsx', 'xls'],
            accept_multiple_files=True,
            help="Upload one or more CSV or Excel files to start chatting with them"
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
                        elif uploaded_file.name.lower().endswith(('.xlsx', '.xls')):
                            df = app.process_excel(uploaded_file, uploaded_file.name)
                        else:
                            st.error(f"Unsupported file type: {uploaded_file.name}")
                            continue
                            
                        if df is not None:
                            st.success(f"{uploaded_file.name} processed successfully!")
                            if hasattr(df, 'shape'):
                                st.write(f"**Shape:** {df.shape}")
                                st.write(f"**Columns:** {list(df.columns)}")
                            else:
                                # This is a summary dataframe for multi-sheet Excel files
                                st.write("**Processed Sheets:**")
                                st.dataframe(df)
                else:
                    st.info(f"✅ {uploaded_file.name} already processed")
        
        # Text file upload
        #st.subheader("📄 Text Files")
        uploaded_text_files = st.file_uploader(
            "Choose text files",
            type=['txt'],
            accept_multiple_files=True,
            help="Upload one or more text files to add to the knowledge base"
        )
        
        if uploaded_text_files:
            for uploaded_file in uploaded_text_files:
                # Check if this text file has already been processed
                if uploaded_file.name not in app.processed_files:
                    with st.spinner(f"Processing text file {uploaded_file.name}..."):
                        success = app.process_text_file(uploaded_file, uploaded_file.name)
                        if success:
                            st.success(f"Text file '{uploaded_file.name}' processed and added to vector database!")
                        else:
                            st.error(f"Failed to process text file '{uploaded_file.name}'")
                else:
                    st.info(f"✅ Text file '{uploaded_file.name}' already processed")
        
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
                app.text_contents.clear()
                app.processed_files.clear()  # Clear processed files tracking
                app.setup_chroma()  # This will clear the ChromaDB collection
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

                    # Generate visualization if appropriate
                    viz = app.generate_visualization(prompt)
                    if viz:
                        st.subheader("📊 Generated Visualization")
                        st.plotly_chart(viz, use_container_width=True)
                        st.caption(f"Chart type: {app.parse_visualization_intent(prompt)['chart_type']}")
                    else:
                        # Debug info
                        intent = app.parse_visualization_intent(prompt)
                        if intent['chart_type']:
                            st.info(f"Debug: Chart type detected: {intent['chart_type']}, Columns: {intent['columns']}, Parameters: {intent['parameters']}")

            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})

        # Clear chat button
        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.rerun()

    else:
        st.info("👆 Please upload a CSV or Excel file in the sidebar to get started!")

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
