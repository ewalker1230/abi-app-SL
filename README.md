# 📊 CSV Chat Assistant

A powerful application that allows you to upload CSV files and chat with your data using natural language. Built with Streamlit, OpenAI, and ChromaDB.

## 🚀 Features

- **CSV Upload & Processing**: Upload any CSV file and automatically process it
- **Natural Language Queries**: Ask questions about your data in plain English
- **Semantic Search**: Find relevant data using AI-powered search
- **Interactive Visualizations**: Automatic chart generation based on your queries
- **Real-time Chat Interface**: Conversational experience with your data
- **Data Insights**: Get AI-powered analysis and insights

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone <your-repo-url>
   cd abi-app
   ```

2. **Install dependencies**:
   ```bash
   pip install -e .
   ```

3. **Set up environment variables**:
   ```bash
   cp env_example.txt .env
   ```
   Then edit `.env` and add your OpenAI API key:
   ```
   OPENAI_API_KEY=your_actual_api_key_here
   ```

## 🎯 Usage

1. **Start the application**:
   ```bash
   streamlit run main.py
   ```

2. **Upload a CSV file** using the sidebar

3. **Start chatting** with your data! Try questions like:
   - "What are the main trends in this data?"
   - "Show me the top 5 sales by region"
   - "What's the average sales amount?"
   - "Create a chart showing sales by category"
   - "Are there any outliers in the data?"

## 📁 Project Structure

```
abi-app/
├── main.py              # Main Streamlit application
├── pyproject.toml       # Project dependencies
├── sample_data.csv      # Sample data for testing
├── env_example.txt      # Environment variables template
└── README.md           # This file
```

## 🔧 Technical Architecture

### Core Components

1. **Streamlit Frontend**: Modern, responsive web interface
2. **Pandas Data Processing**: Efficient CSV handling and manipulation
3. **OpenAI Integration**: Natural language processing and generation
4. **ChromaDB Vector Database**: Semantic search and retrieval
5. **Plotly Visualizations**: Interactive charts and graphs

### Data Flow

1. **Upload**: CSV file is uploaded and processed by Pandas
2. **Indexing**: Data is indexed in ChromaDB for semantic search
3. **Query Processing**: User queries are processed using OpenAI
4. **Context Retrieval**: Relevant data is retrieved from ChromaDB
5. **Response Generation**: AI generates contextual responses
6. **Visualization**: Charts are generated when appropriate

## 🎨 Customization

### Adding New Features

- **Custom Visualizations**: Extend the `generate_visualization` method
- **Additional Data Sources**: Support for Excel, JSON, or database connections
- **Advanced Analytics**: Add statistical analysis capabilities
- **Export Features**: Add data export functionality

### Configuration Options

- **Model Selection**: Switch between different OpenAI models
- **ChromaDB Settings**: Configure vector database parameters
- **UI Customization**: Modify Streamlit theme and layout

## 🔒 Security Considerations

- API keys are stored in environment variables
- Data is processed locally (ChromaDB runs in-memory)
- No data is permanently stored on external servers

## 🚀 Deployment

### Local Development
```bash
streamlit run main.py
```

### Production Deployment
1. Set up a production server
2. Configure environment variables
3. Use a production-grade ChromaDB instance
4. Set up proper authentication and authorization

## 📊 Sample Data

The application includes `sample_data.csv` with sales data for testing:
- Date, Product, Category, Sales, Quantity, Region
- 15 sample records with electronics and furniture sales

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License.

## 🆘 Support

For issues and questions:
1. Check the documentation
2. Review existing issues
3. Create a new issue with detailed information

---

**Happy Data Chatting! 🎉**
