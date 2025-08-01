# 🐘 PostgreSQL + pgvector Setup Guide

This guide will help you set up PostgreSQL with the pgvector extension for vector storage in your CSV chat application.

## 🚀 Quick Setup (Docker)

The easiest way to get started is using Docker:

```bash
# Pull and run PostgreSQL with pgvector
docker run -d \
  --name postgres-vector \
  -e POSTGRES_DB=csv_chat \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_PASSWORD=password \
  -p 5432:5432 \
  pgvector/pgvector:pg15
```

## 🛠️ Manual Installation

### 1. Install PostgreSQL

**macOS (using Homebrew):**
```bash
brew install postgresql
brew services start postgresql
```

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install postgresql postgresql-contrib
sudo systemctl start postgresql
sudo systemctl enable postgresql
```

### 2. Install pgvector Extension

**macOS:**
```bash
brew install pgvector
```

**Ubuntu/Debian:**
```bash
# Add pgvector repository
sudo sh -c 'echo "deb http://apt.postgresql.org/pub/repos/apt $(lsb_release -cs)-pgdg main" > /etc/apt/sources.list.d/pgdg.list'
wget --quiet -O - https://www.postgresql.org/media/keys/ACCC4CF8.asc | sudo apt-key add -
sudo apt update
sudo apt install postgresql-15-pgvector
```

### 3. Create Database and Enable Extension

```bash
# Connect to PostgreSQL
sudo -u postgres psql

# Create database
CREATE DATABASE csv_chat;

# Connect to the database
\c csv_chat

# Enable pgvector extension
CREATE EXTENSION vector;

# Create a test user (optional)
CREATE USER csv_user WITH PASSWORD 'your_password';
GRANT ALL PRIVILEGES ON DATABASE csv_chat TO csv_user;

# Exit
\q
```

## 🔧 Configuration

### Environment Variables

Create a `.env` file in your project root:

```bash
# OpenAI API Configuration
OPENAI_API_KEY=your_openai_api_key_here

# PostgreSQL Configuration
POSTGRES_HOST=localhost
POSTGRES_DB=csv_chat
POSTGRES_USER=postgres
POSTGRES_PASSWORD=password
POSTGRES_PORT=5432
```

### Test Connection

```python
import psycopg2

# Test PostgreSQL connection
conn = psycopg2.connect(
    host="localhost",
    database="csv_chat",
    user="postgres",
    password="password"
)

# Test pgvector extension
with conn.cursor() as cur:
    cur.execute("SELECT version();")
    print("PostgreSQL version:", cur.fetchone())
    
    cur.execute("SELECT * FROM pg_extension WHERE extname = 'vector';")
    if cur.fetchone():
        print("✅ pgvector extension is installed!")
    else:
        print("❌ pgvector extension not found")

conn.close()
```

## 🎯 Running the PostgreSQL Version

```bash
# Install dependencies
pip install -r requirements.txt

# Run the PostgreSQL version
streamlit run postgres_vector_app.py
```

## 🔍 Vector Operations Examples

Once set up, you can perform vector operations:

```sql
-- Create a table with vector column
CREATE TABLE csv_data (
    id SERIAL PRIMARY KEY,
    content TEXT,
    embedding vector(1536)
);

-- Insert data with embeddings
INSERT INTO csv_data (content, embedding) 
VALUES ('sample text', '[0.1, 0.2, 0.3, ...]');

-- Find similar vectors using cosine similarity
SELECT content, 1 - (embedding <=> '[0.1, 0.2, 0.3, ...]') as similarity
FROM csv_data 
ORDER BY embedding <=> '[0.1, 0.2, 0.3, ...]'
LIMIT 5;

-- Combine vector search with SQL analysis
SELECT 
    category,
    AVG(sales) as avg_sales,
    COUNT(*) as count
FROM csv_data 
WHERE 1 - (embedding <=> '[0.1, 0.2, 0.3, ...]') > 0.8
GROUP BY category
ORDER BY avg_sales DESC;
```

## 🚨 Troubleshooting

### Common Issues

1. **pgvector extension not found:**
   ```bash
   # Reinstall pgvector
   sudo apt remove postgresql-15-pgvector
   sudo apt install postgresql-15-pgvector
   ```

2. **Permission denied:**
   ```bash
   # Fix PostgreSQL permissions
   sudo chown -R postgres:postgres /var/lib/postgresql
   sudo chmod 700 /var/lib/postgresql/data
   ```

3. **Connection refused:**
   ```bash
   # Check if PostgreSQL is running
   sudo systemctl status postgresql
   sudo systemctl start postgresql
   ```

### Performance Tips

1. **Create indexes for vector columns:**
   ```sql
   CREATE INDEX ON csv_data USING ivfflat (embedding vector_cosine_ops);
   ```

2. **Use appropriate vector dimensions:**
   - OpenAI text-embedding-ada-002: 1536 dimensions
   - OpenAI text-embedding-3-small: 1536 dimensions
   - OpenAI text-embedding-3-large: 3072 dimensions

3. **Batch operations for large datasets:**
   ```python
   # Use executemany for bulk inserts
   with conn.cursor() as cur:
       cur.executemany(insert_sql, data_batch)
   conn.commit()
   ```

## 🎉 Success!

Once PostgreSQL + pgvector is set up, you'll have:
- ✅ Vector similarity search
- ✅ ACID-compliant transactions
- ✅ SQL integration
- ✅ Production-ready scalability
- ✅ Cost-effective solution

Your CSV chat app will now use PostgreSQL for both structured data and vector storage! 