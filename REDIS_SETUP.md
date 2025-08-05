# Redis Setup Guide for Session Management

This guide will help you set up Redis for session management in your ABI application.

## Prerequisites

1. Redis server installed and running
2. Python `redis` package (already in requirements.txt)

## Installation Options

### Option 1: Local Redis Installation

#### macOS (using Homebrew):
```bash
brew install redis
brew services start redis
```

#### Ubuntu/Debian:
```bash
sudo apt update
sudo apt install redis-server
sudo systemctl start redis-server
sudo systemctl enable redis-server
```

#### Windows:
Download Redis from https://redis.io/download and follow installation instructions.

### Option 2: Docker Redis (Recommended)

Add this to your `docker-compose.yml`:

```yaml
services:
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    command: redis-server --appendonly yes
    restart: unless-stopped

volumes:
  redis_data:
```

Then run:
```bash
docker-compose up -d redis
```

### Option 3: Cloud Redis Services

- **Redis Cloud**: https://redis.com/try-free/
- **AWS ElastiCache**: https://aws.amazon.com/elasticache/
- **Google Cloud Memorystore**: https://cloud.google.com/memorystore

## Environment Variables

Create a `.env` file in your project root with these Redis settings:

```env
# Redis Configuration
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
REDIS_PASSWORD=  # Leave empty for local Redis without password
```

For cloud Redis services, use the provided connection details.

## Testing Redis Connection

Run this Python script to test your Redis connection:

```python
import redis
import os
from dotenv import load_dotenv

load_dotenv()

try:
    r = redis.Redis(
        host=os.getenv("REDIS_HOST", "localhost"),
        port=int(os.getenv("REDIS_PORT", 6379)),
        db=int(os.getenv("REDIS_DB", 0)),
        password=os.getenv("REDIS_PASSWORD", None),
        decode_responses=True
    )
    
    # Test connection
    r.ping()
    print("✅ Redis connection successful!")
    
    # Test basic operations
    r.set("test_key", "test_value")
    value = r.get("test_key")
    print(f"✅ Test value retrieved: {value}")
    
    # Clean up
    r.delete("test_key")
    print("✅ Redis test completed successfully!")
    
except Exception as e:
    print(f"❌ Redis connection failed: {str(e)}")
```

## Session Data Structure

The application stores session data in Redis with the following structure:

- **Session Conversation**: `session:{session_id}:conversation` (Redis List)
- **Session Metadata**: `session:{session_id}:metadata` (Redis String)

### Conversation Entry Format:
```json
{
  "timestamp": "2024-01-15T10:30:00.123456",
  "user_query": "What are the main trends in this data?",
  "assistant_response": "Based on the data analysis...",
  "execution_time": 2.45,
  "metadata": {
    "query_hash": "abc123...",
    "strategy": "semantic_search",
    "complexity": "medium",
    "docs_found": 5,
    "cache_hit": false
  }
}
```

## Session Expiration

Sessions automatically expire after 24 hours (86400 seconds) to prevent Redis memory bloat.

## Monitoring Redis

### Check Redis Memory Usage:
```bash
redis-cli info memory
```

### List All Sessions:
```bash
redis-cli keys "session:*:metadata"
```

### View Session Data:
```bash
redis-cli lrange "session:{session_id}:conversation" 0 -1
```

## Troubleshooting

### Common Issues:

1. **Connection Refused**: Make sure Redis server is running
2. **Authentication Failed**: Check REDIS_PASSWORD in .env
3. **Memory Issues**: Sessions auto-expire after 24 hours
4. **Port Conflicts**: Ensure port 6379 is available

### Redis CLI Commands:
```bash
# Start Redis CLI
redis-cli

# Test connection
ping

# Check Redis info
info

# Monitor Redis operations
monitor

# Exit CLI
exit
```

## Security Considerations

1. **Production**: Use Redis with authentication enabled
2. **Network**: Restrict Redis to localhost or use VPN
3. **Backup**: Enable Redis persistence (AOF/RDB)
4. **Monitoring**: Set up Redis monitoring and alerts

## Performance Tips

1. **Memory**: Monitor Redis memory usage
2. **Connection Pooling**: Redis-py handles this automatically
3. **Pipelining**: Not needed for session management
4. **Expiration**: Sessions auto-expire to prevent memory bloat 