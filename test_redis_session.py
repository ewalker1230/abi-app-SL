#!/usr/bin/env python3
"""
Test script for Redis session management functionality
"""

import os
import sys
import time
from datetime import datetime
from dotenv import load_dotenv

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the SessionManager class
from main import SessionManager

def test_redis_connection():
    """Test basic Redis connection"""
    print("🔍 Testing Redis connection...")
    
    session_manager = SessionManager()
    
    if session_manager.redis_available:
        print("✅ Redis connection successful!")
        return True
    else:
        print("❌ Redis connection failed!")
        return False

def test_session_creation():
    """Test session ID creation"""
    print("\n🔍 Testing session creation...")
    
    session_manager = SessionManager()
    
    # Generate a new session ID
    session_id = session_manager.generate_session_id()
    print(f"✅ Generated session ID: {session_id}")
    
    # Test session ID format (should be a UUID)
    if len(session_id) == 36 and session_id.count('-') == 4:
        print("✅ Session ID format is valid UUID")
        return session_id
    else:
        print("❌ Session ID format is invalid")
        return None

def test_conversation_saving(session_id):
    """Test saving conversation turns"""
    print(f"\n🔍 Testing conversation saving for session: {session_id[:8]}...")
    
    session_manager = SessionManager()
    
    # Test data
    test_queries = [
        "What are the main trends in this data?",
        "Show me the top 5 values",
        "Create a chart for sales data"
    ]
    
    test_responses = [
        "Based on the analysis, the main trends are...",
        "Here are the top 5 values from your dataset...",
        "I've created a chart showing the sales data..."
    ]
    
    for i, (query, response) in enumerate(zip(test_queries, test_responses), 1):
        print(f"  Saving conversation turn {i}...")
        
        # Save conversation turn
        session_manager.save_conversation_turn(
            session_id=session_id,
            user_query=query,
            assistant_response=response,
            execution_time=1.5 + i * 0.2,
            metadata={
                'turn_number': i,
                'test': True,
                'timestamp': datetime.now().isoformat()
            }
        )
        
        time.sleep(0.1)  # Small delay to ensure proper ordering
    
    print("✅ All conversation turns saved successfully!")

def test_conversation_retrieval(session_id):
    """Test retrieving conversation history"""
    print(f"\n🔍 Testing conversation retrieval for session: {session_id[:8]}...")
    
    session_manager = SessionManager()
    
    # Retrieve conversation history
    conversation_history = session_manager.get_conversation_history(session_id)
    
    if conversation_history:
        print(f"✅ Retrieved {len(conversation_history)} conversation turns")
        
        # Display the conversation history
        for i, turn in enumerate(conversation_history, 1):
            print(f"\n  Turn {i}:")
            print(f"    Timestamp: {turn['timestamp']}")
            print(f"    Query: {turn['user_query'][:50]}...")
            print(f"    Response: {turn['assistant_response'][:50]}...")
            print(f"    Execution Time: {turn['execution_time']:.2f}s")
            print(f"    Metadata: {turn['metadata']}")
    else:
        print("❌ No conversation history found")
        return False
    
    return True

def test_session_metadata(session_id):
    """Test session metadata operations"""
    print(f"\n🔍 Testing session metadata for session: {session_id[:8]}...")
    
    session_manager = SessionManager()
    
    # Get session metadata
    metadata = session_manager.get_session_metadata(session_id)
    
    if metadata:
        print("✅ Session metadata retrieved:")
        for key, value in metadata.items():
            print(f"    {key}: {value}")
    else:
        print("❌ No session metadata found")
        return False
    
    return True

def test_session_cleanup(session_id):
    """Test session deletion"""
    print(f"\n🔍 Testing session cleanup for session: {session_id[:8]}...")
    
    session_manager = SessionManager()
    
    # Delete the session
    success = session_manager.delete_session(session_id)
    
    if success:
        print("✅ Session deleted successfully")
        
        # Verify deletion
        conversation_history = session_manager.get_conversation_history(session_id)
        metadata = session_manager.get_session_metadata(session_id)
        
        if not conversation_history and not metadata:
            print("✅ Session data cleanup verified")
            return True
        else:
            print("❌ Session data still exists after deletion")
            return False
    else:
        print("❌ Failed to delete session")
        return False

def test_all_sessions():
    """Test getting all sessions"""
    print("\n🔍 Testing get all sessions...")
    
    session_manager = SessionManager()
    
    # Get all sessions
    all_sessions = session_manager.get_all_sessions()
    
    if all_sessions:
        print(f"✅ Found {len(all_sessions)} active sessions:")
        for session in all_sessions:
            print(f"    Session ID: {session['session_id'][:8]}...")
            print(f"    Created: {session.get('created_at', 'Unknown')}")
            print(f"    Total Turns: {session.get('total_turns', 0)}")
            print("    ---")
    else:
        print("ℹ️ No active sessions found")
    
    return True

def main():
    """Run all tests"""
    print("🚀 Starting Redis Session Management Tests")
    print("=" * 50)
    
    # Load environment variables
    load_dotenv()
    
    # Test 1: Redis Connection
    if not test_redis_connection():
        print("\n❌ Redis connection failed. Please check your Redis setup.")
        print("See REDIS_SETUP.md for setup instructions.")
        return
    
    # Test 2: Session Creation
    session_id = test_session_creation()
    if not session_id:
        print("\n❌ Session creation failed.")
        return
    
    # Test 3: Conversation Saving
    test_conversation_saving(session_id)
    
    # Test 4: Conversation Retrieval
    if not test_conversation_retrieval(session_id):
        print("\n❌ Conversation retrieval failed.")
        return
    
    # Test 5: Session Metadata
    if not test_session_metadata(session_id):
        print("\n❌ Session metadata test failed.")
        return
    
    # Test 6: All Sessions
    test_all_sessions()
    
    # Test 7: Session Cleanup
    if not test_session_cleanup(session_id):
        print("\n❌ Session cleanup failed.")
        return
    
    print("\n" + "=" * 50)
    print("🎉 All Redis Session Management Tests Passed!")
    print("\nYour Redis session management is working correctly.")
    print("You can now run your Streamlit app with session tracking enabled.")

if __name__ == "__main__":
    main() 