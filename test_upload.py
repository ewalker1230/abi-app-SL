import pandas as pd
import os

def test_csv_upload_process():
    """Test the exact CSV upload process from main.py"""
    
    # Simulate the file upload process
    filename = "fake_ad_campaigns.csv"
    temp_path = f"temp_{filename}"
    
    print(f"=== Testing CSV upload process for {filename} ===")
    
    # Step 1: Copy the file (simulating upload)
    try:
        with open('data/fake_ad_campaigns.csv', 'rb') as src:
            content = src.read()
            with open(temp_path, 'wb') as dst:
                dst.write(content)
        
        file_size = os.path.getsize(temp_path)
        print(f"✓ File saved: {temp_path} (size: {file_size} bytes)")
        
        # Step 2: Check first few lines
        with open(temp_path, 'r') as f:
            first_lines = f.readlines()[:3]
            print(f"✓ First few lines: {[line.strip() for line in first_lines]}")
        
        # Step 3: Try pandas read
        try:
            df = pd.read_csv(temp_path)
            print(f"✓ Pandas read successful: shape {df.shape}")
            print(f"✓ Columns: {list(df.columns)}")
            
            # Step 4: Create documents (like the app does)
            documents = []
            for idx, row in df.iterrows():
                row_text = " ".join([f"{col}: {val}" for col, val in row.items()])
                doc = type('Document', (), {
                    'page_content': row_text,
                    'metadata': {
                        'filename': filename,
                        'row_index': idx,
                        'file_type': 'csv',
                        'content_type': 'data_row'
                    }
                })()
                documents.append(doc)
            
            print(f"✓ Created {len(documents)} documents")
            print(f"✓ First document content: {documents[0].page_content[:100]}...")
            
        except Exception as csv_error:
            print(f"✗ Error reading CSV: {csv_error}")
            return False
            
    except Exception as e:
        print(f"✗ Error in file process: {e}")
        return False
    finally:
        # Clean up
        if os.path.exists(temp_path):
            os.remove(temp_path)
            print("✓ Cleaned up temporary file")
    
    return True

if __name__ == "__main__":
    success = test_csv_upload_process()
    print(f"\n{'✓ Test PASSED' if success else '✗ Test FAILED'}") 