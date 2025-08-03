#!/usr/bin/env python3
"""
CLI test script for Excel processing functionality
"""

import pandas as pd
import os
import sys
from pathlib import Path

def create_test_excel_file():
    """Create a test Excel file with multiple sheets"""
    print("📊 Creating test Excel file...")
    
    # Create sample data for different sheets
    sales_data = {
        'Product': ['Laptop', 'Phone', 'Tablet', 'Monitor'],
        'Sales': [1200, 800, 600, 400],
        'Region': ['North', 'South', 'East', 'West'],
        'Date': ['2024-01-15', '2024-01-16', '2024-01-17', '2024-01-18']
    }
    
    inventory_data = {
        'Product': ['Laptop', 'Phone', 'Tablet', 'Monitor', 'Keyboard'],
        'Stock': [50, 100, 75, 30, 200],
        'Warehouse': ['A', 'B', 'A', 'C', 'B'],
        'Last_Updated': ['2024-01-20', '2024-01-20', '2024-01-19', '2024-01-20', '2024-01-18']
    }
    
    customers_data = {
        'Customer_ID': ['C001', 'C002', 'C003', 'C004'],
        'Name': ['John Doe', 'Jane Smith', 'Bob Johnson', 'Alice Brown'],
        'Email': ['john@email.com', 'jane@email.com', 'bob@email.com', 'alice@email.com'],
        'Total_Spent': [2500, 1800, 3200, 950]
    }
    
    # Create DataFrames
    df_sales = pd.DataFrame(sales_data)
    df_inventory = pd.DataFrame(inventory_data)
    df_customers = pd.DataFrame(customers_data)
    
    # Save to Excel file with multiple sheets
    test_file = 'test_data.xlsx'
    with pd.ExcelWriter(test_file, engine='openpyxl') as writer:
        df_sales.to_excel(writer, sheet_name='Sales', index=False)
        df_inventory.to_excel(writer, sheet_name='Inventory', index=False)
        df_customers.to_excel(writer, sheet_name='Customers', index=False)
    
    print(f"✅ Created test file: {test_file}")
    print(f"   - Sales sheet: {len(df_sales)} rows")
    print(f"   - Inventory sheet: {len(df_inventory)} rows")
    print(f"   - Customers sheet: {len(df_customers)} rows")
    
    return test_file

def process_excel_file(file_path):
    """Test the Excel processing function (simplified version)"""
    print(f"\n🔍 Processing Excel file: {file_path}")
    
    try:
        # Read all sheets from the Excel file
        excel_file = pd.ExcelFile(file_path)
        sheet_names = excel_file.sheet_names
        
        print(f"📋 Found {len(sheet_names)} sheets: {sheet_names}")
        
        if len(sheet_names) == 1:
            # Single sheet - read directly
            df = pd.read_excel(file_path)
            print(f"✅ Single sheet processed: {len(df)} rows")
            return df
        else:
            # Multiple sheets - combine all sheets into one document
            print("🔄 Combining multiple sheets...")
            
            # Read all sheets and combine them
            all_dfs = []
            for sheet_name in sheet_names:
                try:
                    df = pd.read_excel(file_path, sheet_name=sheet_name)
                    # Add sheet name as a column to identify the source
                    df['_sheet_name'] = sheet_name
                    all_dfs.append(df)
                    print(f"   ✓ Processed sheet: {sheet_name} ({len(df)} rows)")
                except Exception as sheet_error:
                    print(f"   ⚠ Could not process sheet '{sheet_name}': {str(sheet_error)}")
                    continue
            
            if all_dfs:
                # Combine all dataframes
                combined_df = pd.concat(all_dfs, ignore_index=True)
                print(f"✅ Combined {len(sheet_names)} sheets into one document: {len(combined_df)} total rows")
                
                # Show summary
                print("\n📊 Combined Data Summary:")
                print(f"   Total rows: {len(combined_df)}")
                print(f"   Total columns: {len(combined_df.columns)}")
                print(f"   Columns: {list(combined_df.columns)}")
                
                # Show sheet distribution
                if '_sheet_name' in combined_df.columns:
                    sheet_counts = combined_df['_sheet_name'].value_counts()
                    print("\n📋 Sheet Distribution:")
                    for sheet, count in sheet_counts.items():
                        print(f"   {sheet}: {count} rows")
                
                return combined_df
            else:
                print("❌ No sheets could be processed successfully")
                return None
                
    except Exception as e:
        print(f"❌ Error processing Excel file: {str(e)}")
        return None

def save_processed_data(df, original_file_path):
    """Save the processed combined data to the data folder"""
    if df is None:
        return None
    
    try:
        # Create output filename based on original file
        original_filename = os.path.basename(original_file_path)
        name_without_ext = os.path.splitext(original_filename)[0]
        output_filename = f"processed_{name_without_ext}.csv"
        output_path = os.path.join('data', output_filename)
        
        # Save as CSV (easier to work with than Excel for combined data)
        df.to_csv(output_path, index=False)
        print(f"\n💾 Saved processed data to: {output_path}")
        print(f"   📊 Total rows: {len(df)}")
        print(f"   📋 Total columns: {len(df.columns)}")
        
        # Also save as Excel with multiple sheets (one per original sheet)
        excel_output_filename = f"processed_{name_without_ext}.xlsx"
        excel_output_path = os.path.join('data', excel_output_filename)
        
        with pd.ExcelWriter(excel_output_path, engine='openpyxl') as writer:
            # Save combined data to first sheet
            df.to_excel(writer, sheet_name='Combined_Data', index=False)
            
            # Save individual sheets
            if '_sheet_name' in df.columns:
                for sheet_name in df['_sheet_name'].unique():
                    sheet_data = df[df['_sheet_name'] == sheet_name].copy()
                    # Remove the _sheet_name column for individual sheets
                    sheet_data = sheet_data.drop('_sheet_name', axis=1)
                    # Clean sheet name for Excel (remove invalid characters)
                    clean_sheet_name = "".join(c for c in sheet_name if c.isalnum() or c in (' ', '-', '_'))[:31]
                    sheet_data.to_excel(writer, sheet_name=clean_sheet_name, index=False)
        
        print(f"💾 Saved processed Excel file to: {excel_output_path}")
        print(f"   📊 Combined sheet + {len(df['_sheet_name'].unique())} individual sheets")
        
        return output_path, excel_output_path
        
    except Exception as e:
        print(f"❌ Error saving processed data: {str(e)}")
        return None

def test_data_analysis(df):
    """Test basic data analysis on the combined dataset"""
    if df is None:
        return
    
    print("\n🔍 Data Analysis Test:")
    
    # Basic info
    print(f"   Shape: {df.shape}")
    print(f"   Data types: {df.dtypes.to_dict()}")
    
    # Sample data
    print("\n📋 Sample Data (first 3 rows):")
    print(df.head(3).to_string())
    
    # Check for sheet information
    if '_sheet_name' in df.columns:
        print(f"\n📊 Data by Sheet:")
        for sheet_name in df['_sheet_name'].unique():
            sheet_data = df[df['_sheet_name'] == sheet_name]
            print(f"   {sheet_name}: {len(sheet_data)} rows")
            
            # Show sample from each sheet
            print(f"   Sample from {sheet_name}:")
            sample_cols = [col for col in sheet_data.columns if col != '_sheet_name']
            print(sheet_data[sample_cols].head(2).to_string())
            print()

def main():
    """Main test function"""
    print("🧪 Excel Processing Test")
    print("=" * 50)
    
    # Check if openpyxl is available
    try:
        import openpyxl
        print("✅ openpyxl is available")
    except ImportError:
        print("❌ openpyxl is not installed. Please run: pip install openpyxl")
        return
    
    # Check for command line argument (real file path)
    print(f"🔍 Command line arguments: {sys.argv}")
    
    if len(sys.argv) > 1:
        real_file_path = sys.argv[1]
        print(f"🔍 Checking if file exists: {real_file_path}")
        print(f"🔍 Current working directory: {os.getcwd()}")
        print(f"🔍 File exists: {os.path.exists(real_file_path)}")
        
        if os.path.exists(real_file_path):
            print(f"📁 Using real Excel file: {real_file_path}")
            combined_df = process_excel_file(real_file_path)
            test_data_analysis(combined_df)
            
            # Save the processed data
            save_processed_data(combined_df, real_file_path)
        else:
            print(f"❌ File not found: {real_file_path}")
            print("🔍 Available files in data/ directory:")
            try:
                for file in os.listdir('data/'):
                    print(f"   - {file}")
            except Exception as e:
                print(f"   Error listing data directory: {e}")
            return
    else:
        # Use default file path if no argument provided
        default_file_path = 'data/Telhio_Raw_Data.xlsx'
        print(f"🔍 No argument provided, checking default file: {default_file_path}")
        print(f"🔍 File exists: {os.path.exists(default_file_path)}")
        
        if os.path.exists(default_file_path):
            print(f"📁 Using default Excel file: {default_file_path}")
            combined_df = process_excel_file(default_file_path)
            test_data_analysis(combined_df)
            
            # Save the processed data
            save_processed_data(combined_df, default_file_path)
        else:
            # Create test file if no real file found
            print("📊 No real file found, creating test data...")
            test_file = create_test_excel_file()
            combined_df = process_excel_file(test_file)
            test_data_analysis(combined_df)
            
            # Save the processed test data
            save_processed_data(combined_df, test_file)
            
            # Cleanup test file only
            if os.path.exists(test_file):
                os.remove(test_file)
                print(f"\n🧹 Cleaned up test file: {test_file}")
    
    print("\n✅ Test completed successfully!")

if __name__ == "__main__":
    main() 