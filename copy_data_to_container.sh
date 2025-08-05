#!/bin/bash

echo "📁 Copying data files to Docker container with updated names..."

# Create data directory in container if it doesn't exist
docker exec abi-app mkdir -p /app/data

# Copy Excel files with cleaner names
echo "📊 Copying Excel files..."
docker cp data/Telhio_Raw_Data.xlsx abi-app:/app/data/telhio_raw_data.xlsx
docker cp data/Telhio_GA4_User_Traffic_Data.xlsx abi-app:/app/data/telhio_ga4_traffic_data.xlsx

# Copy CSV files
echo "📈 Copying CSV files..."
docker cp data/fake_ad_campaigns.csv abi-app:/app/data/ad_campaigns.csv
docker cp data/fake_ad_campaigns_20.csv abi-app:/app/data/ad_campaigns_sample.csv
docker cp data/sample_data.csv abi-app:/app/data/sample_data.csv

# Copy text file
echo "📄 Copying text file..."
docker cp data/my_output.txt abi-app:/app/data/output.txt

echo "✅ All files copied successfully!"

# Verify the files are in the container
echo "🔍 Verifying files in container:"
docker exec abi-app ls -la /app/data/ 