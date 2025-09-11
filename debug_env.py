#!/usr/bin/env python3
"""
Debug script to check environment variables in Render
"""
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

print("=== ENVIRONMENT VARIABLES DEBUG ===")
print(f"APCA_API_KEY_ID exists: {'APCA_API_KEY_ID' in os.environ}")
print(f"APCA_API_SECRET_KEY exists: {'APCA_API_SECRET_KEY' in os.environ}")
print(f"APCA_API_BASE_URL: {os.environ.get('APCA_API_BASE_URL', 'NOT SET')}")

# Show first few chars of API key if it exists (for debugging)
api_key = os.environ.get('APCA_API_KEY_ID')
if api_key:
    print(f"API Key starts with: {api_key[:8]}...")
else:
    print("API Key: NOT FOUND")

secret_key = os.environ.get('APCA_API_SECRET_KEY') 
if secret_key:
    print(f"Secret Key starts with: {secret_key[:8]}...")
else:
    print("Secret Key: NOT FOUND")

print("\nAll environment variables:")
for key, value in sorted(os.environ.items()):
    if 'APCA' in key or 'ALPACA' in key:
        print(f"{key}: {value[:10]}..." if value else f"{key}: EMPTY")

print("=== END DEBUG ===")