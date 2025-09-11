#!/usr/bin/env python3
"""
Debug script to check environment variables in Render
"""
import os

print("=== ENVIRONMENT VARIABLES DEBUG ===")
print(f"ALPACA_API_KEY exists: {'ALPACA_API_KEY' in os.environ}")
print(f"ALPACA_SECRET_KEY exists: {'ALPACA_SECRET_KEY' in os.environ}")
print(f"ALPACA_BASE_URL: {os.environ.get('ALPACA_BASE_URL', 'NOT SET')}")

# Show first few chars of API key if it exists (for debugging)
api_key = os.environ.get('ALPACA_API_KEY')
if api_key:
    print(f"API Key starts with: {api_key[:8]}...")
else:
    print("API Key: NOT FOUND")

secret_key = os.environ.get('ALPACA_SECRET_KEY') 
if secret_key:
    print(f"Secret Key starts with: {secret_key[:8]}...")
else:
    print("Secret Key: NOT FOUND")

print("\nAll environment variables:")
for key, value in sorted(os.environ.items()):
    if 'ALPACA' in key:
        print(f"{key}: {value[:10]}..." if value else f"{key}: EMPTY")

print("=== END DEBUG ===")