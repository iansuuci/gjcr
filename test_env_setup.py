#!/usr/bin/env python3
"""Quick test to verify .env file is set up correctly."""

try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✓ python-dotenv loaded successfully")
except ImportError:
    print("✗ python-dotenv not installed. Run: pip3 install python-dotenv")
    exit(1)

import os

# Check for API key in various possible variable names
api_key = os.getenv("GROK_API_KEY") or os.getenv("XAI_API_KEY") or os.getenv("API_KEY_NAME")

if api_key is None:
    print("✗ No API key found in environment variables")
    print("  Please set one of these in your .env file:")
    print("    GROK_API_KEY=your_key_here")
    print("    XAI_API_KEY=your_key_here")
    print("    API_KEY_NAME=your_key_here")
    exit(1)

if api_key == "your_secret_key_here":
    print("✗ API key appears to be a placeholder")
    print("  Please replace 'your_secret_key_here' with your actual Grok API key in .env file")
    exit(1)

print(f"✓ API key found: {api_key[:10]}...{api_key[-4:] if len(api_key) > 14 else '****'}")
print("✓ Environment setup looks good!")

