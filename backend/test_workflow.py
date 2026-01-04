"""
Simple test script to verify the workflow
Run this after starting the server to test all scenarios
"""

import requests
import json

BASE_URL = "http://localhost:8000"

def test_health():
    """Test health endpoint"""
    print("Testing health endpoint...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}\n")

def test_valid_request():
    """Test valid benefit request"""
    print("Testing valid request (Student with Classic card)...")
    payload = {
        "card_number": "4111-****-****-1111",
        "user_context": "student",
        "preferred_language": "en",
        "location": "Chennai"
    }
    response = requests.post(f"{BASE_URL}/benefits", json=payload)
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}\n")

def test_traveler_infinite():
    """Test traveler with Infinite card"""
    print("Testing traveler with Infinite card...")
    payload = {
        "card_number": "4222-****-****-2222",
        "user_context": "traveler",
        "preferred_language": "en",
        "location": "Mumbai"
    }
    response = requests.post(f"{BASE_URL}/benefits", json=payload)
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}\n")

def test_tamil_translation():
    """Test Tamil translation"""
    print("Testing Tamil translation...")
    payload = {
        "card_number": "4111-****-****-1111",
        "user_context": "student",
        "preferred_language": "ta",
        "location": "Chennai"
    }
    response = requests.post(f"{BASE_URL}/benefits", json=payload)
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}\n")

def test_invalid_card():
    """Test invalid card format (unmasked)"""
    print("Testing invalid card format (unmasked)...")
    payload = {
        "card_number": "4111-1234-5678-1111",  # Unmasked - should fail
        "user_context": "student"
    }
    try:
        response = requests.post(f"{BASE_URL}/benefits", json=payload)
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}\n")
    except Exception as e:
        print(f"Error: {e}\n")

def test_unsupported_bin():
    """Test unsupported BIN"""
    print("Testing unsupported BIN...")
    payload = {
        "card_number": "4999-****-****-9999",  # Unsupported BIN
        "user_context": "student"
    }
    response = requests.post(f"{BASE_URL}/benefits", json=payload)
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}\n")

if __name__ == "__main__":
    print("=" * 60)
    print("Visa Benefits Workflow Test Suite")
    print("=" * 60)
    print()
    
    try:
        test_health()
        test_valid_request()
        test_traveler_infinite()
        test_tamil_translation()
        test_invalid_card()
        test_unsupported_bin()
        
        print("=" * 60)
        print("All tests completed!")
        print("=" * 60)
    except requests.exceptions.ConnectionError:
        print("ERROR: Cannot connect to server. Make sure it's running:")
        print("  python backend/main.py")
    except Exception as e:
        print(f"ERROR: {e}")

