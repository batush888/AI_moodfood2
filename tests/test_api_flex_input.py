import pytest
from fastapi.testclient import TestClient
from api.enhanced_routes import app

client = TestClient(app)

def test_accepts_text_input():
    """Test that the API accepts the legacy text_input field."""
    resp = client.post("/enhanced-recommend", json={"text_input": "I want ramen"})
    assert resp.status_code == 200

def test_accepts_query():
    """Test that the API accepts the new query field."""
    resp = client.post("/enhanced-recommend", json={"query": "I want ramen"})
    assert resp.status_code == 200

def test_prefers_text_input_when_both_present():
    """Test that text_input is preferred when both fields are present."""
    resp = client.post("/enhanced-recommend", json={
        "text_input": "I want ramen",
        "query": "I want sushi"
    })
    assert resp.status_code == 200

def test_rejects_missing_text():
    """Test that the API rejects requests with no text input."""
    resp = client.post("/enhanced-recommend", json={"foo": "bar"})
    assert resp.status_code == 422
    assert "Either 'text_input' or 'query' is required" in resp.text

def test_rejects_empty_text():
    """Test that the API rejects requests with empty text."""
    resp = client.post("/enhanced-recommend", json={"text_input": ""})
    assert resp.status_code == 422
    assert "Either 'text_input' or 'query' is required" in resp.text

def test_rejects_empty_query():
    """Test that the API rejects requests with empty query."""
    resp = client.post("/enhanced-recommend", json={"query": ""})
    assert resp.status_code == 422
    assert "Either 'text_input' or 'query' is required" in resp.text

def test_rejects_whitespace_only():
    """Test that the API rejects requests with only whitespace."""
    resp = client.post("/enhanced-recommend", json={"text_input": "   "})
    assert resp.status_code == 422
    assert "Either 'text_input' or 'query' is required" in resp.text

def test_accepts_with_optional_fields():
    """Test that the API accepts requests with optional fields."""
    resp = client.post("/enhanced-recommend", json={
        "query": "I want ramen",
        "image_base64": "base64data",
        "user_context": {"mood": "happy"}
    })
    assert resp.status_code == 200

if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v"])
