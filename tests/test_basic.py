import pytest
import os
from app import app

@pytest.fixture
def client():
    # Set up a testing client for the Flask app
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_home_route(client):
    """Confirm the home page is accessible."""
    response = client.get('/')
    assert response.status_code == 200
    assert response.json['status'] == 'ok'

def test_train_route_ci_mode(client):
    """Confirm the train route responds correctly in CI mode."""
    # Since app.py checks os.getenv("CI"), it should return a 200 even if it skips DB logic
    response = client.get('/train')
    assert response.status_code == 200
    assert "CI mode" in response.json['message']