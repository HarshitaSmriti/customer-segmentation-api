import pytest
from app import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_home_route(client):
    response = client.get('/')
    assert response.status_code == 200
    assert response.get_json()['status'] == 'ok'

def test_train_route_ci(client):
    response = client.get('/train')
    assert response.status_code == 200
    assert "CI mode" in response.get_json()['message']