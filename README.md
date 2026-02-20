# Customer Segmentation API

Production deployment guide for a Flask-based ML inference API on AWS EC2.

## Overview

This service predicts customer segments using a trained ML pipeline:

1. preprocessor.pkl transforms raw input features.
2. lda.pkl reduces feature space.
3. kmeans_model.pkl predicts unsupervised segment.
4. classifier_model.pkl predicts supervised segment.

Optional persistence is implemented with Cassandra (db.py).

## Tech Stack

- Python 3.11+
- Flask API
- scikit-learn + joblib (model loading/inference)
- Apache Cassandra (optional write path)
- Gunicorn (production WSGI server)
- Nginx (reverse proxy on EC2)
- CircleCI (CI/CD)

## Repository Layout

text
.
├── app.py
├── db.py
├── requirements.txt
├── Dockerfile
├── tests/
├── classifier_model.pkl
├── kmeans_model.pkl
├── lda.pkl
└── preprocessor.pkl

## API Endpoints

### GET /

Health/info endpoint.

Example response:
json
{
"status": "ok",
"message": "Customer Segmentation API (Flask)"
}

### GET /train

Placeholder endpoint (returns CI skip message when CI=true).

### POST /predict

Runs preprocessing + LDA + clustering/classification.

Request (JSON):
json
{
"Age": 30,
"Education": 2,
"Marital Status": 1,
"Parental Status": 0,
"Children": 1,
"Income": 65000,
"Total_Spending": 1200,
"Days_as_Customer": 300,
"Recency": 15,
"Wines": 200,
"Fruits": 50,
"Meat": 120,
"Fish": 40,
"Sweets": 30,
"Gold": 10,
"Web": 5,
"Catalog": 2,
"Store": 8,
"Discount Purchases": 3,
"Total Promo": 1,
"NumWebVisitsMonth": 7,
"Family_Size": 3,
"Spending_per_Day": 4,
"Digital_Engagement": 1,
"Offline_Engagement": 1,
"Discount_Ratio": 0.1,
"Premium_Ratio": 0.2,
"Freshness_Score": 0.7,
"Variety_Index": 5
}

Response:
json
{
"kmeans_cluster": 1,
"predicted_cluster": 2,
"status": "success"
}

## Local Development

bash
python3 -m venv .venv
. .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python app.py

Service runs on http://0.0.0.0:5000.

## Running Tests

bash
. .venv/bin/activate
pytest -q

## AWS EC2 Production Deployment

### 1. Provision EC2

Use Ubuntu 22.04/24.04 and allow inbound:

- 22 (SSH) from your IP
- 80 (HTTP) from internet
- 443 (HTTPS) from internet

### 2. Install System Packages

SSH to EC2 and run:

bash
sudo apt update
sudo apt install -y python3.11 python3.11-venv python3-pip nginx git

### 3. Deploy Application

bash
cd /opt
sudo git clone <https://github.com/HarshitaSmriti/customer-segmentation-api.git> customer-segmentation-api
sudo chown -R ubuntu:ubuntu /opt/customer-segmentation-api
cd /opt/customer-segmentation-api

python3.11 -m venv .venv
. .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

### 4. Create systemd Service

Create /etc/systemd/system/customer-segmentation-api.service:

ini
[Unit]
Description=Customer Segmentation API
After=network.target

[Service]
User=ubuntu
Group=ubuntu
WorkingDirectory=/opt/customer-segmentation-api
Environment=PYTHONUNBUFFERED=1
Environment=CASSANDRA_HOST=localhost
Environment=CASSANDRA_PORT=9042
ExecStartPre=/opt/customer-segmentation-api/.venv/bin/python -c "import app; app.init_resources()"
ExecStart=/opt/customer-segmentation-api/.venv/bin/gunicorn -w 2 -b 127.0.0.1:8000 app:app
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target

Enable/start:

bash
sudo systemctl daemon-reload
sudo systemctl enable customer-segmentation-api
sudo systemctl restart customer-segmentation-api
sudo systemctl status customer-segmentation-api

### 5. Configure Nginx Reverse Proxy

Create /etc/nginx/sites-available/customer-segmentation-api:

nginx
server {
listen 80;
server*name *;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

}

Enable config:

bash
sudo ln -s /etc/nginx/sites-available/customer-segmentation-api /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t
sudo systemctl restart nginx

### 6. Optional HTTPS (Let's Encrypt)

If using a domain:

bash
sudo apt install -y certbot python3-certbot-nginx
sudo certbot --nginx -d your-domain.com -d www.your-domain.com

## Cassandra Notes

- Cassandra is optional for prediction API response.
- If Cassandra is unavailable, prediction still returns response; only save operation fails internally.
- For production, use Amazon Keyspaces or a managed Cassandra cluster and set:
  - CASSANDRA_HOST
  - CASSANDRA_PORT

## CI/CD

CircleCI pipeline in .circleci/config.yml:

1. Starts Python and Cassandra service containers.
2. Installs dependencies.
3. Waits for Cassandra readiness.
4. Runs tests.
5. Deploys on main branch (current workflow targets Elastic Beanstalk).

If deploying only to EC2, disable/replace the EB deploy job.

## Troubleshooting

- 503 Models not loaded or prediction errors:
  - verify all .pkl files exist in project root.
  - verify scikit-learn compatibility with trained artifacts.
- ModuleNotFoundError:
  - ensure .venv is active and pip install -r requirements.txt completed.
- Service not starting:
  - sudo journalctl -u customer-segmentation-api -n 200 --no-pager
- Nginx issues:
  - sudo nginx -t

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/HarshitaSmriti/customer-segmentation-api/blob/main/DataModel.ipynb) Open this Notebook to know more about the project working

**Project Author:** Aishwarya Shree,Harshita Smriti
