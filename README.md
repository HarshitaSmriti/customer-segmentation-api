# Customer Segmentation API (Training Project)

This repository contains a training project focused on deploying a Machine Learning model using a professional MLOps pipeline. The project serves a **Latent Dirichlet Allocation (LDA)** model via a Flask API, automated through CircleCI and hosted on AWS.

## 🔗 Live Project Links

* **Live API Endpoint:** [http://customer-segmentation-api-env.eba-7xpxwfyv.us-east-1.elasticbeanstalk.com/](http://customer-segmentation-api-env.eba-7xpxwfyv.us-east-1.elasticbeanstalk.com/)
* **GitHub Repository:** [https://github.com/HarshitaSmriti/customer-segmentation-api](https://www.google.com/search?q=https://github.com/HarshitaSmriti/customer-segmentation-api)

## 🏗️ Technical Stack

* **ML Model:** Scikit-learn (LDA for customer clustering)
* **API Framework:** Flask (Python)
* **Database:** Apache Cassandra
* **CI/CD:** CircleCI
* **Infrastructure:** AWS Elastic Beanstalk (Dockerized)

## 🛠️ CI/CD Workflow

The project implements a fully automated pipeline:

1. **Build & Test:** Installs Python dependencies and verifies the environment.
2. **Integration Testing:** Uses a Dockerized Cassandra instance to test database connectivity.
3. **Automated Deployment:** On every push to the `main` branch, CircleCI triggers the AWS EB CLI to update the production environment.

## 📡 API Usage

### Status Check

**URL:** `GET /`
**Response:**

```json
{
  "status": "ok",
  "message": "Customer Segmentation API running (Flask)"
}

```

## 💻 Local Setup

To run this project locally with Docker:

```bash
# 1. Clone the repository
git clone https://github.com/HarshitaSmriti/customer-segmentation-api.git

# 2. Build the Docker image
docker build -t segmentation-api .

# 3. Run the container
docker run -p 5000:5000 segmentation-api

```

---

**Project Author:** Harshita Smriti
