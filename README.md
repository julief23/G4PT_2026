# AGAPE_2026

AGAPE is a cheminformatics platform that uses molecular descriptors and machine learning models to predict whether small molecules can stabilize G-quadruplex DNA structures. Identifying such ligands is important because G-quadruplexes play key regulatory roles in genomic regions associated with cancer and other diseases. By enabling rapid in silico screening of candidate compounds, AGAPE helps prioritize molecules for experimental validation and supports the discovery of new therapeutic agents.

- Julie FARES

---

Django-based web application for AGAPE: AI-Powered Affinity Predictor for G4-Binders.

---

## Project structure

```text
AGAPE_2026/
├── agape_project/
│   ├── manage.py
│   ├── config/
│   │   ├── __init__.py
│   │   ├── settings.py
│   │   ├── urls.py
│   │   ├── asgi.py
│   │   └── wsgi.py
│   │
│   ├── simulator/
│   └── frontend/
│
├── Dockerfile
├── environment.yml
├── docker-compose.yml
├── .dockerignore
├── .env.example
└── README.md
```

---

## Requirements

- Python 3.11
- pip
- virtual environment support

---

## Installation

### 1. Create a virtual environment

```bash
python3.11 -m venv .venv
```

or

```bash
python -m venv .venv
```

### 2. Activate the virtual environment

Linux / macOS:

```bash
source .venv/bin/activate
```

Windows:

```bash
.venv\Scripts\activate
```

### 3. Upgrade pip and install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

---

## Apply migrations

```bash
cd agape_project
python manage.py makemigrations
python manage.py migrate
```

---

## Run the development server

```bash
python manage.py runserver
```

Open your browser at:

http://127.0.0.1:8000/

---

## Docker deployment

### Build the Docker image

```bash
docker build -t agape_web .
```

### Run the Docker container

```bash
docker run --env-file .env -p 8000:8000 agape_web
```

Open your browser at:

http://localhost:8000/

---

## Environment variables

Create a `.env` file based on `.env.example`:

```env
DJANGO_SECRET_KEY=replace-with-a-secure-key
DJANGO_DEBUG=False
DJANGO_ALLOWED_HOSTS=localhost,127.0.0.1,0.0.0.0
```

---

## Applications

- **frontend**: landing page and navigation
- **simulator**: run a prediction, display results, contact us, FAQ

---

## Notes

- The application runs with Gunicorn inside Docker.
- Static files are collected automatically during Docker build.
- The Conda environment is defined in `environment.yml`.
- For production deployment, replace `DJANGO_ALLOWED_HOSTS` with your server IP or domain name.