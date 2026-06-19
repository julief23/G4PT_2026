<p align="center">
  <img src="G4PT_project/frontend/static/frontend/images/G4PT_python.png" width="700">
</p>

# G4PT_V1
G4PT is an AI-powered cheminformatics platform that combines molecular generation, validation, and machine learning models to design and prioritize small molecules with potential G-quadruplex DNA-stabilizing activity. By integrating user-defined constraints, generative models, and AGAPE-based prediction, G4PT supports the rapid in silico exploration of candidate ligands and helps guide experimental validation for the discovery of new therapeutic agents.

- Julie FARES

---

Django-based web application for G4PT: AI-Powered Generative Model for G4-Binders.

---

## Project structure

```text
G4PT_2026/
├── G4PT_project/
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


## Installation - Docker deployment

### Docker and compose Versions

Docker Engine version: `26.1.3`

docker-compose version: `1.25.0`

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

## First-time setup

Before running the application, create a `.env` file in the project root:
Create a `.env` file based on `.env.example`:

```env
DJANGO_SECRET_KEY=replace-with-a-secure-key
DJANGO_DEBUG=False
DJANGO_ALLOWED_HOSTS=localhost,127.0.0.1,0.0.0.0
```

You can generate a secret key with:

```bash
python -c "import secrets; print(secrets.token_urlsafe(50))"
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

## Citation

If you use G4PT in academic work, please cite