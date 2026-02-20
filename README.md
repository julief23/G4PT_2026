# AGAPE_2026
AGAPE web interface

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
└── requirements.txt
```
---

## Requirements

- Python 3.11
- pip
- virtualenv support

---

## Installation

### 1. Create a virtual environment

```
python3.11 -m venv .venv
```

```
python -m venv .venv
```

### 2. Activate the virtual environment

Activate the virtual environment depending on your operating system:

On Linux / macOS:

```
source .venv/bin/activate
```

On Windows (PowerShell / CMD):

```
.venv\Scripts\activate
```

### 3. Upgrade pip and install dependencies

```
python -m pip install --upgrade pip  
python -m pip install -r requirements.txt
```

---
### Apply migrations

```
python src/manage.py makemigrations
```

```
python src/manage.py migrate
```

---

## Load data from fixtures to recreate the DB (the order of loading matters)
```
python src/manage.py loaddata src/accounts/fixtures/users.json
```
```
python src/manage.py loaddata src/accounts/fixtures/accounts.json
```
```
python src/manage.py loaddata src/plasmids/fixtures/public_collections.json
```
```
python src/manage.py loaddata src/browse/fixtures/browse_data.json
```

---

## Run the development server

```
python src/manage.py runserver
```

Open your browser at:

http://127.0.0.1:8000/

---

## Applications

- **frontend**: landing page and navigation
- **simulator**: run a prediction, display results, contact us, FAQ
---
