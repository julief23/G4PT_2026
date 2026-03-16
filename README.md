# AGAPE_2026
AGAPE is a cheminformatics platform that uses molecular descriptors and machine learning models to predict whether small molecules can stabilize G-quadruplex DNA structures. Identifying such ligands is important because G-quadruplexes play key regulatory roles in genomic regions associated with cancer and other diseases. By enabling rapid in-silico screening of candidate compounds, AGAPE helps prioritize molecules for experimental validation and supports the discovery of new therapeutic agents.

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
python manage.py makemigrations
```

```
python manage.py migrate
```

---

## Run the development server

```
python manage.py runserver
```

Open your browser at:

http://127.0.0.1:8000/

---

## Applications

- **frontend**: landing page and navigation
- **simulator**: run a prediction, display results, contact us, FAQ
---
