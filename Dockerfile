FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# LightGBM a besoin de la bibliotheque OpenMP (absente des images slim)
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependances d'abord (couche mise en cache tant que requirements.txt ne change pas)
ENV PIP_DEFAULT_TIMEOUT=300 \
    PIP_RETRIES=10 \
    PIP_NO_CACHE_DIR=1

# Installation en plusieurs couches : chaque groupe reussi reste en cache,
# ce qui permet de reprendre apres une coupure reseau.
# (liste alignee sur requirements.txt, sans pytest/httpx qui servent seulement aux tests)
RUN pip install duckdb
RUN pip install numpy
RUN pip install pandas pyarrow
RUN pip install scikit-learn lightgbm
RUN pip install fastapi uvicorn
RUN pip install plotly
RUN pip install streamlit

# Code uniquement : donnees, modeles et rapports sont montes en volumes (docker-compose.yml)
COPY src/ src/
COPY api/ api/
COPY app/ app/
COPY sql/ sql/

EXPOSE 8000 8501
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]