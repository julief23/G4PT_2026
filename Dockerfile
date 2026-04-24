FROM condaforge/miniforge3:latest

WORKDIR /app

COPY environment.yml .
RUN mamba env create -f environment.yml && mamba clean -afy

COPY . .

WORKDIR /app/agape_project

RUN /opt/conda/envs/agapenv/bin/python manage.py collectstatic --noinput

EXPOSE 8000

CMD ["/opt/conda/envs/agapenv/bin/gunicorn", "config.wsgi:application", "--bind", "0.0.0.0:8000"]