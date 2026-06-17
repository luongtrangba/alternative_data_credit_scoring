FROM python:3.12-slim AS base

WORKDIR /app

COPY pyproject.toml .
COPY src/ src/
COPY configs/ configs/
COPY data/raw/ data/raw/
COPY data/feature_dictionary.json data/
COPY data/sme_altdata_sample.csv data/

RUN pip install --no-cache-dir .

# Generate data + train model at build time
# (produces artifacts in data/ and t4_training/models/)
RUN python -m scoresight.data_generator \
    && python -m scoresight.features.dsr_calculator \
    && python -m scoresight.training.train

EXPOSE 8000

CMD ["uvicorn", "scoresight.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
