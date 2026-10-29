# Stage 2: Install dependencies and built package to a virtual environment. This
# means we only have the main dependencies installed at this stage and the package
# itself.
FROM python:3.12-slim AS deps

RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

ENV POETRY_HOME="/opt/poetry"
ENV POETRY_VERSION=1.8.3
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="$POETRY_HOME/bin:$PATH"

WORKDIR /app

RUN poetry config virtualenvs.create true && \
    poetry config virtualenvs.in-project true

COPY pyproject.toml poetry.lock ./

RUN poetry install --only=main --no-root

COPY src/pm25ml/ ./pm25ml/
COPY README.md ./

RUN poetry build && poetry run pip install --no-deps dist/*.whl

# Stage 3: Minimal runtime image
FROM python:3.12-slim AS runtime

ARG PM25ML_COMMIT_SHA=
ENV PM25ML_COMMIT_SHA=${PM25ML_COMMIT_SHA}

RUN apt-get update && apt-get install -y libgomp1 && rm -rf /var/lib/apt/lists/*

# Security: create a non-root user to run the application
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Create working directory and give ownership to the non-root user
WORKDIR /app
RUN chown appuser:appuser /app
VOLUME /app/.config/gcloud

# Copy only what's needed from previous stages
COPY --from=deps /app/.venv/lib/python3.12/site-packages/ /usr/local/lib/python3.12/site-packages/
COPY --from=deps /app/.venv/lib/*.so* /usr/local/lib/
COPY ./assets/ ./assets/

# Set the user to the non-root user, this ensures that the app can't write to the earlier
# stages' files but can read them.
USER appuser
