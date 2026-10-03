FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY migrations ./migrations
COPY alembic.ini ./
COPY tests/fixtures ./tests/fixtures
RUN pip install --no-cache-dir .
# Offline demo: docker run <image> sh -c "alpha-lab init --db /tmp/demo.db && alpha-lab --help"
CMD ["alpha-lab", "--help"]
