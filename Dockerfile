# ---------------------------------------------------------------------------
# Container image for the AI Customer Complaint & Case Processing System.
#
# It runs the web version (streamlit_app.py). The batch version still works
# inside the same image if you ever want it:
#
#     docker run --rm -e OPENAI_API_KEY=sk-... complaint-processor python app.py
#
# Build and run locally:
#
#     docker build -t complaint-processor .
#     docker run -p 8501:8501 -e OPENAI_API_KEY=sk-your-key complaint-processor
#
# Then open http://localhost:8501
#
# The API key is NEVER copied into the image. It is passed in at run time,
# which is also how Azure and AWS supply it in production.
# ---------------------------------------------------------------------------

FROM python:3.11-slim

# Python housekeeping:
#   PYTHONDONTWRITEBYTECODE - do not litter the image with .pyc files
#   PYTHONUNBUFFERED        - print log lines immediately, so the cloud
#                             platform's log stream shows them live
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install the dependencies first, in their own layer. Docker caches this
# layer, so changing the application code does not reinstall everything.
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Now the application itself.
COPY . .

# Run as a non-root user. If the container is ever compromised, the attacker
# does not get root inside it.
RUN useradd --create-home --shell /bin/bash appuser \
    && chown -R appuser:appuser /app
USER appuser

# Streamlit's default port. Azure and AWS override it by setting $PORT.
ENV PORT=8501
EXPOSE 8501

# The platform's health check hits this path, which Streamlit serves itself.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:' + os.environ.get('PORT', '8501') + '/_stcore/health')"

# Shell form on purpose, so that $PORT is expanded at start-up.
#
#   --server.address=0.0.0.0   listen on every interface, not just localhost,
#                              otherwise the platform cannot reach the app
#   --server.headless=true     do not try to open a browser or ask for an email
#   --server.enableCORS=false  the app sits behind the platform's HTTPS proxy
#   --server.enableXsrfProtection=false
#                              same reason; without this some proxies break
#                              the websocket and the page never finishes loading
CMD streamlit run streamlit_app.py \
    --server.port=$PORT \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --server.enableCORS=false \
    --server.enableXsrfProtection=false \
    --browser.gatherUsageStats=false
