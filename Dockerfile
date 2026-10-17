# 1. Use a lightweight official Python 3.11 Linux image (~150MB instead of 1GB)
ARG PIP_TIMEOUT=300
FROM python:3.11-slim

RUN pip install --upgrade pip
# 2. Prevent Python from writing .pyc files and buffer stdout for real-time Docker logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 3. Set the working directory inside the container
WORKDIR /app

# 4. Copy ONLY requirements.txt first to leverage Docker layer caching
COPY requirements.txt .

# 5. Install dependencies inside the container
RUN pip install --no-cache-dir --default-timeout=300 -r requirements.txt

# 6. Copy the rest of your project (filtered by .dockerignore)
COPY . .

# 7. Expose Streamlit's default port
EXPOSE 8501

# 8. Add a production healthcheck so cloud orchestrators know if the app crashes
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# 9. Run Streamlit bound to 0.0.0.0 so traffic outside the container can reach it
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]