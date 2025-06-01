# Dockerfile

# 1. Use a slim Python 3.10 base image
FROM python:3.10-slim

# 2. Ensure Python output is unbuffered (so logs appear immediately)
ENV PYTHONUNBUFFERED=1

# 3. Set /app as our working directory
WORKDIR /app

# 4. Copy only requirements.txt first (leverages Docker cache)
COPY requirements.txt .

# 5. Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy the rest of our app code into /app
COPY . .

# 7. Expose port 5000 (Flask default). Render will map its internal $PORT to this.
EXPOSE 5000

# 8. Run the Flask app (which now calls app.run(host="0.0.0.0", port=$PORT))
CMD ["python", "app.py"]
