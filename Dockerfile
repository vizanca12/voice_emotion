# Hugging Face Space (Docker SDK). Free Spaces are CPU-only, so the CPU wheels
# are what we install -- they are ~200 MB instead of the 2.5 GB CUDA build.
FROM python:3.11-slim

RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/home/user/.cache

WORKDIR /home/user/app

COPY --chown=user requirements-deploy.txt .
RUN pip install --no-cache-dir \
        torch==2.5.1 torchaudio==2.5.1 torchvision==0.20.1 \
        --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements-deploy.txt

COPY --chown=user src/ ./src/
COPY --chown=user app/ ./app/
COPY --chown=user checkpoints/ ./checkpoints/

# Hugging Face fixes the port at 7860; Cloud Run injects its own $PORT (8080)
# and ignores EXPOSE. Defaulting to 7860 keeps one image working on both.
# The shell form is required here -- the exec form does not expand variables.
ENV PORT=7860
EXPOSE 7860
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
