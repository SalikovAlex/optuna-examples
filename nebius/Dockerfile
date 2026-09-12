FROM python:3.12-slim@sha256:2fe5997d249a808b8eeea52c58a1dbffbba28754dc11699ef5c029f2d818ce79
WORKDIR /example
COPY requirements-worker.txt constraints-worker.txt ./
# Linux PyPI PyTorch wheels include their CUDA userspace dependencies.
# The cloud runtime must provide a compatible NVIDIA driver.
RUN pip install --no-cache-dir -r requirements-worker.txt && pip check
COPY common.py journal.py worker.py prepare_data.py LICENSE FASHIONMNIST-LICENSE ./
RUN python prepare_data.py /data
ENV PYTHONUNBUFFERED=1 NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=compute,utility
CMD ["python", "/example/worker.py", "/example/assignment.json"]
