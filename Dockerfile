FROM python:3.12-slim

WORKDIR /processor

# Pinned Python dependencies (nilearn pulls in scipy/scikit-learn), plus the
# Lambda runtime interface client for running as a Lambda function.
COPY requirements.txt /processor/requirements.txt
RUN pip install --no-cache-dir -r /processor/requirements.txt awslambdaric==2.2.1

COPY processor/ /processor/processor/
COPY VERSION /processor/VERSION

ENV PYTHONPATH="/processor" \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    MPLCONFIGDIR=/tmp/matplotlib

# Runs as uid 1000 on the platform; nothing here needs root.
USER 1000

# Runs main() once as a container, or serves processor.handler as a Lambda
# function (see processor/entry.py).
ENTRYPOINT ["python3", "-m", "processor.entry"]
