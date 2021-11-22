FROM tensorflow/tensorflow
WORKDIR /root

ARG GITHUB_OWNER_ARG
ENV GITHUB_OWNER=${GITHUB_OWNER_ARG}

ARG GITHUB_REPO_ARG
ENV GITHUB_REPO=${GITHUB_REPO_ARG}

ARG GITHUB_WORKFLOW_ARG
ENV GITHUB_WORKFLOW=${GITHUB_WORKFLOW_ARG}

ARG GITHUB_TOKEN_ARG
ENV GITHUB_TOKEN=${GITHUB_TOKEN_ARG}

ARG GCS_BUCKET_ARG
ENV GCS_BUCKET=${GCS_BUCKET_ARG}

ARG MODEL_TAG_ARG
ENV MODEL_TAG=${MODEL_TAG_ARG}

ARG WEB_HOOK_URL_ARG
ENV WEB_HOOK_URL=${WEB_HOOK_URL_ARG}

# Copies the trainer code
COPY utils.py /root/utils.py
COPY train.py /root/mnist.py
COPY data/ /root/data/
RUN ls -la /root/data/*

RUN pip install pillow
RUN pip install numpy

# Sets up the entry point to invoke the trainer.
ENTRYPOINT ["python", "mnist.py", "--epochs", "20", "--batchsize", "128"]