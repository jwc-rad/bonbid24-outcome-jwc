FROM python:3.9-slim

RUN groupadd -r user && useradd -m --no-log-init -r -g user user

RUN apt-get update && apt-get install -y git

RUN mkdir -p /opt/app /input /output \
    && chown user:user /opt/app /input /output

USER user
WORKDIR /opt/app

ENV PATH="/home/user/.local/bin:${PATH}"

RUN python -m pip install --user -U pip && python -m pip install --user pip-tools

# COPY sources
COPY --chown=user:user ./config /opt/app/config
COPY --chown=user:user ./data /opt/app/data
COPY --chown=user:user ./model /opt/app/model
COPY --chown=user:user ./networks /opt/app/networks
COPY --chown=user:user ./src /opt/app/src
COPY --chown=user:user ./utils /opt/app/utils
COPY --chown=user:user ./weights /opt/app/weights

COPY --chown=user:user requirements.txt train.py process.py /opt/app/

# install requirements
RUN python -m piptools sync requirements.txt

ENTRYPOINT [ "python", "-m", "process" ]