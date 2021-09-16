FROM pytorch/pytorch:1.9.0-cuda10.2-cudnn7-runtime

RUN apt-get update
RUN apt-get install -y libglib2.0-0 libsm6 libxrender1 libxext6 libgl1-mesa-dev

RUN pip install --upgrade pip
RUN pip install poetry
COPY pyproject.toml ./
COPY poetry.lock ./
RUN poetry config virtualenvs.create false
RUN poetry install --no-root --no-dev
