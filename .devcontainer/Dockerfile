# read the doc: https://huggingface.co/docs/hub/spaces-sdks-docker
# you will also find guides on how best to write your Dockerfile

FROM python:3.11

ENV VIRTUAL_ENV=/opt/venv
RUN python -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# The two following lines are requirements for the Dev Mode to be functional
# Learn more about the Dev Mode at https://huggingface.co/dev-mode-explorers
RUN useradd -m -u 1000 user
WORKDIR /app

COPY --chown=user ./requirements.txt requirements.txt
RUN apt-get -y update
RUN apt-get -y upgrade
#RUN apt-get install -y sqlite3 libsqlite3-dev
RUN pip install --no-cache-dir --upgrade -r requirements.txt
#EXPOSE 1433

COPY --chown=user . /app
#RUN /usr/bin/sqlite3 /app/database/source_cache.sqlite3
CMD ["uvicorn", "rag_app.main:app", "--host", "0.0.0.0", "--port", "7860"]