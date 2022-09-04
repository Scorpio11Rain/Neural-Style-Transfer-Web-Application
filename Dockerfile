FROM python:3.8-slim-buster
COPY . /NST_web_app
WORKDIR /NST_web_app
RUN pip install -r requirements.txt
EXPOSE 8501
ENTRYPOINT ["streamlit","run"]
CMD ["main.py"]