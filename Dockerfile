FROM python:3.8

COPY . .
RUN pip install flask
RUN pip install -e .

EXPOSE 5000
ENTRYPOINT ["python"]

CMD ["app.py"]
