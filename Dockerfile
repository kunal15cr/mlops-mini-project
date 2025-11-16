FROM python:3.9

WORKDIR /app

COPY flask_app/requirements.txt /app/

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Ensure nltk data is installed to a known global path and available at runtime
ENV NLTK_DATA=/usr/local/share/nltk_data
RUN mkdir -p ${NLTK_DATA} && \
    python -m nltk.downloader -d ${NLTK_DATA} stopwords wordnet

COPY flask_app/ /app/

COPY models/vectorizer.pkl /app/models/vectorizer.pkl

EXPOSE 5000

CMD ["gunicorn", "-b", "0.0.0.0:5000", "python", "app.py" ]