import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from urllib.parse import parse_qs, urlparse
import json
import pandas as pd
from datetime import datetime
import uuid
import os
from typing import Callable, Any
from wsgiref.simple_server import make_server

nltk.download('vader_lexicon', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('stopwords', quiet=True)

adj_noun_pairs_count = {}
sia = SentimentIntensityAnalyzer()
stop_words = set(stopwords.words('english'))

reviews = pd.read_csv('data/reviews.csv').to_dict('records')

class ReviewAnalyzerServer:
    def __init__(self) -> None:
        self.load_reviews()

    def load_reviews(self):
        self.reviews = reviews

    def reset_reviews(self):
        self.load_reviews()

    def analyze_sentiment(self, review_body):
        sentiment_scores = sia.polarity_scores(review_body)
        return sentiment_scores

    def __call__(self, environ: dict[str, Any], start_response: Callable[..., Any]) -> bytes:
        """
        The environ parameter is a dictionary containing some useful
        HTTP request information such as: REQUEST_METHOD, CONTENT_LENGTH, QUERY_STRING,
        PATH_INFO, CONTENT_TYPE, etc.
        """

        if environ["REQUEST_METHOD"] == "GET":
            # Create the response body from the reviews and convert to a JSON byte string
            
            
            # Write your code here
            query = parse_qs(environ['QUERY_STRING'])
            location = query.get('location',[None])[0]
            start_date = query.get('start_date',[None])[0]
            end_date = query.get('end_date',[None])[0]
            

            filtered_reviews = self.reviews

            if location:
                filtered_reviews = [
                    review for review in filtered_reviews if review['Location'] == location
                    ]
            

            if start_date:
                start_date = datetime.strptime(start_date, '%Y-%m-%d')
                filtered_reviews = [
                    review for review in filtered_reviews 
                    if datetime.strptime(review['Timestamp'], '%Y-%m-%d %H:%M:%S') >= start_date
                ]

            if end_date:
                end_date = datetime.strptime(end_date, '%Y-%m-%d')
                filtered_reviews = [
                    review for review in filtered_reviews 
                    if datetime.strptime(review['Timestamp'], '%Y-%m-%d %H:%M:%S') <= end_date
                ]
                
            for review in filtered_reviews:
                sentiment = self.analyze_sentiment(review['ReviewBody'])
                review['sentiment'] = {
                    'neg': sentiment['neg'],
                    'neu': sentiment['neu'],
                    'pos': sentiment['pos'],
                    'compound': sentiment['compound']
                }

            filtered_reviews.sort(key=lambda x:x['sentiment']['compound'],reverse=True)

            response_body = json.dumps(filtered_reviews, indent=2).encode("utf-8")
            # Set the appropriate response headers
            start_response("200 OK", [
            ("Content-Type", "application/json"),
            ("Content-Length", str(len(response_body)))
             ])
            return [response_body]


        if environ["REQUEST_METHOD"] == "POST":
            # Write your code here
            try:
                content_length = int(environ.get('CONTENT_LENGTH',0))
                post_data = environ['wsgi.input'].read(content_length).decode('utf-8')
                post_data = parse_qs(post_data)
                location = post_data.get('Location',[None])[0]
                review_body = post_data.get('ReviewBody',[None])[0]

                if not location or not review_body:
                    start_response("400 Bad Request",[("Content-Type","text/plain")])
                    return [b"Location and Review are required. "]
                

                # Validate location
                valid_locations = ["San Diego, California", "Denver, Colorado"]
                if location not in valid_locations:
                    start_response("400 Bad Request", [("Content-Type", "text/plain")])
                    return [b"Invalid Location"]
                
                new_review = {
                    "ReviewId": str(uuid.uuid4()),
                    "Location": location,
                    "Timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    "ReviewBody": review_body 
                }

                self.reviews.append(new_review)
                response_body = json.dumps(new_review, indent=2).encode("utf-8")
                start_response("201 Created", [("Content-Type", "application/json"), ("Content-Length", str(len(response_body)))])
                return [response_body]
            except Exception as e:
                start_response("500 Internal Server Error", [("Content-Type", "text/plain")])
                return [str(e).encode('utf-8')]
            

if __name__ == "__main__":
    app = ReviewAnalyzerServer()
    port = os.environ.get('PORT', 8000)
    with make_server("", port, app) as httpd:
        print(f"Listening on port {port}...")
        httpd.serve_forever()