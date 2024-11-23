from flask import Flask, request, render_template, jsonify
import pickle
import requests

# Load the trained vectorizer and model
vectorizer = pickle.load(open("vectorizer.pkl", 'rb'))
model = pickle.load(open("finalized_model.pkl", 'rb'))

# News API settings
news_api_key = 'YOUR_NEWS_API_KEY'  # Replace with your actual News API key
news_url = 'https://newsapi.org/v2/top-headlines'

# Initialize Flask app
app = Flask(__name__)

# Function to fetch live news articles
def fetch_news():
    params = {
        'apiKey': news_api_key,
        'country': 'us',  # Fetching news from the US
        'pageSize': 5  # Number of articles to fetch
    }
    response = requests.get(news_url, params=params)
    news_data = response.json()
    articles = news_data['articles']
    return [article['title'] + " " + article['description'] for article in articles]

# Home route
@app.route('/')
def home():
    return render_template("index.html")

# Real-time prediction for a user-submitted news headline
@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    if request.method == "POST":
        news = str(request.form['news'])
        print("User news input:", news)

        # Transform the news text and make the prediction
        predict = model.predict(vectorizer.transform([news]))[0]
        print("Prediction:", predict)

        # Convert the prediction to readable text
        prediction_text = "Fake News" if predict == 0 else "Real News"

        return render_template("prediction.html", prediction_text=f"News headline is -> {prediction_text}")

    else:
        return render_template("prediction.html")

# Route to fetch live news, make predictions, and show them in real-time
@app.route('/fetch_and_predict', methods=['GET'])
def fetch_and_predict():
    # Fetch live news
    news_articles = fetch_news()

    predictions = []
    for article in news_articles:
        # Predict the news article's authenticity
        predict = model.predict(vectorizer.transform([article]))[0]
        result = 'Fake News' if predict == 0 else 'Real News'
        predictions.append({'article': article, 'prediction': result})

    # Return the predictions in a JSON format to display on the frontend
    return jsonify({'predictions': predictions})

if __name__ == '__main__':
    app.debug = True
    app.run()
