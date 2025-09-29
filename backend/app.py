from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from models.ml_models import FakeNewsDetector
from utils.news_fetcher import NewsFetcher
from utils.text_processor import TextProcessor

app = Flask(__name__)
CORS(app)

# Initialize components
detector = FakeNewsDetector()
news_fetcher = NewsFetcher()
text_processor = TextProcessor()

@app.route('/api/predict', methods=['POST'])
def predict_news():
    try:
        data = request.json
        text = data.get('text', '')
        title = data.get('title', '')
        
        if not text and not title:
            return jsonify({'error': 'No text provided'}), 400
        
        # Combine title and text
        full_text = f"{title} {text}".strip()
        
        # Process text
        processed_text = text_processor.process(full_text)
        
        # Get predictions from all models
        predictions = detector.predict(processed_text)
        
        # Calculate confidence scores
        response = {
            'prediction': predictions['ensemble']['label'],
            'confidence': predictions['ensemble']['confidence'],
            'models': {
                'logistic_regression': predictions['logistic_regression'],
                'random_forest': predictions['random_forest'],
                'naive_bayes': predictions['naive_bayes']
            },
            'analysis': text_processor.get_text_stats(full_text),
            'processed_text': processed_text[:200] + '...' if len(processed_text) > 200 else processed_text
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/verify-news', methods=['POST'])
def verify_news():
    try:
        data = request.json
        query = data.get('query', '')
        
        if not query:
            return jsonify({'error': 'No query provided'}), 400
        
        # Fetch related news
        articles = news_fetcher.search_news(query)
        
        response = {
            'articles': articles,
            'total': len(articles)
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/live-news', methods=['GET'])
def get_live_news():
    try:
        category = request.args.get('category', 'general')
        articles = news_fetcher.get_top_headlines(category=category)
        
        return jsonify({
            'articles': articles,
            'total': len(articles)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'model_loaded': detector.is_loaded()})

if __name__ == '__main__':
    app.run(debug=True, port=5000)