🚀 Quick Start                    
                      
                      
  
Prerequisites

Python 3.8+
Node.js 14+
(Optional) NewsAPI Key

Installation
1. Clone Repository
bashgit clone https://github.com/yourusername/fake-news-detector.git
cd fake-news-detector
2. Backend Setup
bashcd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
3. Frontend Setup
bashcd frontend
npm install
4. Run Application
bash# Terminal 1 - Backend
cd backend
python app.py

# Terminal 2 - Frontend
cd frontend
npm start
Open http://localhost:3000 in your browser.
📁 Project Structure
fake-news-detector/
├── backend/
│   ├── app.py                 # Flask API
│   ├── models/
│   │   └── ml_models.py       # ML models
│   ├── utils/
│   │   ├── text_processor.py  # NLP processing
│   │   └── news_fetcher.py    # NewsAPI integration
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── App.js
│   │   └── components/
│   │       ├── NewsInput.js
│   │       ├── ResultDisplay.js
│   │       └── LiveNews.js
│   └── package.json
└── README.md
🎯 Usage

Analyze News: Paste article text → Get prediction with confidence score
View Models: See individual predictions from all 3 ML models
Check Stats: Review text analysis and sentiment scores
Verify Live: Search related news articles to cross-reference claims
Browse News: Explore latest headlines by category

🔧 API Endpoints
EndpointMethodDescription/api/predictPOSTAnalyze news article/api/verify-newsPOSTSearch related articles/api/live-newsGETGet top headlines/api/healthGETHealth check
🧠 How It Works

Text Processing: Cleans, tokenizes, and lemmatizes input text
Feature Extraction: TF-IDF vectorization with n-grams
Multi-Model Prediction: 3 models vote on authenticity
Ensemble Decision: Combines predictions with confidence scoring
Analysis: Generates sentiment and linguistic statistics

🛠️ Tech Stack
Backend: Flask, scikit-learn, NLTK, TextBlob, NewsAPI
Frontend: React, Axios, CSS3
ML Models: Logistic Regression, Random Forest, Naive Bayes
📊 Training Custom Models
bash# Prepare CSV with columns: 'text', 'label' (0=fake, 1=real)
cd backend
python train_model.py
Recommended Datasets: LIAR, FakeNewsNet, ISOT
⚙️ Configuration
NewsAPI Key (Optional):
bash# Windows
set NEWS_API_KEY=your_key_here

# Mac/Linux
export NEWS_API_KEY=your_key_here
🐛 Troubleshooting                    
                      
                      
  
Backend won't start:

Activate virtual environment
Install requirements: pip install -r requirements.txt

Frontend errors:

Delete node_modules and run npm install

NLTK errors:
pythonimport nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
🤝 Contributing

Fork the repo
Create feature branch: git checkout -b feature/NewFeature
Commit changes: git commit -m 'Add NewFeature'
Push: git push origin feature/NewFeature
Open Pull Request

📝 License
MIT License - see LICENSE file
🙏 Acknowledgments

NewsAPI - News data provider
scikit-learn - ML framework
NLTK - NLP toolkit

📧 Contact
gogilkrishnan368@gmail.com
                                                                              
  
  
  
  
