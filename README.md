🚀 Quick Start                    
                      
                      
  Prerequisites

Python 3.8+
Node.js 14+
(Optional) NewsAPI Key
Installation1. Clone Repository
bashgit clone https://github.com/yourusername/fake-news-detector.git
cd fake-news-detector2. Backend Setup
bashcd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"3. Frontend Setup
bashcd frontend
npm install4. Run Application
bash# Terminal 1 - Backend
cd backend
python app.py

# Terminal 2 - Frontend
cd frontend
npm startOpen http://localhost:3000 in your browser.📁 Project Structurefake-news-detector/
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
└── README.md🎯 Usage
Analyze News: Paste article text → Get prediction with confidence score
View Models: See individual predictions from all 3 ML models
Check Stats: Review text analysis and sentiment scores
Verify Live: Search related news articles to cross-reference claims
Browse News: Explore latest headlines by category

 Troubleshooting                    
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
mail: gogulkrishnan368@gmail.com                                                                                                                                                                
Project: github.com/GOGULKRISHNAN368/fake-news-detector                                        
  
  
