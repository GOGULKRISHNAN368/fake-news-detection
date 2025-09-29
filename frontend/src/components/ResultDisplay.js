import React, { useState } from 'react';
import './ResultDisplay.css';

function ResultDisplay({ result }) {
  const [verifyQuery, setVerifyQuery] = useState('');
  const [verifyResults, setVerifyResults] = useState(null);
  const [verifying, setVerifying] = useState(false);

  if (result.error) {
    return (
      <div className="result-container error">
        <div className="error-icon">⚠️</div>
        <h3>Error</h3>
        <p>{result.error}</p>
      </div>
    );
  }

  const isFake = result.prediction === 'Fake';
  const confidenceColor = result.confidence > 75 ? '#10b981' : result.confidence > 50 ? '#f59e0b' : '#ef4444';

  const handleVerify = async () => {
    if (!verifyQuery.trim()) return;

    setVerifying(true);
    try {
      const response = await fetch('http://localhost:5000/api/verify-news', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ query: verifyQuery }),
      });

      const data = await response.json();
      setVerifyResults(data);
    } catch (error) {
      console.error('Error verifying news:', error);
    } finally {
      setVerifying(false);
    }
  };

  return (
    <div className="result-container">
      <div className={`result-header ${isFake ? 'fake' : 'real'}`}>
        <div className="result-icon">
          {isFake ? '❌' : '✅'}
        </div>
        <div className="result-main">
          <h2>Prediction: {result.prediction}</h2>
          <div className="confidence-bar">
            <div className="confidence-label">
              Confidence: {result.confidence}%
            </div>
            <div className="confidence-track">
              <div
                className="confidence-fill"
                style={{
                  width: `${result.confidence}%`,
                  backgroundColor: confidenceColor
                }}
              />
            </div>
          </div>
        </div>
      </div>

      <div className="models-section">
        <h3>Individual Model Predictions</h3>
        <div className="models-grid">
          <div className="model-card">
            <h4>Logistic Regression</h4>
            <p className={`model-prediction ${result.models.logistic_regression.label.toLowerCase()}`}>
              {result.models.logistic_regression.label}
            </p>
            <span className="model-confidence">
              {result.models.logistic_regression.confidence}%
            </span>
          </div>

          <div className="model-card">
            <h4>Random Forest</h4>
            <p className={`model-prediction ${result.models.random_forest.label.toLowerCase()}`}>
              {result.models.random_forest.label}
            </p>
            <span className="model-confidence">
              {result.models.random_forest.confidence}%
            </span>
          </div>

          <div className="model-card">
            <h4>Naive Bayes</h4>
            <p className={`model-prediction ${result.models.naive_bayes.label.toLowerCase()}`}>
              {result.models.naive_bayes.label}
            </p>
            <span className="model-confidence">
              {result.models.naive_bayes.confidence}%
            </span>
          </div>
        </div>
      </div>

      {result.analysis && (
        <div className="analysis-section">
          <h3>Text Analysis</h3>
          <div className="analysis-grid">
            <div className="analysis-item">
              <span className="analysis-label">Word Count</span>
              <span className="analysis-value">{result.analysis.word_count}</span>
            </div>
            <div className="analysis-item">
              <span className="analysis-label">Avg Word Length</span>
              <span className="analysis-value">{result.analysis.avg_word_length}</span>
            </div>
            <div className="analysis-item">
              <span className="analysis-label">Exclamations</span>
              <span className="analysis-value">{result.analysis.exclamation_count}</span>
            </div>
            <div className="analysis-item">
              <span className="analysis-label">Questions</span>
              <span className="analysis-value">{result.analysis.question_count}</span>
            </div>
            <div className="analysis-item">
              <span className="analysis-label">Sentiment</span>
              <span className="analysis-value">
                {result.analysis.sentiment_polarity > 0 ? '😊 Positive' : 
                 result.analysis.sentiment_polarity < 0 ? '😞 Negative' : '😐 Neutral'}
              </span>
            </div>
            <div className="analysis-item">
              <span className="analysis-label">Subjectivity</span>
              <span className="analysis-value">
                {(result.analysis.sentiment_subjectivity * 100).toFixed(0)}%
              </span>
            </div>
          </div>
        </div>
      )}

      <div className="verify-section">
        <h3>Verify with Live News Search</h3>
        <div className="verify-input-group">
          <input
            type="text"
            placeholder="Enter keywords to search related news..."
            value={verifyQuery}
            onChange={(e) => setVerifyQuery(e.target.value)}
            onKeyPress={(e) => e.key === 'Enter' && handleVerify()}
          />
          <button onClick={handleVerify} disabled={verifying || !verifyQuery.trim()}>
            {verifying ? 'Searching...' : 'Search'}
          </button>
        </div>

        {verifyResults && (
          <div className="verify-results">
            <h4>Found {verifyResults.total} related articles</h4>
            <div className="articles-list">
              {verifyResults.articles.map((article, index) => (
                <div key={index} className="article-item">
                  <h5>{article.title}</h5>
                  <p>{article.description}</p>
                  <div className="article-meta">
                    <span className="article-source">{article.source}</span>
                    <a href={article.url} target="_blank" rel="noopener noreferrer">
                      Read More →
                    </a>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default ResultDisplay;