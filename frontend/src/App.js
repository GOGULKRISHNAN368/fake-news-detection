import React, { useState } from 'react';
import NewsInput from './components/NewsInput';
import ResultDisplay from './components/ResultDisplay';
import LiveNews from './components/LiveNews';
import './App.css';

function App() {
  const [activeTab, setActiveTab] = useState('detector');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleAnalyze = async (newsData) => {
    setLoading(true);
    setResult(null);

    try {
      const response = await fetch('http://localhost:5000/api/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(newsData),
      });

      const data = await response.json();
      
      if (response.ok) {
        setResult(data);
      } else {
        setResult({ error: data.error || 'An error occurred' });
      }
    } catch (error) {
      setResult({ error: 'Failed to connect to server. Make sure backend is running.' });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="App">
      <header className="app-header">
        <div className="header-content">
          <h1>🔍 Fake News Detector</h1>
          <p>AI-Powered News Verification System</p>
        </div>
      </header>

      <div className="tab-navigation">
        <button
          className={`tab-button ${activeTab === 'detector' ? 'active' : ''}`}
          onClick={() => setActiveTab('detector')}
        >
          📝 Analyze News
        </button>
        <button
          className={`tab-button ${activeTab === 'live' ? 'active' : ''}`}
          onClick={() => setActiveTab('live')}
        >
          📰 Live News
        </button>
      </div>

      <main className="main-content">
        {activeTab === 'detector' ? (
          <div className="detector-section">
            <NewsInput onAnalyze={handleAnalyze} loading={loading} />
            {loading && (
              <div className="loading-spinner">
                <div className="spinner"></div>
                <p>Analyzing news article...</p>
              </div>
            )}
            {result && <ResultDisplay result={result} />}
          </div>
        ) : (
          <LiveNews />
        )}
      </main>

      <footer className="app-footer">
        <p>© 2025 Fake News Detector | Powered by Machine Learning & NLP</p>
      </footer>
    </div>
  );
}

export default App;