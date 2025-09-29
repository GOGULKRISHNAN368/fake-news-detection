import React, { useState, useEffect } from 'react';
import './LiveNews.css';

function LiveNews() {
  const [articles, setArticles] = useState([]);
  const [loading, setLoading] = useState(true);
  const [category, setCategory] = useState('general');
  const [error, setError] = useState(null);

  const categories = [
    'general',
    'business',
    'technology',
    'science',
    'health',
    'sports',
    'entertainment'
  ];

  useEffect(() => {
    fetchNews();
  }, [category]);

  const fetchNews = async () => {
    setLoading(true);
    setError(null);

    try {
      const response = await fetch(
        `http://localhost:5000/api/live-news?category=${category}`
      );
      const data = await response.json();

      if (response.ok) {
        setArticles(data.articles || []);
      } else {
        setError(data.error || 'Failed to fetch news');
      }
    } catch (err) {
      setError('⚠️ Failed to connect to server. Make sure backend is running.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="live-news-container">
      <div className="live-news-header">
        <h2>📰 Live News Feed</h2>
        <p>Latest headlines from trusted sources</p>
      </div>

      <div className="category-selector">
        {categories.map((cat) => (
          <button
            key={cat}
            className={`category-btn ${category === cat ? 'active' : ''}`}
            onClick={() => setCategory(cat)}
          >
            {cat.charAt(0).toUpperCase() + cat.slice(1)}
          </button>
        ))}
      </div>

      {loading && (
        <div className="loading-container">
          <div className="spinner"></div>
          <p>Loading news...</p>
        </div>
      )}

      {error && (
        <div className="error-container">
          <p>{error}</p>
        </div>
      )}

      {!loading && !error && articles.length > 0 && (
        <div className="news-grid">
          {articles.map((article, index) => (
            <div key={index} className="news-card">
              <div className="news-image">
                <img
                  src={article.image || 'https://via.placeholder.com/300x200?text=No+Image'}
                  alt={article.title}
                />
              </div>
              <div className="news-content">
                <div className="news-source">
                  {article.source}
                  {article.published_at && (
                    <span className="news-date">
                      {new Date(article.published_at).toLocaleDateString()}
                    </span>
                  )}
                </div>
                <h3>{article.title}</h3>
                <p>{article.description}</p>
                <a
                  href={article.url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="read-more"
                >
                  Read Full Article →
                </a>
              </div>
            </div>
          ))}
        </div>
      )}

      {!loading && !error && articles.length === 0 && (
        <div className="empty-state">
          <p>No articles found for this category.</p>
        </div>
      )}
    </div>
  );
}

export default LiveNews;
