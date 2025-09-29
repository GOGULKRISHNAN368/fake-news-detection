import React, { useState } from 'react';
import './NewsInput.css';

function NewsInput({ onAnalyze, loading }) {
  const [title, setTitle] = useState('');
  const [text, setText] = useState('');

  const handleSubmit = (e) => {
    e.preventDefault();
    if (title.trim() || text.trim()) {
      onAnalyze({ title, text });
    }
  };

  const handleClear = () => {
    setTitle('');
    setText('');
  };

  const sampleNews = () => {
    setTitle('New Study Reveals Climate Change Impact');
    setText('Researchers at major universities have published findings in peer-reviewed journals showing significant changes in global weather patterns. The comprehensive study analyzed data from multiple sources and included input from international climate experts.');
  };

  return (
    <div className="news-input-container">
      <div className="input-header">
        <h2>Enter News Article</h2>
        <button onClick={sampleNews} className="sample-btn" type="button">
          Try Sample
        </button>
      </div>

      <form onSubmit={handleSubmit} className="news-form">
        <div className="form-group">
          <label htmlFor="title">Article Title</label>
          <input
            id="title"
            type="text"
            value={title}
            onChange={(e) => setTitle(e.target.value)}
            placeholder="Enter the news headline..."
            disabled={loading}
          />
        </div>

        <div className="form-group">
          <label htmlFor="text">Article Content</label>
          <textarea
            id="text"
            value={text}
            onChange={(e) => setText(e.target.value)}
            placeholder="Paste the full article text here..."
            rows="10"
            disabled={loading}
          />
          <div className="char-count">
            {text.length} characters
          </div>
        </div>
        <                    
  div className="form-actions">
          <button
            type="button"
            onClick={handleClear}
            className="clear-btn"
            disabled={loading}
          >
            Clear
          </button>
          <button
            type="submit"
            className="analyze-btn"
            disabled={loading || (!title.trim() && !text.trim())}
          >
            {loading ? 'Analyzing...' : 'Analyze News'}
          </button>
        </div>
      </form>
    </div>
  );
}

export default NewsInput;