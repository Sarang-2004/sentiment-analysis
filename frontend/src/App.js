import React, { useState } from 'react';
import './App.css';

function App() {
  const [text, setText] = useState("");
  const [sentiment, setSentiment] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleAnalyze = async () => {
    if (!text.trim()) {
      setError("Please enter text for sentiment analysis.");
      return;
    }
    setError(null);
    setLoading(true);
    try {
      const response = await fetch("http://localhost:5000/analyze", {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text })
      });
      const data = await response.json();
      setSentiment(data.Sentiment);
    } catch (err) {
      setError("Failed to fetch sentiment analysis. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container">
      <div className="analysis-card">
        <h1>Sentiment Analysis</h1>

        <textarea
          placeholder="Enter text to analyze..."
          value={text}
          onChange={(e) => setText(e.target.value)}
        />

        <button
          onClick={handleAnalyze}
          disabled={loading}
          className={loading ? 'button-disabled' : ''}
        >
          {loading ? "Analyzing..." : "Analyze Sentiment"}
        </button>

        {error && (
          <div className="error-message">
            {error}
          </div>
        )}

        {sentiment && (
          <div className="result-container">
            <div className={`sentiment-result ${sentiment.toLowerCase()}`}>
              {sentiment}
            </div>
            <div className="sentiment-bar-container">
              <div className={`sentiment-bar ${sentiment.toLowerCase()}`}></div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;