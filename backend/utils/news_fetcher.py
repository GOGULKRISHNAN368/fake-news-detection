from newsapi import NewsApiClient
import os
from datetime import datetime, timedelta

class NewsFetcher:
    def __init__(self):
        # Get API key from environment variable or use a placeholder
        api_key = os.environ.get('NEWS_API_KEY', 'YOUR_NEWS_API_KEY_HERE')
        
        if api_key == 'YOUR_NEWS_API_KEY_HERE':
            print("Warning: Using placeholder API key. Get a free key from https://newsapi.org/")
            self.newsapi = None
        else:
            self.newsapi = NewsApiClient(api_key=api_key)
    
    def search_news(self, query, days=7):
        """Search for news articles related to a query"""
        if not self.newsapi:
            return self._get_dummy_articles()
        
        try:
            # Calculate date range
            to_date = datetime.now()
            from_date = to_date - timedelta(days=days)
            
            # Search articles
            response = self.newsapi.get_everything(
                q=query,
                from_param=from_date.strftime('%Y-%m-%d'),
                to=to_date.strftime('%Y-%m-%d'),
                language='en',
                sort_by='relevancy',
                page_size=10
            )
            
            return self._format_articles(response.get('articles', []))
        
        except Exception as e:
            print(f"Error fetching news: {e}")
            return self._get_dummy_articles()
    
    def get_top_headlines(self, category='general', country='us'):
        """Get top headlines"""
        if not self.newsapi:
            return self._get_dummy_articles()
        
        try:
            response = self.newsapi.get_top_headlines(
                category=category,
                country=country,
                page_size=20
            )
            
            return self._format_articles(response.get('articles', []))
        
        except Exception as e:
            print(f"Error fetching headlines: {e}")
            return self._get_dummy_articles()
    
    def _format_articles(self, articles):
        """Format articles for response"""
        formatted = []
        for article in articles:
            formatted.append({
                'title': article.get('title', ''),
                'description': article.get('description', ''),
                'url': article.get('url', ''),
                'source': article.get('source', {}).get('name', 'Unknown'),
                'published_at': article.get('publishedAt', ''),
                'image': article.get('urlToImage', '')
            })
        return formatted
    
    def _get_dummy_articles(self):
        """Return dummy articles when API is not available"""
        return [
            {
                'title': 'News API Key Required',
                'description': 'Please set NEWS_API_KEY environment variable with your API key from newsapi.org',
                'url': 'https://newsapi.org/',
                'source': 'System',
                'published_at': datetime.now().isoformat(),
                'image': ''
            }
        ]