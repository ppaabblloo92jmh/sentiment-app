
import numpy as np
import requests
from transformers import pipeline
from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
from datetime import datetime, timedelta
import os
import time
from functools import lru_cache

nltk.download('vader_lexicon', quiet=True)

class SentimentAnalyzer:
    def __init__(self):
        self.finbert = pipeline("text-classification", model="ProsusAI/finbert")
        self.vader = SentimentIntensityAnalyzer()
        self.news_api_key = os.getenv("NEWS_API_KEY")
        self.twitter_bearer_token = os.getenv("TWITTER_BEARER_TOKEN")
        self.financial_lexicon = {
            'bullish': 1.5, 'moon': 2.0, 'rally': 1.7, 'long': 1.3,
            'bearish': -1.5, 'crash': -2.0, 'dump': -1.8, 'short': -1.3,
            'btc': 0.8, 'bitcoin': 0.9, 'halving': 1.2, 'fud': -1.4,
            'pump': 1.6, 'hodl': 0.7
        }
        self.last_api_call = 0
        self.api_cooldown = 1.2

    def _rate_limit(self):
        elapsed = time.time() - self.last_api_call
        if elapsed < self.api_cooldown:
            time.sleep(self.api_cooldown - elapsed)
        self.last_api_call = time.time()

    def _get_news(self, query="Bitcoin", hours=24, max_articles=20):
        if not self.news_api_key:
            return []
        try:
            self._rate_limit()
            url = f"https://newsapi.org/v2/everything?q={query}&pageSize={max_articles}&apiKey={self.news_api_key}"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            articles = response.json().get('articles', [])
            valid_articles = []
            for art in articles:
                try:
                    pub_date = datetime.fromisoformat(art['publishedAt'].replace('Z', '+00:00'))
                    if datetime.utcnow() - pub_date < timedelta(hours=hours):
                        valid_articles.append({
                            'title': art['title'],
                            'content': art['content'][:1000] if art['content'] else "",
                            'source': art['source']['name'],
                            'date': art['publishedAt']
                        })
                except Exception:
                    continue
            return valid_articles
        except Exception:
            return []

    def _get_tweets(self, query="Bitcoin OR BTC", max_results=50):
        if not self.twitter_bearer_token:
            return []
        try:
            self._rate_limit()
            url = "https://api.twitter.com/2/tweets/search/recent"
            headers = {"Authorization": f"Bearer {self.twitter_bearer_token}"}
            params = {
                'query': f'{query} -is:retweet lang:en',
                'max_results': max_results,
                'tweet.fields': 'created_at,public_metrics'
            }
            response = requests.get(url, headers=headers, params=params, timeout=10)
            response.raise_for_status()
            tweets = response.json().get('data', [])
            return [{
                'text': tweet['text'],
                'likes': tweet['public_metrics']['like_count'],
                'date': tweet['created_at']
            } for tweet in tweets]
        except Exception:
            return []

    def _enhance_financial_sentiment(self, text):
        sentiment = 0
        text_lower = text.lower()
        for word, value in self.financial_lexicon.items():
            count = text_lower.count(word)
            if count > 0:
                sentiment += value * (count * 0.15)
        return sentiment

    @lru_cache(maxsize=512)
    def analyze_text(self, text):
        try:
            finbert_result = self.finbert(text[:1024])[0]
            finbert_score = 0.5 if finbert_result['label'] == 'neutral' else                             0.8 if finbert_result['label'] == 'positive' else -0.8
            vader_score = self.vader.polarity_scores(text)['compound']
            blob_score = TextBlob(text).sentiment.polarity
            custom_score = self._enhance_financial_sentiment(text) * 0.3
            weighted_score = (
                finbert_score * 0.5 + 
                vader_score * 0.3 + 
                blob_score * 0.1 + 
                custom_score * 0.1
            )
            return {
                'score': weighted_score,
                'finbert': finbert_result,
                'vader': vader_score,
                'textblob': blob_score,
                'custom': custom_score
            }
        except Exception:
            return {
                'score': 0,
                'finbert': {'label': 'error', 'score': 0},
                'vader': 0,
                'textblob': 0,
                'custom': 0
            }

    def get_market_sentiment(self):
        try:
            news = self._get_news(max_articles=15)
            tweets = self._get_tweets(max_results=30)
            if not news and not tweets:
                return {
                    'sentiment': "Datos insuficientes ⚠️",
                    'total_score': 0,
                    'news_avg': 0,
                    'tweets_avg': 0,
                    'sources_analyzed': 0
                }
            news_scores = [self.analyze_text(n['title'] + " " + n['content']) for n in news]
            tweet_scores = [self.analyze_text(t['text']) for t in tweets]
            news_avg = np.mean([n['score'] for n in news_scores]) if news_scores else 0
            tweets_avg = np.mean([t['score'] for t in tweet_scores]) if tweet_scores else 0
            total_score = (news_avg * 0.6) + (tweets_avg * 0.4)
            if total_score > 0.25:
                sentiment = "FUERTEMENTE ALCISTA 🚀"
            elif total_score > 0.08:
                sentiment = "Alcista 👍"
            elif total_score < -0.25:
                sentiment = "FUERTEMENTE BAJISTA 💥"
            elif total_score < -0.08:
                sentiment = "Bajista 👎"
            else:
                sentiment = "Neutral 😐"
            return {
                'sentiment': sentiment,
                'total_score': total_score,
                'news_avg': news_avg,
                'tweets_avg': tweets_avg,
                'sources_analyzed': len(news) + len(tweets)
            }
        except Exception:
            return {
                'sentiment': "Error en análisis ⚠️",
                'total_score': 0,
                'news_avg': 0,
                'tweets_avg': 0,
                'sources_analyzed': 0
            }
