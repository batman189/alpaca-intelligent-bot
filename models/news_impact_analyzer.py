"""
AI-Powered News Analysis Module
Phase 1 Enhancement from Strategic Analysis & Optimization Plan

Analyze news sentiment and predict price impact for trading decisions
"""

import asyncio
import logging
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import json
import re
from collections import defaultdict
import time

# Try to import sentiment analysis libraries
try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

@dataclass
class NewsArticle:
    """Container for news article data"""
    title: str
    content: str
    symbol: str
    timestamp: datetime
    source: str
    url: Optional[str] = None
    sentiment_score: Optional[float] = None
    impact_prediction: Optional[float] = None
    keywords: List[str] = field(default_factory=list)

@dataclass
class SentimentAnalysis:
    """Sentiment analysis results"""
    sentiment_score: float  # -1 to 1 (negative to positive)
    confidence: float       # 0 to 1
    keywords: List[str]
    impact_magnitude: float # Expected price impact (%)
    time_horizon: int       # Minutes until impact
    category: str          # 'earnings', 'news', 'analyst', 'regulatory'

class NewsImpactAnalyzer:
    """
    Analyze news sentiment and predict price impact
    As specified in Strategic Analysis Phase 1
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Historical impact correlation data
        self.impact_history: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
        
        # Keyword impact weights
        self.positive_keywords = {
            'earnings beat', 'revenue growth', 'profit surge', 'acquisition', 'partnership',
            'breakthrough', 'expansion', 'upgrade', 'bullish', 'outperform', 'buy rating',
            'strong results', 'exceeds expectations', 'record high', 'momentum'
        }
        
        self.negative_keywords = {
            'earnings miss', 'revenue decline', 'loss', 'lawsuit', 'investigation',
            'bankruptcy', 'downgrade', 'bearish', 'underperform', 'sell rating',
            'disappointing', 'below expectations', 'regulatory issues', 'recall'
        }
        
        # Impact multipliers by category
        self.category_multipliers = {
            'earnings': 3.0,
            'analyst': 1.5,
            'regulatory': 2.5,
            'partnership': 2.0,
            'news': 1.0
        }
        
        self.logger.info("📰 News Impact Analyzer initialized")
        if not TEXTBLOB_AVAILABLE:
            self.logger.warning("⚠️ TextBlob not available - using basic sentiment analysis")
    
    async def analyze_symbol_news(self, symbol: str, hours_back: int = 4) -> List[SentimentAnalysis]:
        """
        Analyze recent news for a specific symbol
        
        Args:
            symbol: Stock symbol to analyze
            hours_back: How many hours of news to analyze
            
        Returns:
            List of sentiment analyses
        """
        try:
            self.logger.debug(f"🔍 Analyzing news sentiment for {symbol}")
            
            # Get recent news articles
            articles = await self._fetch_news_articles(symbol, hours_back)
            
            if not articles:
                self.logger.debug(f"No recent news found for {symbol}")
                return []
            
            # Analyze each article
            analyses = []
            for article in articles:
                analysis = await self._analyze_article_sentiment(article)
                if analysis:
                    analyses.append(analysis)
            
            # Combine and weight the analyses
            combined_analysis = self._combine_sentiment_analyses(analyses, symbol)
            
            return [combined_analysis] if combined_analysis else []
            
        except Exception as e:
            self.logger.error(f"Error analyzing news for {symbol}: {e}")
            return []
    
    async def _fetch_news_articles(self, symbol: str, hours_back: int) -> List[NewsArticle]:
        """
        Fetch recent news articles for symbol
        Using multiple sources for comprehensive coverage
        """
        articles = []
        
        try:
            # Try Yahoo Finance news first (free)
            if YFINANCE_AVAILABLE:
                articles.extend(await self._fetch_yfinance_news(symbol, hours_back))
            
            # Add other news sources here
            # articles.extend(await self._fetch_alpha_vantage_news(symbol, hours_back))
            # articles.extend(await self._fetch_finviz_news(symbol, hours_back))
            
            return articles
            
        except Exception as e:
            self.logger.error(f"Error fetching news for {symbol}: {e}")
            return []
    
    async def _fetch_yfinance_news(self, symbol: str, hours_back: int) -> List[NewsArticle]:
        """Fetch news from Yahoo Finance"""
        try:
            ticker = yf.Ticker(symbol)
            
            # Get news data with proper error handling
            try:
                news_data = ticker.news
            except Exception as news_fetch_error:
                self.logger.error(f"Failed to fetch news data for {symbol}: {news_fetch_error}")
                return []
            
            # Validate news_data
            if not news_data:
                self.logger.debug(f"No news data available for {symbol}")
                return []
            
            if not isinstance(news_data, list):
                self.logger.error(f"Unexpected news data format for {symbol}: {type(news_data)}")
                return []
            
            articles = []
            cutoff_time = datetime.now() - timedelta(hours=hours_back)
            
            for item in news_data[:10]:  # Limit to recent articles
                try:
                    # Validate item structure
                    if not isinstance(item, dict):
                        self.logger.debug(f"Skipping invalid news item: {type(item)}")
                        continue
                    
                    # Get timestamp with fallback
                    timestamp = item.get('providerPublishTime', 0)
                    if not timestamp:
                        self.logger.debug("Skipping news item with no timestamp")
                        continue
                    
                    # Convert timestamp safely
                    try:
                        article_time = datetime.fromtimestamp(timestamp)
                    except (ValueError, TypeError, OSError) as ts_error:
                        self.logger.debug(f"Invalid timestamp {timestamp}: {ts_error}")
                        continue
                    
                    if article_time < cutoff_time:
                        continue
                    
                    # Validate required fields
                    title = item.get('title', '').strip()
                    if not title:
                        self.logger.debug("Skipping news item with no title")
                        continue
                    
                    article = NewsArticle(
                        title=title,
                        content=item.get('summary', '').strip(),
                        symbol=symbol,
                        timestamp=article_time,
                        source='Yahoo Finance',
                        url=item.get('link', '')
                    )
                    
                    articles.append(article)
                    
                except Exception as e:
                    self.logger.debug(f"Error processing news item for {symbol}: {e}")
                    continue
            
            self.logger.debug(f"Fetched {len(articles)} valid articles from Yahoo Finance for {symbol}")
            return articles
            
        except Exception as e:
            self.logger.error(f"Error fetching Yahoo Finance news for {symbol}: {e}")
            return []
    
    async def _analyze_article_sentiment(self, article: NewsArticle) -> Optional[SentimentAnalysis]:
        """
        Analyze sentiment of a single article
        """
        try:
            combined_text = f"{article.title} {article.content}"
            
            # Extract keywords
            keywords = self._extract_keywords(combined_text)
            
            # Calculate sentiment score
            sentiment_score = self._calculate_sentiment_score(combined_text, keywords)
            
            # Determine category
            category = self._categorize_news(combined_text, keywords)
            
            # Predict impact magnitude
            impact_magnitude = self._predict_impact_magnitude(
                sentiment_score, keywords, category, article.symbol
            )
            
            # Estimate time horizon
            time_horizon = self._estimate_time_horizon(category)
            
            # Calculate confidence
            confidence = self._calculate_confidence(sentiment_score, keywords, category)
            
            analysis = SentimentAnalysis(
                sentiment_score=sentiment_score,
                confidence=confidence,
                keywords=keywords,
                impact_magnitude=impact_magnitude,
                time_horizon=time_horizon,
                category=category
            )
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing article sentiment: {e}")
            return None
    
    def _calculate_sentiment_score(self, text: str, keywords: List[str]) -> float:
        """
        Calculate sentiment score from text and keywords
        """
        try:
            # Use TextBlob if available
            if TEXTBLOB_AVAILABLE:
                blob = TextBlob(text.lower())
                base_sentiment = blob.sentiment.polarity
            else:
                # Fallback to keyword-based sentiment
                base_sentiment = self._keyword_based_sentiment(text.lower())
            
            # Enhance with keyword analysis
            keyword_sentiment = 0
            for keyword in keywords:
                if keyword in self.positive_keywords:
                    keyword_sentiment += 0.3
                elif keyword in self.negative_keywords:
                    keyword_sentiment -= 0.3
            
            # Combine scores
            final_sentiment = (base_sentiment * 0.7) + (keyword_sentiment * 0.3)
            
            # Clamp to [-1, 1]
            return max(-1.0, min(1.0, final_sentiment))
            
        except Exception as e:
            self.logger.error(f"Error calculating sentiment: {e}")
            return 0.0
    
    def _keyword_based_sentiment(self, text: str) -> float:
        """Fallback sentiment analysis using keywords"""
        positive_count = sum(1 for keyword in self.positive_keywords if keyword in text)
        negative_count = sum(1 for keyword in self.negative_keywords if keyword in text)
        
        if positive_count + negative_count == 0:
            return 0.0
        
        return (positive_count - negative_count) / (positive_count + negative_count)
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract relevant keywords from text"""
        text_lower = text.lower()
        keywords = []
        
        # Check for positive keywords
        for keyword in self.positive_keywords:
            if keyword in text_lower:
                keywords.append(keyword)
        
        # Check for negative keywords
        for keyword in self.negative_keywords:
            if keyword in text_lower:
                keywords.append(keyword)
        
        # Add financial terms
        financial_terms = ['earnings', 'revenue', 'profit', 'loss', 'guidance', 'outlook']
        for term in financial_terms:
            if term in text_lower:
                keywords.append(term)
        
        return keywords
    
    def _categorize_news(self, text: str, keywords: List[str]) -> str:
        """Categorize news article"""
        text_lower = text.lower()
        
        if any(term in text_lower for term in ['earnings', 'quarterly', 'q1', 'q2', 'q3', 'q4']):
            return 'earnings'
        elif any(term in text_lower for term in ['upgrade', 'downgrade', 'rating', 'analyst']):
            return 'analyst'
        elif any(term in text_lower for term in ['regulatory', 'fda', 'sec', 'investigation']):
            return 'regulatory'
        elif any(term in text_lower for term in ['acquisition', 'merger', 'partnership']):
            return 'partnership'
        else:
            return 'news'
    
    def _predict_impact_magnitude(self, sentiment_score: float, keywords: List[str], 
                                 category: str, symbol: str) -> float:
        """
        Predict expected price impact magnitude
        """
        try:
            # Base impact from sentiment
            base_impact = abs(sentiment_score) * 2.0  # Up to 2% base impact
            
            # Category multiplier
            category_multiplier = self.category_multipliers.get(category, 1.0)
            
            # Keyword intensity
            keyword_intensity = min(len(keywords) * 0.2, 1.0)
            
            # Historical correlation (if available)
            historical_multiplier = self._get_historical_multiplier(symbol, category)
            
            # Combine factors
            predicted_impact = base_impact * category_multiplier * (1 + keyword_intensity) * historical_multiplier
            
            # Cap at reasonable levels
            return min(predicted_impact, 10.0)  # Max 10% predicted impact
            
        except Exception as e:
            self.logger.error(f"Error predicting impact: {e}")
            return 0.0
    
    def _get_historical_multiplier(self, symbol: str, category: str) -> float:
        """Get historical impact multiplier for symbol/category combination"""
        key = f"{symbol}_{category}"
        history = self.impact_history.get(key, [])
        
        if not history:
            return 1.0
        
        # Calculate average historical impact
        avg_impact = sum(impact for sentiment, impact in history) / len(history)
        return max(0.5, min(2.0, avg_impact))  # Clamp between 0.5x and 2x
    
    def _estimate_time_horizon(self, category: str) -> int:
        """Estimate time horizon for impact in minutes"""
        horizons = {
            'earnings': 30,      # Impact within 30 minutes
            'analyst': 60,       # Impact within 1 hour
            'regulatory': 120,   # Impact within 2 hours
            'partnership': 240,  # Impact within 4 hours
            'news': 180         # Impact within 3 hours
        }
        return horizons.get(category, 180)
    
    def _calculate_confidence(self, sentiment_score: float, keywords: List[str], category: str) -> float:
        """Calculate confidence in the analysis"""
        try:
            # Base confidence from sentiment strength
            base_confidence = abs(sentiment_score)
            
            # Keyword confidence
            keyword_confidence = min(len(keywords) * 0.1, 0.4)
            
            # Category confidence
            category_confidence = {
                'earnings': 0.8,
                'analyst': 0.6,
                'regulatory': 0.7,
                'partnership': 0.75,
                'news': 0.5
            }.get(category, 0.5)
            
            # Combine confidences
            total_confidence = (base_confidence * 0.4) + (keyword_confidence * 0.2) + (category_confidence * 0.4)
            
            return min(total_confidence, 0.95)  # Cap at 95%
            
        except Exception as e:
            self.logger.error(f"Error calculating confidence: {e}")
            return 0.5
    
    def _combine_sentiment_analyses(self, analyses: List[SentimentAnalysis], symbol: str) -> Optional[SentimentAnalysis]:
        """
        Combine multiple sentiment analyses into a single result
        """
        if not analyses:
            return None
        
        try:
            # Weight by confidence and recency
            total_weight = 0
            weighted_sentiment = 0
            weighted_impact = 0
            all_keywords = []
            
            for analysis in analyses:
                weight = analysis.confidence
                total_weight += weight
                weighted_sentiment += analysis.sentiment_score * weight
                weighted_impact += analysis.impact_magnitude * weight
                all_keywords.extend(analysis.keywords)
            
            if total_weight == 0:
                return None
            
            # Calculate combined metrics
            combined_sentiment = weighted_sentiment / total_weight
            combined_impact = weighted_impact / total_weight
            combined_confidence = min(total_weight / len(analyses), 0.95)
            
            # Determine dominant category
            categories = [a.category for a in analyses]
            dominant_category = max(set(categories), key=categories.count)
            
            # Unique keywords
            unique_keywords = list(set(all_keywords))
            
            combined = SentimentAnalysis(
                sentiment_score=combined_sentiment,
                confidence=combined_confidence,
                keywords=unique_keywords,
                impact_magnitude=combined_impact,
                time_horizon=analyses[0].time_horizon,  # Use first analysis's time horizon
                category=dominant_category
            )
            
            self.logger.info(f"📊 {symbol} News Sentiment: {combined_sentiment:.2f} "
                           f"(confidence: {combined_confidence:.2f}, "
                           f"predicted impact: {combined_impact:.2f}%)")
            
            return combined
            
        except Exception as e:
            self.logger.error(f"Error combining analyses: {e}")
            return None
    
    async def learn_from_trade_outcome(self, symbol: str, category: str, 
                                     predicted_impact: float, actual_impact: float):
        """
        Learn from trade outcomes to improve future predictions
        """
        try:
            key = f"{symbol}_{category}"
            self.impact_history[key].append((predicted_impact, actual_impact))
            
            # Keep only recent history (last 50 outcomes)
            if len(self.impact_history[key]) > 50:
                self.impact_history[key] = self.impact_history[key][-50:]
            
            self.logger.debug(f"📚 Learning: {symbol} {category} - "
                            f"Predicted: {predicted_impact:.2f}%, Actual: {actual_impact:.2f}%")
            
        except Exception as e:
            self.logger.error(f"Error learning from outcome: {e}")
    
    def get_analysis_summary(self, symbol: str, analysis: SentimentAnalysis) -> str:
        """
        Generate human-readable analysis summary
        """
        try:
            sentiment_desc = "Positive" if analysis.sentiment_score > 0.1 else "Negative" if analysis.sentiment_score < -0.1 else "Neutral"
            
            summary = (f"📰 {symbol} News Analysis: {sentiment_desc} sentiment ({analysis.sentiment_score:.2f}) "
                      f"with {analysis.confidence:.1%} confidence. "
                      f"Category: {analysis.category.title()}, "
                      f"Predicted impact: {analysis.impact_magnitude:.1f}% "
                      f"within {analysis.time_horizon} minutes.")
            
            if analysis.keywords:
                summary += f" Key terms: {', '.join(analysis.keywords[:3])}"
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error generating summary: {e}")
            return f"News analysis available for {symbol}"

# Global instance
news_analyzer = NewsImpactAnalyzer()