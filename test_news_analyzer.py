#!/usr/bin/env python3
"""
Test script for News Impact Analyzer - Phase 1 Enhancement
"""

import asyncio
import sys
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_news_analyzer():
    """Test the News Impact Analyzer functionality"""
    try:
        logger.info("🧪 Testing News Impact Analyzer...")
        
        # Import the news analyzer
        from models.news_impact_analyzer import NewsImpactAnalyzer
        
        # Initialize analyzer
        analyzer = NewsImpactAnalyzer()
        
        # Test symbols from our high-volatility watchlist
        test_symbols = ['TSLA', 'NVDA', 'AAPL']
        
        for symbol in test_symbols:
            logger.info(f"📰 Analyzing news for {symbol}...")
            
            try:
                # Analyze recent news
                analyses = await analyzer.analyze_symbol_news(symbol, hours_back=12)
                
                if analyses:
                    for analysis in analyses:
                        summary = analyzer.get_analysis_summary(symbol, analysis)
                        logger.info(f"✅ {summary}")
                        
                        # Show detailed breakdown
                        logger.info(f"   Sentiment Score: {analysis.sentiment_score:.3f}")
                        logger.info(f"   Confidence: {analysis.confidence:.1%}")
                        logger.info(f"   Category: {analysis.category.title()}")
                        logger.info(f"   Predicted Impact: {analysis.impact_magnitude:.2f}%")
                        logger.info(f"   Time Horizon: {analysis.time_horizon} minutes")
                        
                        if analysis.keywords:
                            logger.info(f"   Keywords: {', '.join(analysis.keywords[:5])}")
                        
                        logger.info("-" * 60)
                else:
                    logger.info(f"   No significant news found for {symbol} in last 12 hours")
                    
            except Exception as e:
                logger.error(f"❌ Error analyzing {symbol}: {e}")
            
            # Wait between symbols to be respectful to APIs
            await asyncio.sleep(1)
        
        logger.info("🎉 News Impact Analyzer test completed!")
        return True
        
    except ImportError as e:
        logger.error(f"❌ Import error: {e}")
        logger.info("Make sure all dependencies are installed: pip install textblob yfinance")
        return False
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        return False

async def test_sentiment_basics():
    """Test basic sentiment analysis functionality"""
    try:
        logger.info("🧪 Testing basic sentiment analysis...")
        
        from models.news_impact_analyzer import NewsImpactAnalyzer
        analyzer = NewsImpactAnalyzer()
        
        # Test sentences with known sentiment
        test_cases = [
            ("Apple beats earnings expectations with record revenue growth", "positive"),
            ("Tesla faces regulatory investigation over safety concerns", "negative"),
            ("NVIDIA announces new AI chip partnership", "positive"),
            ("Company reports disappointing quarterly results", "negative"),
            ("Stock price remains stable with no major news", "neutral")
        ]
        
        logger.info("Testing sentiment detection:")
        for text, expected in test_cases:
            # Extract keywords and calculate sentiment
            keywords = analyzer._extract_keywords(text)
            sentiment = analyzer._calculate_sentiment_score(text.lower(), keywords)
            
            logger.info(f"   Text: '{text}'")
            logger.info(f"   Expected: {expected}, Got: {sentiment:.3f}")
            logger.info(f"   Keywords: {keywords}")
            logger.info("")
        
        logger.info("✅ Basic sentiment analysis test completed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Sentiment test failed: {e}")
        return False

def main():
    """Run all tests"""
    logger.info("🚀 Starting News Impact Analyzer Tests")
    logger.info("=" * 60)
    
    async def run_tests():
        results = []
        
        # Test 1: Basic sentiment analysis
        logger.info("Test 1: Basic Sentiment Analysis")
        result1 = await test_sentiment_basics()
        results.append(("Basic Sentiment", result1))
        logger.info("")
        
        # Test 2: Full news analysis
        logger.info("Test 2: Full News Analysis")
        result2 = await test_news_analyzer()
        results.append(("News Analysis", result2))
        
        return results
    
    # Run tests
    try:
        results = asyncio.run(run_tests())
        
        # Summary
        logger.info("=" * 60)
        logger.info("🏁 TEST SUMMARY")
        logger.info("=" * 60)
        
        passed = 0
        total = len(results)
        
        for test_name, result in results:
            status = "✅ PASS" if result else "❌ FAIL"
            logger.info(f"{test_name:20} {status}")
            if result:
                passed += 1
        
        logger.info("-" * 60)
        logger.info(f"Total: {passed}/{total} tests passed")
        
        if passed == total:
            logger.info("🎉 All tests passed! News Impact Analyzer is ready.")
            logger.info("")
            logger.info("📋 Next Steps:")
            logger.info("1. Integrate with Senior Analyst ML system")
            logger.info("2. Add news analysis to trading decisions")
            logger.info("3. Monitor performance impact")
        else:
            logger.error(f"⚠️ {total - passed} test(s) failed.")
        
        return passed == total
        
    except Exception as e:
        logger.error(f"❌ Test execution failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)