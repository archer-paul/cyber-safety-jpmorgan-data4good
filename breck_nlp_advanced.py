"""
ADVANCED NLP MODULE - Breck Foundation
=======================================
Natural Language Processing for:
1. Feedback text analysis (sentiment, themes, keywords)
2. Cyber safety news trend analysis
3. Topic modeling and classification

Author: Hackathon Team
Date: October 2025
"""

import pandas as pd
import numpy as np
import re
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# TEXT PREPROCESSING
# ============================================================================

class TextPreprocessor:
    """
    Advanced text preprocessing utilities.
    """
    
    # Expanded stopwords for better analysis
    STOPWORDS = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'was',
        'were', 'been', 'be', 'have', 'has', 'had', 'do', 'does', 'did',
        'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can',
        'is', 'am', 'are', 'this', 'that', 'these', 'those', 'i', 'you',
        'he', 'she', 'it', 'we', 'they', 'what', 'which', 'who', 'when',
        'where', 'why', 'how', 'all', 'each', 'every', 'both', 'few', 'more',
        'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own',
        'same', 'so', 'than', 'too', 'very', 's', 't', 'just', 'don', 'now',
        'also', 'as', 'if', 'there', 'their', 'them', 'then', 'get', 'make',
        'go', 'see', 'know', 'take', 'think', 'come', 'give', 'use', 'find',
        'tell', 'ask', 'work', 'seem', 'feel', 'try', 'leave', 'call', 'way',
        'like', 'back', 'look', 'thing', 'much', 'any', 'well', 'said', 'one',
        'two', 'three', 'lot', 'bit', 're'
    }
    
    @staticmethod
    def clean_text(text: str) -> str:
        """
        Clean and normalize text.
        
        Args:
            text: Raw text string
            
        Returns:
            Cleaned text
        """
        if pd.isna(text) or text is None:
            return ""
        
        # Convert to string and lowercase
        text = str(text).lower()
        
        # Remove special characters but keep spaces and basic punctuation
        text = re.sub(r'[^\w\s\.\!\?]', ' ', text)
        
        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    @staticmethod
    def tokenize(text: str, min_length: int = 2, remove_stopwords: bool = True) -> List[str]:
        """
        Tokenize text into words.
        
        Args:
            text: Cleaned text string
            min_length: Minimum word length to keep
            remove_stopwords: Whether to remove stopwords
            
        Returns:
            List of tokens
        """
        words = text.split()
        
        if remove_stopwords:
            words = [w for w in words if w not in TextPreprocessor.STOPWORDS]
        
        words = [w for w in words if len(w) >= min_length]
        
        return words
    
    @staticmethod
    def extract_ngrams(text: str, n: int = 2) -> List[Tuple[str, ...]]:
        """
        Extract n-grams from text.
        
        Args:
            text: Text string
            n: Size of n-grams (2 for bigrams, 3 for trigrams)
            
        Returns:
            List of n-grams
        """
        words = TextPreprocessor.tokenize(text, remove_stopwords=False)
        return [tuple(words[i:i+n]) for i in range(len(words)-n+1)]


# ============================================================================
# FEEDBACK TEXT ANALYZER
# ============================================================================

class FeedbackTextAnalyzer:
    """
    Analyze feedback text from youth, adults, and other stakeholders.
    """
    
    def __init__(self, feedback_df: pd.DataFrame):
        """
        Initialize with feedback DataFrame.
        
        Args:
            feedback_df: DataFrame with text feedback columns
        """
        self.df = feedback_df
        self.preprocessor = TextPreprocessor()
        self.results = {}
    
    def analyze_positive_feedback(self, text_column: str = 'liked_most') -> Dict:
        """
        Analyze what participants liked about the workshops.
        
        Args:
            text_column: Column name containing positive feedback
            
        Returns:
            Dictionary with analysis results
        """
        print(f"Analyzing positive feedback from '{text_column}'...")
        
        if text_column not in self.df.columns:
            print(f"Column '{text_column}' not found")
            return {}
        
        # Get all non-empty responses
        responses = self.df[text_column].dropna()
        
        # Extract keywords
        all_words = []
        for text in responses:
            cleaned = self.preprocessor.clean_text(text)
            words = self.preprocessor.tokenize(cleaned)
            all_words.extend(words)
        
        # Get top keywords
        word_freq = Counter(all_words)
        top_keywords = word_freq.most_common(25)
        
        # Extract bigrams for context
        all_bigrams = []
        for text in responses:
            cleaned = self.preprocessor.clean_text(text)
            bigrams = self.preprocessor.extract_ngrams(cleaned, n=2)
            all_bigrams.extend(bigrams)
        
        bigram_freq = Counter(all_bigrams)
        top_bigrams = bigram_freq.most_common(15)
        
        # Categorize themes
        themes = self._categorize_positive_themes(responses)
        
        results = {
            'total_responses': len(responses),
            'top_keywords': top_keywords,
            'top_bigrams': top_bigrams,
            'themes': themes
        }
        
        print(f"Analyzed {len(responses)} responses")
        print(f"Top 5 keywords: {[k[0] for k in top_keywords[:5]]}")
        print(f"Key themes: {list(themes.keys())}")
        
        return results
    
    def analyze_improvement_suggestions(self, text_column: str = 'improvements') -> Dict:
        """
        Analyze suggestions for improvement.
        
        Args:
            text_column: Column name containing improvement suggestions
            
        Returns:
            Dictionary with analysis results
        """
        print(f"Analyzing improvement suggestions from '{text_column}'...")
        
        if text_column not in self.df.columns:
            print(f"Column '{text_column}' not found")
            return {}
        
        responses = self.df[text_column].dropna()
        
        # Identify "nothing" responses
        nothing_responses = responses[responses.str.lower().str.contains('nothing|no|none|n/a', na=False)]
        substantive_responses = responses[~responses.str.lower().str.contains('^(nothing|no|none|n/?a)$', na=False, regex=True)]
        
        # Extract keywords from substantive suggestions
        all_words = []
        for text in substantive_responses:
            cleaned = self.preprocessor.clean_text(text)
            words = self.preprocessor.tokenize(cleaned)
            all_words.extend(words)
        
        word_freq = Counter(all_words)
        top_keywords = word_freq.most_common(20)
        
        # Categorize improvement themes
        themes = self._categorize_improvement_themes(substantive_responses)
        
        results = {
            'total_responses': len(responses),
            'nothing_responses': len(nothing_responses),
            'substantive_responses': len(substantive_responses),
            'top_keywords': top_keywords,
            'improvement_themes': themes
        }
        
        print(f"Analyzed {len(responses)} responses")
        print(f"'Nothing' responses: {len(nothing_responses)} ({len(nothing_responses)/len(responses)*100:.1f}%)")
        print(f"Key improvement areas: {list(themes.keys())[:5]}")
        
        return results
    
    def perform_sentiment_analysis(self, text_column: str) -> Dict:
        """
        Perform basic sentiment analysis on text.
        
        Args:
            text_column: Column name to analyze
            
        Returns:
            Dictionary with sentiment scores
        """
        print(f"Performing sentiment analysis on '{text_column}'...")
        
        if text_column not in self.df.columns:
            return {}
        
        # Simple lexicon-based sentiment
        positive_words = {
            'good', 'great', 'excellent', 'amazing', 'helpful', 'useful', 'interesting',
            'engaging', 'informative', 'educational', 'important', 'valuable', 'enjoyable',
            'loved', 'liked', 'wonderful', 'fantastic', 'effective', 'clear', 'powerful',
            'relevant', 'impactful', 'necessary', 'appreciated', 'inspiring', 'eye-opening'
        }
        
        negative_words = {
            'boring', 'confusing', 'unclear', 'missing', 'lacking', 'insufficient',
            'poor', 'bad', 'wrong', 'difficult', 'hard', 'complicated', 'too', 'very'
        }
        
        sentiments = []
        for text in self.df[text_column].dropna():
            cleaned = self.preprocessor.clean_text(text)
            words = set(self.preprocessor.tokenize(cleaned, remove_stopwords=False))
            
            pos_count = len(words & positive_words)
            neg_count = len(words & negative_words)
            
            # Calculate sentiment score (-1 to 1)
            if pos_count + neg_count > 0:
                sentiment = (pos_count - neg_count) / (pos_count + neg_count)
            else:
                sentiment = 0
            
            sentiments.append(sentiment)
        
        # Categorize
        positive = sum(1 for s in sentiments if s > 0.2)
        neutral = sum(1 for s in sentiments if -0.2 <= s <= 0.2)
        negative = sum(1 for s in sentiments if s < -0.2)
        
        results = {
            'mean_sentiment': np.mean(sentiments) if sentiments else 0,
            'median_sentiment': np.median(sentiments) if sentiments else 0,
            'std_sentiment': np.std(sentiments) if sentiments else 0,
            'distribution': {
                'positive': positive,
                'neutral': neutral,
                'negative': negative
            }
        }
        
        print(f"Mean sentiment: {results['mean_sentiment']:.3f}")
        print(f"Distribution: {positive} positive, {neutral} neutral, {negative} negative")
        
        return results
    
    def _categorize_positive_themes(self, responses: pd.Series) -> Dict[str, int]:
        """
        Categorize positive feedback into themes.
        """
        themes = defaultdict(int)
        
        # Define theme keywords
        theme_keywords = {
            'Story/Personal': ['story', 'breck', 'personal', 'real', 'life', 'lewis', 'tragic', 'happened'],
            'Grooming': ['grooming', 'groomed', 'groomer', 'manipulation', 'predator', 'warning', 'signs'],
            'Educational': ['learned', 'information', 'knowledge', 'understand', 'aware', 'facts', 'teach'],
            'Safety': ['safety', 'safe', 'protect', 'danger', 'risk', 'careful', 'secure'],
            'Interactive': ['interactive', 'quiz', 'questions', 'discussion', 'activities', 'engaging'],
            'Presenter': ['presenter', 'speaker', 'delivery', 'explained', 'talked', 'presented']
        }
        
        for text in responses:
            cleaned = self.preprocessor.clean_text(text)
            for theme, keywords in theme_keywords.items():
                if any(keyword in cleaned for keyword in keywords):
                    themes[theme] += 1
        
        return dict(sorted(themes.items(), key=lambda x: x[1], reverse=True))
    
    def _categorize_improvement_themes(self, responses: pd.Series) -> Dict[str, int]:
        """
        Categorize improvement suggestions into themes.
        """
        themes = defaultdict(int)
        
        theme_keywords = {
            'More time/depth': ['more', 'longer', 'time', 'detail', 'depth', 'expand', 'elaborate'],
            'More interactivity': ['interactive', 'activities', 'games', 'quiz', 'participation', 'hands-on'],
            'Visuals/videos': ['video', 'visual', 'images', 'picture', 'graphic', 'media', 'clips'],
            'More examples': ['example', 'case', 'scenario', 'situation', 'real-life'],
            'Q&A session': ['question', 'answer', 'ask', 'clarify', 'discuss'],
            'Social media': ['social', 'media', 'platform', 'instagram', 'tiktok', 'snapchat'],
            'Statistics/facts': ['statistic', 'data', 'number', 'fact', 'figure', 'research']
        }
        
        for text in responses:
            cleaned = self.preprocessor.clean_text(text)
            for theme, keywords in theme_keywords.items():
                if any(keyword in cleaned for keyword in keywords):
                    themes[theme] += 1
        
        return dict(sorted(themes.items(), key=lambda x: x[1], reverse=True))
    
    def generate_word_cloud_data(self, text_column: str, max_words: int = 100) -> List[Dict]:
        """
        Generate data for word cloud visualization.
        
        Args:
            text_column: Column to analyze
            max_words: Maximum number of words to return
            
        Returns:
            List of dictionaries with word and frequency
        """
        if text_column not in self.df.columns:
            return []
        
        all_words = []
        for text in self.df[text_column].dropna():
            cleaned = self.preprocessor.clean_text(text)
            words = self.preprocessor.tokenize(cleaned)
            all_words.extend(words)
        
        word_freq = Counter(all_words)
        
        return [
            {'word': word, 'frequency': count}
            for word, count in word_freq.most_common(max_words)
        ]


# ============================================================================
# CYBER SAFETY NEWS ANALYZER
# ============================================================================

class CyberSafetyNewsAnalyzer:
    """
    Analyze cyber safety news trends to identify emerging risks and topics.
    This module can integrate with news APIs or analyze provided news data.
    """
    
    # Key cyber safety topics to track
    SAFETY_TOPICS = {
        'online_grooming': ['grooming', 'predator', 'exploitation', 'child abuse', 'sextortion'],
        'cyberbullying': ['cyberbully', 'harassment', 'trolling', 'online abuse', 'hate'],
        'privacy': ['privacy', 'data protection', 'personal information', 'tracking'],
        'social_media': ['social media', 'instagram', 'tiktok', 'snapchat', 'facebook'],
        'gaming': ['gaming', 'online game', 'minecraft', 'fortnite', 'roblox'],
        'ai_risks': ['ai', 'artificial intelligence', 'deepfake', 'chatbot', 'generative'],
        'misinformation': ['misinformation', 'fake news', 'disinformation', 'conspiracy'],
        'mental_health': ['mental health', 'anxiety', 'depression', 'wellbeing', 'self-harm'],
        'sexting': ['sexting', 'nude', 'explicit', 'image sharing'],
        'identity_theft': ['identity theft', 'fraud', 'scam', 'phishing']
    }
    
    def __init__(self, news_data: Optional[pd.DataFrame] = None):
        """
        Initialize news analyzer.
        
        Args:
            news_data: DataFrame with news articles (columns: date, title, content, source)
        """
        self.news_data = news_data
        self.preprocessor = TextPreprocessor()
        self.results = {}
    
    def classify_article_topics(self, text: str) -> List[str]:
        """
        Classify an article into relevant safety topics.
        
        Args:
            text: Article text
            
        Returns:
            List of relevant topic categories
        """
        cleaned = self.preprocessor.clean_text(text)
        topics = []
        
        for topic, keywords in self.SAFETY_TOPICS.items():
            if any(keyword in cleaned for keyword in keywords):
                topics.append(topic)
        
        return topics
    
    def analyze_news_trends(self) -> Dict:
        """
        Analyze trends in cyber safety news coverage.
        
        Returns:
            Dictionary with trend analysis
        """
        if self.news_data is None or len(self.news_data) == 0:
            print("No news data provided")
            return {}
        
        print("Analyzing cyber safety news trends...")
        
        # Classify all articles
        topic_counts = defaultdict(int)
        
        for idx, row in self.news_data.iterrows():
            text = f"{row.get('title', '')} {row.get('content', '')}"
            topics = self.classify_article_topics(text)
            for topic in topics:
                topic_counts[topic] += 1
        
        # Calculate trends over time if dates available
        temporal_trends = {}
        if 'date' in self.news_data.columns:
            self.news_data['date'] = pd.to_datetime(self.news_data['date'], errors='coerce')
            self.news_data['month'] = self.news_data['date'].dt.to_period('M')
            
            # Count articles per month
            temporal_trends = self.news_data.groupby('month').size().to_dict()
        
        results = {
            'total_articles': len(self.news_data),
            'topic_frequency': dict(sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)),
            'temporal_trends': temporal_trends
        }
        
        print(f"Analyzed {len(self.news_data)} articles")
        print(f"Top topics: {list(results['topic_frequency'].keys())[:5]}")
        
        self.results['news_trends'] = results
        return results
    
    def identify_emerging_risks(self, lookback_months: int = 6) -> List[str]:
        """
        Identify emerging cyber safety risks based on recent news.
        
        Args:
            lookback_months: Number of months to analyze for trends
            
        Returns:
            List of emerging risk topics
        """
        if self.news_data is None or 'date' not in self.news_data.columns:
            return []
        
        recent_cutoff = pd.Timestamp.now() - pd.DateOffset(months=lookback_months)
        recent_news = self.news_data[self.news_data['date'] >= recent_cutoff]
        
        # Analyze recent vs historical topic distribution
        # This is a simplified version - could be more sophisticated
        emerging_risks = []
        
        return emerging_risks
    
    def generate_recommendations(self, workshop_feedback: Dict) -> List[str]:
        """
        Generate recommendations for workshop content based on news trends.
        
        Args:
            workshop_feedback: Results from feedback analysis
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        if 'news_trends' in self.results:
            top_topics = list(self.results['news_trends']['topic_frequency'].keys())[:5]
            
            for topic in top_topics:
                if topic == 'ai_risks':
                    recommendations.append(
                        "Consider adding content about AI risks (deepfakes, AI chatbots) - emerging trend in news"
                    )
                elif topic == 'gaming':
                    recommendations.append(
                        "Expand gaming safety section - high media coverage of gaming-related risks"
                    )
                # Add more topic-specific recommendations
        
        return recommendations


# ============================================================================
# COMPREHENSIVE NLP ORCHESTRATOR
# ============================================================================

def run_complete_nlp_analysis(feedback_df: pd.DataFrame, 
                              news_df: Optional[pd.DataFrame] = None) -> Dict:
    """
    Run complete NLP analysis pipeline.
    
    Args:
        feedback_df: DataFrame with feedback data
        news_df: Optional DataFrame with news articles
        
    Returns:
        Dictionary with all NLP analysis results
    """
    print("=" * 80)
    print("COMPREHENSIVE NLP ANALYSIS")
    print("=" * 80)
    print()
    
    results = {
        'timestamp': pd.Timestamp.now().isoformat(),
        'feedback_analysis': {},
        'news_analysis': {}
    }
    
    # Analyze feedback text
    feedback_analyzer = FeedbackTextAnalyzer(feedback_df)
    
    if 'liked_most' in feedback_df.columns:
        results['feedback_analysis']['positive'] = feedback_analyzer.analyze_positive_feedback('liked_most')
        results['feedback_analysis']['positive_sentiment'] = feedback_analyzer.perform_sentiment_analysis('liked_most')
    
    if 'improvements' in feedback_df.columns:
        results['feedback_analysis']['improvements'] = feedback_analyzer.analyze_improvement_suggestions('improvements')
    
    if 'additions' in feedback_df.columns:
        results['feedback_analysis']['additions'] = feedback_analyzer.analyze_improvement_suggestions('additions')
    
    # Analyze news if provided
    if news_df is not None:
        news_analyzer = CyberSafetyNewsAnalyzer(news_df)
        results['news_analysis'] = news_analyzer.analyze_news_trends()
        results['recommendations'] = news_analyzer.generate_recommendations(results['feedback_analysis'])
    
    print("\n" + "=" * 80)
    print("NLP ANALYSIS COMPLETE")
    print("=" * 80)
    
    return results


# ============================================================================
# EXPORT UTILITIES
# ============================================================================

def export_nlp_results(results: Dict, output_path: str = '/home/claude/breck_nlp_results.json'):
    """
    Export NLP results to JSON file.
    
    Args:
        results: Dictionary with NLP analysis results
        output_path: Path for output file
    """
    import json
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nNLP results exported to: {output_path}")
    return output_path


if __name__ == "__main__":
    print("Advanced NLP Module for Breck Foundation")
    print("This module should be imported and used with the main framework")
