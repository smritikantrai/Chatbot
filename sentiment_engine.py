"""
sentiment_engine.py — VADER Sentiment + Electronics Category Support
"""
import re, string
import pandas as pd
import nltk
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
from nltk.corpus import stopwords
STOP_WORDS = set(stopwords.words('english'))

def clean_text(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'http\S+', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    return re.sub(r'\s+', ' ', text).strip()

def preprocess_dataframe(df):
    col_map = {}
    for col in df.columns:
        l = col.lower().strip()
        if l in ('text','reviewtext','review_body','review','reviews'): col_map[col]='Text'
        elif l in ('score','overall','star_rating','rating','stars','rate'): col_map[col]='Score'
        elif l in ('time','unixreviewtime','date','reviewtime','review_date'): col_map[col]='Time'
        elif l in ('category','categories','product_category'): col_map[col]='category'
        elif l in ('product_name','productname','name','product_title','title'): col_map[col]='product_name'
    df = df.rename(columns=col_map)
    if 'Text' not in df.columns:
        str_cols = [c for c in df.columns if df[c].dtype==object]
        if str_cols: df = df.rename(columns={str_cols[0]:'Text'})
        else: raise ValueError("No text column found!")
    df = df.head(12000).copy()
    df.dropna(subset=['Text'], inplace=True)
    df = df[df['Text'].str.strip()!=''].reset_index(drop=True)
    df['clean_text'] = df['Text'].apply(clean_text)
    if 'Time' in df.columns:
        try:
            df['date'] = pd.to_datetime(df['Time'], unit='s', errors='coerce')
            if df['date'].isna().all():
                df['date'] = pd.to_datetime(df['Time'], errors='coerce')
        except: df['date'] = pd.to_datetime(df['Time'], errors='coerce')
        df['month_year'] = df['date'].dt.to_period('M').astype(str)
        df['month_year'] = df['month_year'].replace('NaT','Unknown')
    else:
        df['date'] = pd.NaT
        df['month_year'] = 'Unknown'
    return df

_analyzer = SentimentIntensityAnalyzer()

def get_sentiment_score(text):
    return _analyzer.polarity_scores(str(text))['compound'] if text else 0.0

def classify_sentiment(score):
    return 'Positive' if score>0.05 else ('Negative' if score<-0.05 else 'Neutral')

def run_sentiment_analysis(df):
    df['sentiment_score'] = df['clean_text'].apply(get_sentiment_score)
    df['sentiment'] = df['sentiment_score'].apply(classify_sentiment)
    return df

def get_sentiment_summary(df):
    total = len(df)
    counts = df['sentiment'].value_counts()
    pos,neg,neu = counts.get('Positive',0), counts.get('Negative',0), counts.get('Neutral',0)
    return {'total':total,'positive':pos,'negative':neg,'neutral':neu,
            'positive_pct':round(pos/total*100,1) if total else 0,
            'negative_pct':round(neg/total*100,1) if total else 0,
            'neutral_pct':round(neu/total*100,1) if total else 0}

def get_monthly_trend(df):
    trend = df[df['month_year']!='Unknown'].copy()
    if trend.empty: return pd.DataFrame()
    pivot = trend.groupby(['month_year','sentiment']).size().unstack(fill_value=0).reset_index()
    return pivot.sort_values('month_year')

def get_category_summary(df):
    if 'category' not in df.columns: return pd.DataFrame()
    return df.groupby(['category','sentiment']).size().unstack(fill_value=0).reset_index()

def get_top_products(df, sentiment='Positive', top_n=5):
    if 'product_name' not in df.columns: return pd.DataFrame()
    filtered = df[df['sentiment']==sentiment]
    top = filtered['product_name'].value_counts().head(top_n).reset_index()
    top.columns = ['Product','Count']
    return top
