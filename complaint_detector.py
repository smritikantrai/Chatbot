import re

COMPLAINT_KEYWORDS = {
    'battery issue': [r'battery', r'charging', r'charge', r'power', r'drain', r'heat', r'overheat'],
    'delivery issue': [r'delivery', r'shipping', r'courier', r'arrived', r'late', r'delay', r'package arrived'],
    'poor quality': [r'quality', r'faulty', r'broken', r'damaged', r'cheap', r'scratch', r'malfunction'],
    'display issue': [r'display', r'screen', r'pixel', r'brightness', r'flicker', r'touchscreen', r'glass'],
    'audio issue': [r'sound', r'audio', r'speaker', r'volume', r'bass', r'noise', r'mic', r'headphone'],
    'packaging issue': [r'packaging', r'package', r'box', r'wrap', r'bubble', r'sealed', r'protect'],
    'refund issue': [r'refund', r'return', r'exchange', r'replacement', r'money back'],
    'service issue': [r'support', r'customer service', r'service', r'help', r'warranty', r'repair', r'care'],
}

CATEGORY_ORDER = [
    'delivery issue', 'battery issue', 'poor quality', 'display issue',
    'audio issue', 'packaging issue', 'refund issue', 'service issue'
]


def _match_category(text):
    text = str(text).lower()
    for category, patterns in COMPLAINT_KEYWORDS.items():
        for pattern in patterns:
            if re.search(pattern, text):
                return category
    return None


def get_complaint_summary(df):
    neg_mask = df['sentiment'] == 'Negative'
    neg_df = df[neg_mask].copy()
    neg_text_series = neg_df['Text'] if 'Text' in neg_df.columns else neg_df.iloc[:, 0]

    categories = {}
    for _, row in neg_df.iterrows():
        text = row.get('clean_text') or row.get('Text') or ''
        category = _match_category(text)
        if category:
            categories[category] = categories.get(category, 0) + 1

    if categories:
        sorted_cats = dict(sorted(categories.items(), key=lambda x: (-x[1], CATEGORY_ORDER.index(x[0]) if x[0] in CATEGORY_ORDER else 999)))
    else:
        sorted_cats = {}

    top_complaint = next(iter(sorted_cats), 'general issues')
    top_count = sorted_cats.get(top_complaint, 0)
    insight = (
        f"The most frequent negative feedback relates to **{top_complaint}**"
        f" with {top_count:,} mentions among negative reviews."
        if top_complaint != 'general issues' else
        "Complaint analysis could not identify a dominant issue from negative reviews."
    )

    return {
        'categories': sorted_cats,
        'neg_text_series': neg_text_series,
        'top_complaint': top_complaint,
        'insight': insight,
    }
