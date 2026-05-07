"""
chatbot_engine.py — Electronics-focused BI Chatbot
"""

INTENTS = {
    'overall_sentiment':['overall sentiment','general sentiment','how do customers feel','what is the sentiment','customer opinion','overall feeling','how are reviews','what do people think','sentiment overview'],
    'positive_ratio':['how many positive','positive review','positive percentage','positive ratio','good reviews','happy customers','satisfied','positive count'],
    'negative_ratio':['how many negative','negative review','negative percentage','bad reviews','unhappy','dissatisfied','negative count'],
    'neutral_ratio':['neutral','how many neutral','neutral review','neutral percentage','mixed reviews'],
    'top_complaints':['top complaint','common complaint','main complaint','biggest complaint','what are complaints','customer complaints','issues reported','most common issue','problems','top issue','what issues'],
    'worst_month':['worst month','bad month','most negative month','which month','lowest sentiment','worst period'],
    'best_month':['best month','most positive month','highest sentiment','best period'],
    'monthly_trend':['monthly trend','trend','over time','month by month','sentiment trend','monthly sentiment','timeline'],
    'recommendations':['recommend','suggestion','improve','business advice','what should','how to improve','action','what can be done','business improvement','what to do'],
    'total_reviews':['total review','how many review','number of review','count of review','review count','total count'],
    'top_products':['top product','best product','most reviewed','popular product','which product','product performance'],
    'battery':['battery','battery life','battery drain','charging issue','charge problem'],
    'delivery':['delivery','shipping','arrived late','transit','dispatch','courier'],
    'quality':['quality','build quality','material','durability','cheap','flimsy','poor quality'],
    'display':['display','screen','resolution','brightness','pixels'],
    'audio':['sound','audio','speaker','bass','volume','headphone'],
    'laptop':['laptop','macbook','notebook','computer','processor','ram','storage','ssd'],
    'greeting':['hello','hi','hey','good morning','good evening','greetings','howdy'],
    'help':['help','what can you do','what can i ask','commands','capabilities','features','options'],
}

def _match_intent(query):
    q = query.lower().strip()
    for intent, patterns in INTENTS.items():
        for p in patterns:
            if p in q:
                return intent
    return 'unknown'

def generate_response(query, summary, complaint_data, monthly_df, df=None):
    intent = _match_intent(query)

    if intent == 'greeting':
        return ("👋 Hello! I'm your **Electronics Review Intelligence Agent**.\n\n"
                "I analyse customer reviews from Flipkart, Amazon & other platforms.\n\n"
                "Try asking:\n- *What is the overall sentiment?*\n- *What are the top complaints?*\n"
                "- *Which month was worst?*\n- *Tell me about battery issues*\n- *Top products?*")

    if intent == 'help':
        return ("🤖 **I can answer:**\n\n"
                "- Overall / positive / negative sentiment\n"
                "- Total review count\n"
                "- Top customer complaints\n"
                "- Battery, display, audio, delivery issues\n"
                "- Best & worst months\n"
                "- Top rated products\n"
                "- Business improvement recommendations\n\n"
                "Just ask naturally in English!")

    if not summary:
        return "⚠️ Data not loaded yet. Please wait for the analysis to complete."

    total = summary['total']
    pos, neg, neu = summary['positive_pct'], summary['negative_pct'], summary['neutral_pct']

    if intent == 'overall_sentiment':
        dominant = max([('Positive',pos),('Negative',neg),('Neutral',neu)], key=lambda x:x[1])[0]
        emoji = {'Positive':'😊','Negative':'😟','Neutral':'😐'}[dominant]
        return (f"{emoji} **Overall customer sentiment for electronics is {dominant}.**\n\n"
                f"| Sentiment | % | Count |\n|---|---|---|\n"
                f"| 😊 Positive | {pos}% | {summary['positive']:,} |\n"
                f"| 😟 Negative | {neg}% | {summary['negative']:,} |\n"
                f"| 😐 Neutral | {neu}% | {summary['neutral']:,} |")

    if intent == 'total_reviews':
        return f"📊 **{total:,} electronics reviews** were analysed across Flipkart, Amazon & other platforms."

    if intent == 'positive_ratio':
        return (f"😊 **{pos}%** of electronics reviews are Positive ({summary['positive']:,} reviews).\n\n"
                "Customers are generally happy with product features, design and performance.")

    if intent == 'negative_ratio':
        return (f"😟 **{neg}%** of reviews are Negative ({summary['negative']:,} reviews).\n\n"
                "Key issues include delivery problems, battery complaints and quality concerns.")

    if intent == 'neutral_ratio':
        return f"😐 **{neu}%** of reviews are Neutral ({summary['neutral']:,} reviews) — mixed experiences."

    if intent == 'top_complaints':
        cats = complaint_data.get('categories', {})
        if not cats: return "No complaint data available."
        top5 = list(cats.items())[:5]
        lines = "\n".join([f"  {i+1}. **{k.title()}** — {v} mentions" for i,(k,v) in enumerate(top5)])
        return f"🚨 **Top Electronics Customer Complaints:**\n\n{lines}"

    if intent == 'battery':
        return ("🔋 **Battery Issues in Electronics Reviews:**\n\n"
                "Battery complaints are among the most common issues reported. Customers frequently mention:\n"
                "- Rapid battery drain after a few months\n- Slow charging speeds\n"
                "- Battery not lasting advertised duration\n- Swelling/overheating issues\n\n"
                "💡 *Recommendation: Source batteries from certified suppliers; offer 1-year battery warranty.*")

    if intent == 'delivery':
        return ("🚚 **Delivery Issues in Electronics Reviews:**\n\n"
                "Delivery complaints include:\n- Late or missed deliveries\n"
                "- Products arriving damaged due to poor packaging\n- Wrong items delivered\n\n"
                "💡 *Recommendation: Partner with reliable courier services; add real-time tracking.*")

    if intent == 'quality':
        return ("🔍 **Quality Issues in Electronics Reviews:**\n\n"
                "Quality complaints include:\n- Build quality feeling cheap or flimsy\n"
                "- Products not matching specifications\n- Early failure within warranty period\n\n"
                "💡 *Recommendation: Implement pre-shipment 3-point quality inspection.*")

    if intent == 'display':
        return ("🖥️ **Display/Screen Issues in Electronics Reviews:**\n\n"
                "Display complaints include:\n- Dead pixels or screen flickering\n"
                "- Poor brightness in sunlight\n- Colour accuracy issues\n\n"
                "💡 *Recommendation: Enforce display panel quality checks before dispatch.*")

    if intent == 'audio':
        return ("🔊 **Audio/Speaker Issues in Electronics Reviews:**\n\n"
                "Audio complaints include:\n- Poor bass quality\n- Low maximum volume\n"
                "- Distortion at high volumes\n- Bluetooth connectivity drops\n\n"
                "💡 *Recommendation: Partner with certified audio component suppliers.*")

    if intent == 'laptop':
        return ("💻 **Laptop Category Insights:**\n\n"
                "Laptop reviews frequently mention:\n- Performance: RAM & processor speed\n"
                "- Battery life as a key purchase factor\n- Build quality and keyboard feel\n"
                "- Value for money comparisons (MacBook vs Windows)\n\n"
                "Top brands with positive reviews include Apple MacBook, Dell, HP and Lenovo.")

    if intent == 'top_products':
        if df is not None and 'product_name' in df.columns:
            from sentiment_engine import get_top_products
            top = get_top_products(df, 'Positive', 5)
            if not top.empty:
                lines = "\n".join([f"  {i+1}. {row['Product'][:60]} — {row['Count']} positive reviews"
                                   for i, row in top.iterrows()])
                return f"🏆 **Top 5 Products by Positive Reviews:**\n\n{lines}"
        return "Product data not available in current dataset."

    if intent == 'worst_month':
        if monthly_df is None or monthly_df.empty:
            return "Monthly trend not available — date column missing or not parseable in this dataset."
        if 'Negative' not in monthly_df.columns:
            return "Negative sentiment column not found in monthly data."
        idx = monthly_df['Negative'].idxmax()
        m, c = monthly_df.loc[idx,'month_year'], monthly_df.loc[idx,'Negative']
        return (f"📉 **Worst month: {m}** with **{c:,} negative reviews**.\n\n"
                "Investigate product batches, supplier changes or logistics issues during this period.")

    if intent == 'best_month':
        if monthly_df is None or monthly_df.empty:
            return "Monthly trend not available."
        if 'Positive' not in monthly_df.columns:
            return "Positive column not found."
        idx = monthly_df['Positive'].idxmax()
        m, c = monthly_df.loc[idx,'month_year'], monthly_df.loc[idx,'Positive']
        return f"📈 **Best month: {m}** with **{c:,} positive reviews**! 🎉"

    if intent == 'monthly_trend':
        if monthly_df is None or monthly_df.empty:
            return "Monthly trend not available — date column could not be parsed."
        first, last = monthly_df['month_year'].iloc[0], monthly_df['month_year'].iloc[-1]
        return (f"📅 Monthly trend covers **{len(monthly_df)} months** from **{first}** to **{last}**.\n\n"
                "View the **Monthly Sentiment Trend** chart in the Dashboard tab.")

    if intent == 'recommendations':
        top_cat = complaint_data.get('top_complaint','general issues')
        return _build_recommendations(neg, top_cat)

    return ("🤔 I didn't quite catch that. Try:\n"
            "- *'Overall sentiment?'*\n- *'Top complaints?'*\n"
            "- *'Battery issues?'*\n- *'Worst month?'*\n- *'Recommendations?'*")

def _build_recommendations(neg_pct, top_complaint):
    mapping = {
        'delivery issue': "🚚 **Logistics** — Partner with reliable couriers; add real-time GPS tracking.",
        'damaged product': "📦 **Packaging** — Use double-walled boxes + foam inserts for electronics.",
        'poor quality': "🔍 **Quality Control** — Pre-shipment 3-point inspection; quarterly supplier audits.",
        'battery issue': "🔋 **Battery QA** — Source from certified suppliers; offer extended warranty.",
        'refund issue': "💰 **Returns** — One-click 30-day return portal; 48hr refund processing.",
        'packaging issue': "🎁 **Packaging Standards** — Tamper-evident seals + moisture-resistant wraps.",
        'service issue': "🎧 **Support** — Empathy-first training; <2 min live chat response time.",
    }
    recs = []
    if top_complaint in mapping:
        recs.append(mapping[top_complaint])
    if neg_pct > 30:
        recs.append("⚠️ **Urgent** — 30%+ negative rate signals systemic quality issues. Immediate review needed.")
    elif neg_pct > 15:
        recs.append("📊 **Monitor** — Negative rate above 15%. Set up weekly sentiment dashboards.")
    else:
        recs.append("✅ **Maintain** — Healthy negative rate. Focus on converting neutrals to positives.")
    recs += [
        "🌟 **Loyalty Programme** — Reward repeat positive reviewers with discount coupons.",
        "📧 **Post-purchase Email** — Ask for feedback 3 days after delivery.",
        "🤖 **AI Support Bot** — Handle top 10 FAQs to reduce ticket load by 40%.",
    ]
    result = "💡 **Business Improvement Recommendations:**\n\n"
    result += "\n".join(f"{i+1}. {r}" for i,r in enumerate(recs))
    return result
