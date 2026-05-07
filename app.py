"""
app.py — AI-Powered Electronics Review Intelligence Agent
Hardcoded dataset: merged Flipkart + Amazon electronics reviews
Run: streamlit run app.py
"""

import os, warnings
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from wordcloud import WordCloud

warnings.filterwarnings('ignore')

from sentiment_engine import (preprocess_dataframe, run_sentiment_analysis,
    get_sentiment_summary, get_monthly_trend, get_category_summary, get_top_products)
from complaint_detector import get_complaint_summary
from chatbot_engine import generate_response

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Electronics Review Agent", page_icon="⚡",
                   layout="wide", initial_sidebar_state="expanded")

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');
html,body,[class*="css"]{font-family:'Sora',sans-serif;}
.stApp{background:linear-gradient(135deg,#0a0e1a 0%,#0d1b2a 50%,#0a1628 100%);color:#e8eaf0;}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#0d1b2a,#091422);border-right:1px solid #1e3a5f;}
[data-testid="stSidebar"] *{color:#c8d8e8!important;}
.main-header{background:linear-gradient(90deg,#0d1b2a,#1a3a5c,#0d1b2a);border:1px solid #00d4ff;border-radius:16px;padding:28px 36px;margin-bottom:28px;}
.main-header h1{font-size:2.1rem;font-weight:700;color:#00d4ff;margin:0;}
.main-header p{color:#8aabb8;margin:6px 0 0;font-size:0.93rem;}
.kpi-card{background:linear-gradient(135deg,#0d1b2a,#122540);border:1px solid #1e3a5f;border-radius:14px;padding:22px 20px;text-align:center;transition:border-color 0.3s,transform 0.2s;position:relative;overflow:hidden;}
.kpi-card:hover{border-color:#00d4ff;transform:translateY(-2px);}
.kpi-card::after{content:'';position:absolute;bottom:0;left:0;right:0;height:3px;}
.kpi-pos::after{background:linear-gradient(90deg,#22c55e,#86efac);}
.kpi-neg::after{background:linear-gradient(90deg,#ef4444,#fca5a5);}
.kpi-neu::after{background:linear-gradient(90deg,#f59e0b,#fde68a);}
.kpi-tot::after{background:linear-gradient(90deg,#00d4ff,#0ea5e9);}
.kpi-value{font-size:2.4rem;font-weight:700;font-family:'JetBrains Mono',monospace;line-height:1;}
.kpi-label{font-size:0.78rem;color:#8aabb8;text-transform:uppercase;letter-spacing:1.5px;margin-top:6px;}
.kpi-icon{font-size:1.5rem;margin-bottom:8px;}
.section-header{font-size:1.1rem;font-weight:600;color:#00d4ff;border-left:3px solid #00d4ff;padding-left:12px;margin:24px 0 14px;letter-spacing:0.3px;}
.chat-user{background:linear-gradient(135deg,#1a3a5c,#1e4570);border:1px solid #2563a8;border-radius:16px 16px 4px 16px;padding:12px 18px;margin:8px 0 8px 60px;color:#e8f0f8;font-size:0.92rem;}
.chat-bot{background:linear-gradient(135deg,#0d2010,#0f2a14);border:1px solid #166534;border-radius:16px 16px 16px 4px;padding:12px 18px;margin:8px 60px 8px 0;color:#d1fae5;font-size:0.92rem;}
.chat-label-user{text-align:right;font-size:0.7rem;color:#6090b8;margin-bottom:2px;}
.chat-label-bot{font-size:0.7rem;color:#4a9e68;margin-bottom:2px;}
.rec-card{background:linear-gradient(135deg,#0f1e0a,#152b10);border:1px solid #2d6a2d;border-radius:12px;padding:16px 20px;margin-bottom:10px;color:#bbf7d0;font-size:0.88rem;}
.badge{display:inline-block;padding:3px 10px;border-radius:20px;font-size:0.75rem;font-weight:600;margin-right:4px;}
.badge-pos{background:#052e16;color:#4ade80;border:1px solid #166534;}
.badge-neg{background:#2d0a0a;color:#f87171;border:1px solid #991b1b;}
.badge-neu{background:#1c1507;color:#fbbf24;border:1px solid #92400e;}
.stButton>button{
    background:#0d1b2a !important;
    color:#c8d8e8 !important;
    border:1px solid #00d4ff !important;
    border-radius:8px !important;
    font-family:'Sora',sans-serif !important;
    font-size:0.85rem !important;
}
.stButton>button:hover{
    background:#1a3a5c !important;
    color:#ffffff !important;
    border-color:#ffffff !important;
}
#MainMenu,footer,header{visibility:hidden;}
</style>
""", unsafe_allow_html=True)

# ── Dataset path — HARDCODED to bundled dataset ───────────────────────────────
DATASET_PATH = os.path.join(os.path.dirname(__file__), "data", "amazon_reviews.csv")

# ── Load & analyse ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_and_analyse(filepath):
    raw = pd.read_csv(filepath)
    df  = preprocess_dataframe(raw)
    df  = run_sentiment_analysis(df)
    summary       = get_sentiment_summary(df)
    monthly_trend = get_monthly_trend(df)
    complaint_data= get_complaint_summary(df)
    return df, summary, monthly_trend, complaint_data

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚡ Electronics Agent")
    st.markdown("---")
    st.markdown("**Dataset:** Flipkart + Amazon\nElectronics Reviews")
    st.markdown("**Rows:** Up to 12,000")
    st.markdown("**Engine:** VADER NLP")
    st.markdown("---")
    st.markdown("### 🧭 Navigation")
    page = st.radio("Go to",
        ["📊 Dashboard","💬 Chatbot","📋 Dataset Preview","💡 Recommendations"],
        label_visibility="collapsed")
    st.markdown("---")
    st.markdown("<small style='color:#4a7090'>AI Electronics Review Agent<br>Built for Academic Assignment</small>", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
  <h1>⚡ Electronics Review Intelligence Agent</h1>
  <p>Flipkart &amp; Amazon Electronics · Sentiment Analysis · Complaint Detection · Business Intelligence Chatbot</p>
</div>
""", unsafe_allow_html=True)

# ── Load data ──────────────────────────────────────────────────────────────────
df = summary = monthly_trend = complaint_data = None
data_ready = False

if os.path.exists(DATASET_PATH):
    with st.spinner("⚙️ Analysing 12,000 electronics reviews… please wait ~20 seconds on first run."):
        try:
            df, summary, monthly_trend, complaint_data = load_and_analyse(DATASET_PATH)
            data_ready = True
        except Exception as e:
            st.error(f"❌ Error: {e}")
else:
    st.error(f"❌ Dataset not found at `{DATASET_PATH}`. Make sure `data/amazon_reviews.csv` exists.")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
if page == "📊 Dashboard" and data_ready:

    # KPIs
    st.markdown('<div class="section-header">📈 Key Performance Indicators</div>', unsafe_allow_html=True)
    k1,k2,k3,k4 = st.columns(4)
    def kpi(icon,val,label,cls):
        return f'<div class="kpi-card {cls}"><div class="kpi-icon">{icon}</div><div class="kpi-value">{val}</div><div class="kpi-label">{label}</div></div>'
    with k1: st.markdown(kpi("📦",f"{summary['total']:,}","Total Reviews","kpi-tot"), unsafe_allow_html=True)
    with k2: st.markdown(kpi("😊",f"{summary['positive_pct']}%","Positive","kpi-pos"), unsafe_allow_html=True)
    with k3: st.markdown(kpi("😟",f"{summary['negative_pct']}%","Negative","kpi-neg"), unsafe_allow_html=True)
    with k4: st.markdown(kpi("😐",f"{summary['neutral_pct']}%","Neutral","kpi-neu"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Row 1: Pie + Monthly trend
    ca, cb = st.columns([1,2])
    with ca:
        st.markdown('<div class="section-header">🥧 Sentiment Distribution</div>', unsafe_allow_html=True)
        fig_pie = px.pie(
            names=['Positive','Negative','Neutral'],
            values=[summary['positive'],summary['negative'],summary['neutral']],
            color_discrete_map={'Positive':'#22c55e','Negative':'#ef4444','Neutral':'#f59e0b'},
            hole=0.45)
        fig_pie.update_layout(paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#c8d8e8',family='Sora'),legend=dict(orientation='h',y=-0.2),margin=dict(t=10,b=10))
        st.plotly_chart(fig_pie, use_container_width=True)

    with cb:
        st.markdown('<div class="section-header">📅 Monthly Sentiment Trend</div>', unsafe_allow_html=True)
        if monthly_trend is not None and not monthly_trend.empty:
            fig_line = go.Figure()
            for sent, color in [('Positive','#22c55e'),('Negative','#ef4444'),('Neutral','#f59e0b')]:
                if sent in monthly_trend.columns:
                    fig_line.add_trace(go.Scatter(x=monthly_trend['month_year'],y=monthly_trend[sent],
                        name=sent,mode='lines+markers',line=dict(color=color,width=2.5),marker=dict(size=5)))
            fig_line.update_layout(paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#c8d8e8',family='Sora'),
                xaxis=dict(gridcolor='#1e3a5f'),yaxis=dict(gridcolor='#1e3a5f'),
                legend=dict(orientation='h',y=1.05),margin=dict(t=10,b=40),hovermode='x unified')
            st.plotly_chart(fig_line, use_container_width=True)
        else:
            st.info("Monthly trend not available — date column could not be parsed in this dataset.")

    # Row 2: Complaints + WordCloud
    cc, cd = st.columns([1,1])
    with cc:
        st.markdown('<div class="section-header">🚨 Top Complaint Categories</div>', unsafe_allow_html=True)
        cats = complaint_data.get('categories',{})
        if cats:
            top_cats = dict(list(cats.items())[:7])
            fig_bar = px.bar(x=list(top_cats.values()),y=list(top_cats.keys()),orientation='h',
                color=list(top_cats.values()),
                color_continuous_scale=[[0,'#1e3a5f'],[0.5,'#00d4ff'],[1,'#ef4444']],
                labels={'x':'Mentions','y':''})
            fig_bar.update_layout(paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#c8d8e8',family='Sora'),coloraxis_showscale=False,
                yaxis=dict(autorange='reversed'),margin=dict(t=10,b=10),xaxis=dict(gridcolor='#1e3a5f'))
            st.plotly_chart(fig_bar, use_container_width=True)

    with cd:
        st.markdown('<div class="section-header">☁️ Negative Review Word Cloud</div>', unsafe_allow_html=True)
        neg_series = complaint_data.get('neg_text_series')
        if neg_series is not None and len(neg_series) > 0:
            neg_text = ' '.join(neg_series.dropna().astype(str).tolist())
            wc = WordCloud(width=700,height=350,background_color=None,mode='RGBA',
                colormap='RdYlBu_r',max_words=120,prefer_horizontal=0.8).generate(neg_text)
            fig_wc, ax = plt.subplots(figsize=(7,3.5))
            fig_wc.patch.set_alpha(0.0)
            ax.imshow(wc,interpolation='bilinear'); ax.axis('off')
            st.pyplot(fig_wc, use_container_width=True)
            plt.close(fig_wc)

    # Row 3: Top products + Category breakdown
    ce, cf = st.columns([1,1])
    with ce:
        st.markdown('<div class="section-header">🏆 Top Products — Positive Reviews</div>', unsafe_allow_html=True)
        from sentiment_engine import get_top_products
        top_pos = get_top_products(df, 'Positive', 8)
        if not top_pos.empty:
            top_pos['Product_short'] = top_pos['Product'].str[:45]
            fig_top = px.bar(top_pos,x='Count',y='Product_short',orientation='h',
                color='Count',color_continuous_scale=[[0,'#052e16'],[1,'#22c55e']],
                labels={'Count':'Positive Reviews','Product_short':''})
            fig_top.update_layout(paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#c8d8e8',family='Sora'),coloraxis_showscale=False,
                yaxis=dict(autorange='reversed'),margin=dict(t=10,b=10),xaxis=dict(gridcolor='#1e3a5f'))
            st.plotly_chart(fig_top, use_container_width=True)

    with cf:
        st.markdown('<div class="section-header">📦 Sentiment by Category</div>', unsafe_allow_html=True)
        from sentiment_engine import get_category_summary
        cat_summary = get_category_summary(df)
        if not cat_summary.empty:
            fig_cat = go.Figure()
            colors = {'Positive':'#22c55e','Negative':'#ef4444','Neutral':'#f59e0b'}
            for sent in ['Positive','Negative','Neutral']:
                if sent in cat_summary.columns:
                    fig_cat.add_trace(go.Bar(name=sent,x=cat_summary['category'],
                        y=cat_summary[sent],marker_color=colors[sent]))
            fig_cat.update_layout(barmode='group',paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',font=dict(color='#c8d8e8',family='Sora'),
                legend=dict(orientation='h',y=1.05),margin=dict(t=10,b=40),
                xaxis=dict(gridcolor='#1e3a5f'),yaxis=dict(gridcolor='#1e3a5f'))
            st.plotly_chart(fig_cat, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: CHATBOT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "💬 Chatbot":
    st.markdown('<div class="section-header">💬 Electronics Business Intelligence Chatbot</div>', unsafe_allow_html=True)
    st.markdown("<small style='color:#6090b8'>Ask anything about electronics reviews — sentiment, battery issues, delivery, top products, recommendations…</small>", unsafe_allow_html=True)

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = [{'role':'bot','text':(
            "👋 Hello! I'm your **Electronics Review Intelligence Agent**.\n\n"
            "I've analysed thousands of Flipkart & Amazon electronics reviews.\n\n"
            "Ask me:\n- *Overall sentiment?*\n- *Battery issues?*\n"
            "- *Top complaints?*\n- *Which month was worst?*\n- *Business recommendations?*"
        )}]

    for msg in st.session_state.chat_history:
        if msg['role']=='user':
            st.markdown(f'<div class="chat-label-user">You</div><div class="chat-user">{msg["text"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-label-bot">⚡ Agent</div><div class="chat-bot">{msg["text"]}</div>', unsafe_allow_html=True)

    # Quick buttons
    st.markdown("<br>**Quick Queries:**", unsafe_allow_html=True)
    qq = ["Overall sentiment?","Top complaints?","Battery issues?","Worst month?","Top products?","Recommendations?"]
    cols = st.columns(3)
    for i,q in enumerate(qq):
        with cols[i%3]:
            if st.button(q, use_container_width=True, key=f"qq_{i}"):
                st.session_state.chat_history.append({'role':'user','text':q})
                r = generate_response(q, summary or {}, complaint_data or {}, monthly_trend, df)
                st.session_state.chat_history.append({'role':'bot','text':r})
                st.rerun()

    with st.form("chat_form", clear_on_submit=True):
        ci, cb2 = st.columns([5,1])
        with ci:
            user_input = st.text_input("Your message", placeholder="Ask about electronics reviews…", label_visibility="collapsed")
        with cb2:
            submitted = st.form_submit_button("Send ➤", use_container_width=True)

    if submitted and user_input.strip():
        st.session_state.chat_history.append({'role':'user','text':user_input.strip()})
        reply = generate_response(user_input.strip(), summary or {}, complaint_data or {}, monthly_trend, df)
        st.session_state.chat_history.append({'role':'bot','text':reply})
        st.rerun()

    if st.button("🗑 Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: DATASET PREVIEW
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📋 Dataset Preview":
    st.markdown('<div class="section-header">📋 Dataset Preview</div>', unsafe_allow_html=True)
    if data_ready:
        c1,c2,c3 = st.columns(3)
        c1.metric("Total Rows",f"{len(df):,}")
        c2.metric("Columns",len(df.columns))
        c3.metric("Categories", df['category'].nunique() if 'category' in df.columns else "N/A")

        st.markdown("#### First 100 Rows")
        show = [c for c in ['product_name','category','Text','Score','sentiment','sentiment_score','month_year'] if c in df.columns]
        st.dataframe(df[show].head(100), use_container_width=True, height=400)

        st.markdown("#### Sentiment Breakdown")
        bd = df['sentiment'].value_counts().reset_index()
        bd.columns = ['Sentiment','Count']
        bd['%'] = (bd['Count']/len(df)*100).round(2).astype(str)+'%'
        st.dataframe(bd, use_container_width=True)

        if 'category' in df.columns:
            st.markdown("#### Reviews by Category")
            cat_counts = df['category'].value_counts().reset_index()
            cat_counts.columns = ['Category','Reviews']
            st.dataframe(cat_counts, use_container_width=True)
    else:
        st.warning("Dataset not loaded.")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: RECOMMENDATIONS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "💡 Recommendations":
    st.markdown('<div class="section-header">💡 Business Improvement Recommendations</div>', unsafe_allow_html=True)
    if data_ready:
        st.markdown(f"**Based on {summary['total']:,} electronics reviews | {summary['negative_pct']}% negative**")
        st.markdown("---")

        top_cat = complaint_data.get('top_complaint','general issues')
        rec_map = {
            'delivery issue':  ("🚚","Logistics Overhaul","Partner with multiple courier services. Add real-time GPS tracking. Send proactive delay notifications via SMS/email."),
            'damaged product': ("📦","Packaging Reinforcement","Use double-walled corrugated boxes for electronics. Add foam inserts and 'FRAGILE' labels. Shrink-wrap all items before boxing."),
            'poor quality':    ("🔍","Quality Control","Implement a pre-shipment 3-point quality checklist. Enforce quarterly supplier certification audits. Introduce product batch testing."),
            'battery issue':   ("🔋","Battery QA Programme","Source only from ISO-certified battery manufacturers. Offer 2-year battery warranty. Set up a battery replacement service portal."),
            'refund issue':    ("💰","Streamlined Returns","One-click return portal with 30-day no-questions-asked policy. Target 48-hour refund processing. Offer instant store credit as an option."),
            'packaging issue': ("🎁","Packaging Standards","Tamper-evident seals on all electronics. Moisture-resistant outer wraps. Random packaging integrity testing before dispatch."),
            'service issue':   ("🎧","Customer Support","Empathy-first training for all support agents. Live chat with <2 min response target. CSAT survey after every resolved ticket."),
        }

        if top_cat in rec_map:
            icon,title,detail = rec_map[top_cat]
            st.markdown(f'<div class="rec-card"><strong>{icon} Priority Action — {title}</strong><br><br>{detail}</div>', unsafe_allow_html=True)

        general = [
            ("🌟","Loyalty Programme","Launch points-based rewards for repeat electronics buyers. Reward 5-star reviewers with exclusive discount coupons to boost positive UGC."),
            ("📧","Post-Purchase Follow-Up","Send personalised emails 3 days after delivery. Include direct review link to maximise review volume."),
            ("📊","Weekly BI Dashboards","Automated weekly sentiment reports for management. Track negative review spikes and react within 48 hours."),
            ("🤖","AI Support Bot","Deploy self-service AI bot for top 10 FAQ queries — reduces support ticket load by up to 40%."),
            ("📱","Social Listening","Monitor Twitter/Instagram mentions alongside reviews. Early complaint detection prevents PR crises."),
            ("🏭","Supplier Scorecards","Rate suppliers monthly on quality complaint rate. Drop suppliers scoring below threshold for 3 consecutive months."),
        ]
        for icon,title,detail in general:
            st.markdown(f'<div class="rec-card"><strong>{icon} {title}</strong><br><br>{detail}</div>', unsafe_allow_html=True)

        st.markdown("---")
        st.markdown('<div class="section-header">🔬 Complaint Analysis Insight</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="rec-card">{complaint_data.get("insight","No data.")}</div>', unsafe_allow_html=True)
    else:
        st.warning("Dataset not loaded.")
