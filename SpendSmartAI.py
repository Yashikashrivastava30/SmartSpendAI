"""
SpendSmartAI — Advanced Edition
Professional AI-powered personal finance advisor with ML anomaly detection,
multi-agent analysis, and a polished dark-themed dashboard UI.
"""

import os
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Any, Tuple
import asyncio
from datetime import datetime, timedelta
from dotenv import load_dotenv
import json
import logging
from pydantic import BaseModel, Field
import csv
from io import StringIO
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder, StandardScaler

from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.genai import types

# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

APP_NAME = "spendsmart_ai"
USER_ID  = "default_user"

load_dotenv()
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")

# ─────────────────────────────────────────────
# Page config & Custom CSS (must be first Streamlit call)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="SpendSmartAI",
    page_icon="💸",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": None,
        "Report a bug": None,
        "About": "SpendSmartAI — AI-powered personal finance advisor"
    }
)

CUSTOM_CSS = """
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Root Variables ── */
:root {
    --bg-base:        #0b0f1a;
    --bg-card:        #111827;
    --bg-card2:       #1a2236;
    --border:         rgba(99,120,200,0.18);
    --border-glow:    rgba(99,120,200,0.45);
    --accent:         #6c8aff;
    --accent2:        #a78bfa;
    --accent3:        #34d399;
    --accent-warm:    #f59e0b;
    --accent-danger:  #ef4444;
    --text-primary:   #eef2ff;
    --text-secondary: #9aa5c4;
    --text-muted:     #5a6484;
    --font-serif:     'DM Serif Display', Georgia, serif;
    --font-sans:      'DM Sans', system-ui, sans-serif;
    --font-mono:      'JetBrains Mono', monospace;
    --radius:         14px;
    --radius-sm:      8px;
    --shadow:         0 4px 32px rgba(0,0,0,0.45);
    --shadow-glow:    0 0 40px rgba(108,138,255,0.12);
    --transition:     0.22s cubic-bezier(0.4,0,0.2,1);
}

/* ── Global ── */
html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg-base) !important;
    color: var(--text-primary) !important;
    font-family: var(--font-sans) !important;
}

/* Hide default Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }
.block-container { padding: 1.5rem 2.5rem 3rem !important; max-width: 1400px; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0e1525 0%, #0b0f1a 100%) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] > div:first-child { padding: 1.5rem 1.2rem; }
.sidebar-logo {
    display: flex; align-items: center; gap: 10px;
    margin-bottom: 1.6rem; padding-bottom: 1.2rem;
    border-bottom: 1px solid var(--border);
}
.sidebar-logo .logo-icon {
    width: 42px; height: 42px;
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    border-radius: 10px; display: flex; align-items: center;
    justify-content: center; font-size: 1.3rem;
    box-shadow: 0 0 16px rgba(108,138,255,0.35);
}
.sidebar-logo .logo-text { line-height: 1; }
.sidebar-logo .logo-title {
    font-family: var(--font-serif); font-size: 1.15rem;
    color: var(--text-primary); display: block;
}
.sidebar-logo .logo-sub {
    font-size: 0.68rem; color: var(--accent); letter-spacing: 0.1em;
    text-transform: uppercase; font-weight: 600;
}

/* ── Cards ── */
.ss-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.4rem 1.6rem;
    margin-bottom: 1.2rem;
    box-shadow: var(--shadow);
    transition: border-color var(--transition), box-shadow var(--transition);
    position: relative; overflow: hidden;
}
.ss-card::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, var(--accent), var(--accent2), transparent);
    opacity: 0.6;
}
.ss-card:hover {
    border-color: var(--border-glow);
    box-shadow: var(--shadow), var(--shadow-glow);
}
.ss-card-title {
    font-family: var(--font-serif); font-size: 1.15rem;
    color: var(--text-primary); margin: 0 0 0.8rem;
    display: flex; align-items: center; gap: 8px;
}

/* ── Hero Header ── */
.ss-hero {
    padding: 2.2rem 0 1.2rem;
    margin-bottom: 1.5rem;
    border-bottom: 1px solid var(--border);
    position: relative;
}
.ss-hero-eyebrow {
    font-family: var(--font-mono); font-size: 0.7rem;
    color: var(--accent); letter-spacing: 0.18em;
    text-transform: uppercase; margin-bottom: 0.5rem;
}
.ss-hero h1 {
    font-family: var(--font-serif); font-size: 2.8rem; font-weight: 400;
    color: var(--text-primary); margin: 0 0 0.5rem;
    line-height: 1.1;
}
.ss-hero h1 em { font-style: italic; color: var(--accent); }
.ss-hero-sub {
    font-size: 0.95rem; color: var(--text-secondary);
    max-width: 600px; line-height: 1.6;
}

/* ── KPI Metric Cards ── */
.kpi-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; margin-bottom: 1.5rem; }
.kpi-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.2rem 1.4rem;
    position: relative; overflow: hidden;
    transition: var(--transition);
}
.kpi-card:hover { border-color: var(--border-glow); transform: translateY(-2px); }
.kpi-card .kpi-label {
    font-size: 0.72rem; color: var(--text-muted);
    text-transform: uppercase; letter-spacing: 0.1em;
    font-weight: 600; margin-bottom: 0.4rem;
}
.kpi-card .kpi-value {
    font-family: var(--font-serif); font-size: 1.9rem;
    color: var(--text-primary); line-height: 1;
}
.kpi-card .kpi-delta {
    font-size: 0.78rem; margin-top: 0.35rem; font-weight: 500;
}
.kpi-card .kpi-icon {
    position: absolute; right: 1rem; top: 1rem;
    font-size: 1.5rem; opacity: 0.18;
}
.kpi-positive { color: var(--accent3); }
.kpi-negative { color: var(--accent-danger); }
.kpi-neutral  { color: var(--accent); }

.kpi-card.kpi-green { border-left: 3px solid var(--accent3); }
.kpi-card.kpi-red   { border-left: 3px solid var(--accent-danger); }
.kpi-card.kpi-blue  { border-left: 3px solid var(--accent); }
.kpi-card.kpi-amber { border-left: 3px solid var(--accent-warm); }

/* ── Section headings ── */
.ss-section-label {
    font-family: var(--font-mono); font-size: 0.68rem;
    color: var(--accent); letter-spacing: 0.16em;
    text-transform: uppercase; margin-bottom: 0.3rem; font-weight: 500;
}
.ss-section-title {
    font-family: var(--font-serif); font-size: 1.55rem;
    color: var(--text-primary); margin: 0 0 1rem; font-weight: 400;
}

/* ── Insight Pills (recs / warnings) ── */
.insight-pill {
    border-radius: var(--radius-sm);
    padding: 0.85rem 1.1rem;
    margin-bottom: 0.7rem;
    border-left: 3px solid;
    background: var(--bg-card2);
    font-size: 0.88rem; line-height: 1.55;
}
.insight-pill.green { border-color: var(--accent3); }
.insight-pill.amber { border-color: var(--accent-warm); }
.insight-pill.red   { border-color: var(--accent-danger); }
.insight-pill.blue  { border-color: var(--accent); }
.insight-pill strong { color: var(--text-primary); display: block; margin-bottom: 0.2rem; font-size: 0.92rem; }
.insight-pill span  { color: var(--text-secondary); }
.insight-pill .savings-badge {
    display: inline-block; margin-top: 0.4rem;
    background: rgba(52,211,153,0.1); color: var(--accent3);
    border: 1px solid rgba(52,211,153,0.25); border-radius: 99px;
    padding: 0.15rem 0.6rem; font-size: 0.72rem; font-weight: 600;
    font-family: var(--font-mono);
}

/* ── Severity Badges ── */
.severity-badge {
    display: inline-flex; align-items: center; gap: 4px;
    padding: 0.2rem 0.65rem; border-radius: 99px;
    font-size: 0.7rem; font-weight: 700; letter-spacing: 0.06em;
    text-transform: uppercase;
}
.sev-high   { background: rgba(239,68,68,0.15);  color: #f87171; border: 1px solid rgba(239,68,68,0.3); }
.sev-medium { background: rgba(245,158,11,0.15); color: #fbbf24; border: 1px solid rgba(245,158,11,0.3); }
.sev-low    { background: rgba(52,211,153,0.12); color: #6ee7b7; border: 1px solid rgba(52,211,153,0.25);}

/* ── Progress bar ── */
.ss-progress-wrap { margin: 0.5rem 0 1rem; }
.ss-progress-label { display: flex; justify-content: space-between; font-size: 0.78rem; color: var(--text-secondary); margin-bottom: 0.3rem; }
.ss-progress-bar { height: 6px; background: var(--bg-card2); border-radius: 99px; overflow: hidden; }
.ss-progress-fill { height: 100%; border-radius: 99px; background: linear-gradient(90deg, var(--accent), var(--accent2)); transition: width 0.8s cubic-bezier(0.4,0,0.2,1); }
.ss-progress-fill.green { background: linear-gradient(90deg, #059669, var(--accent3)); }

/* ── Divider ── */
.ss-divider { border: none; border-top: 1px solid var(--border); margin: 1.5rem 0; }

/* ── Streamlit Overrides ── */
/* Tabs */
[data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid var(--border) !important;
    gap: 0 !important;
}
[data-baseweb="tab"] {
    background: transparent !important;
    border: none !important;
    color: var(--text-muted) !important;
    font-family: var(--font-sans) !important;
    font-size: 0.85rem !important; font-weight: 500 !important;
    padding: 0.6rem 1.2rem !important;
    border-radius: var(--radius-sm) var(--radius-sm) 0 0 !important;
    transition: var(--transition) !important;
}
[aria-selected="true"][data-baseweb="tab"] {
    color: var(--accent) !important;
    border-bottom: 2px solid var(--accent) !important;
    background: rgba(108,138,255,0.06) !important;
}
[data-baseweb="tab"]:hover { color: var(--text-primary) !important; }

/* Number inputs & text inputs */
[data-baseweb="input"] {
    background: var(--bg-card2) !important;
    border-color: var(--border) !important;
    border-radius: var(--radius-sm) !important;
    font-family: var(--font-mono) !important;
    color: var(--text-primary) !important;
}
[data-baseweb="input"]:focus-within {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px rgba(108,138,255,0.2) !important;
}

/* Radio buttons */
[data-baseweb="radio"] { gap: 1.2rem !important; }
[data-testid="stRadio"] label { color: var(--text-secondary) !important; font-size: 0.88rem !important; }

/* Buttons */
[data-testid="baseButton-primary"], .stButton > button {
    background: linear-gradient(135deg, var(--accent), var(--accent2)) !important;
    border: none !important; color: #fff !important;
    font-family: var(--font-sans) !important; font-weight: 600 !important;
    border-radius: var(--radius-sm) !important;
    padding: 0.6rem 1.8rem !important;
    letter-spacing: 0.04em !important;
    box-shadow: 0 4px 14px rgba(108,138,255,0.35) !important;
    transition: var(--transition) !important;
}
[data-testid="baseButton-primary"]:hover, .stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(108,138,255,0.5) !important;
}

/* Download button */
[data-testid="baseButton-secondary"] {
    background: var(--bg-card2) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-secondary) !important;
    font-family: var(--font-sans) !important;
    border-radius: var(--radius-sm) !important;
}

/* Metrics */
[data-testid="stMetric"] { background: var(--bg-card); border: 1px solid var(--border); border-radius: var(--radius-sm); padding: 0.8rem 1rem; }
[data-testid="stMetricLabel"] { color: var(--text-muted) !important; font-size: 0.72rem !important; text-transform: uppercase; letter-spacing: 0.08em; }
[data-testid="stMetricValue"] { color: var(--text-primary) !important; font-family: var(--font-serif) !important; font-size: 1.6rem !important; }
[data-testid="stMetricDelta"] svg { display: none; }

/* Dataframe / table */
[data-testid="stDataFrame"] {
    background: var(--bg-card2) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
}

/* Expander */
[data-testid="stExpander"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    margin-bottom: 0.5rem !important;
}
[data-testid="stExpander"]:hover { border-color: var(--border-glow) !important; }
[data-testid="stExpanderToggleIcon"] { color: var(--accent) !important; }

/* Alerts */
[data-testid="stAlert"] {
    background: var(--bg-card2) !important;
    border-radius: var(--radius-sm) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-secondary) !important;
    font-size: 0.86rem !important;
}
.stSuccess { border-color: rgba(52,211,153,0.3) !important; }
.stWarning { border-color: rgba(245,158,11,0.3) !important; }
.stError   { border-color: rgba(239,68,68,0.3) !important; }
.stInfo    { border-color: rgba(108,138,255,0.3) !important; }

/* Spinner */
[data-testid="stSpinner"] { color: var(--accent) !important; }

/* File uploader */
[data-testid="stFileUploader"] {
    background: var(--bg-card2) !important;
    border: 1.5px dashed var(--border) !important;
    border-radius: var(--radius) !important;
    transition: var(--transition) !important;
}
[data-testid="stFileUploader"]:hover { border-color: var(--accent) !important; }

/* Selectbox */
[data-baseweb="select"] > div {
    background: var(--bg-card2) !important;
    border-color: var(--border) !important;
    color: var(--text-primary) !important;
    border-radius: var(--radius-sm) !important;
}

/* Subheader override */
h2, h3 {
    font-family: var(--font-serif) !important;
    color: var(--text-primary) !important;
    font-weight: 400 !important;
}
h2 { font-size: 1.5rem !important; }
h3 { font-size: 1.2rem !important; }

/* Plotly chart container */
[data-testid="stPlotlyChart"] { border-radius: var(--radius) !important; overflow: hidden !important; }

/* Scrollbar */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg-base); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 99px; }
::-webkit-scrollbar-thumb:hover { background: var(--border-glow); }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Plotly shared theme
# ─────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Sans, sans-serif", color="#9aa5c4", size=12),
    title_font=dict(family="DM Serif Display, serif", color="#eef2ff", size=16),
    legend=dict(
        bgcolor="rgba(17,24,39,0.8)", bordercolor="rgba(99,120,200,0.18)",
        borderwidth=1, font=dict(size=11)
    ),
    margin=dict(t=45, b=35, l=45, r=20),
    xaxis=dict(gridcolor="rgba(99,120,200,0.1)", zerolinecolor="rgba(99,120,200,0.2)"),
    yaxis=dict(gridcolor="rgba(99,120,200,0.1)", zerolinecolor="rgba(99,120,200,0.2)"),
)
PALETTE = ["#6c8aff","#a78bfa","#34d399","#f59e0b","#ef4444","#fb923c","#38bdf8","#f472b6"]


# ─────────────────────────────────────────────
# Pydantic Models
# ─────────────────────────────────────────────

class SpendingCategory(BaseModel):
    category: str = Field(..., description="Expense category name")
    amount: float = Field(..., description="Amount spent in this category")
    percentage: Optional[float] = Field(None, description="Percentage of total spending")

class SpendingRecommendation(BaseModel):
    category: str = Field(..., description="Category for recommendation")
    recommendation: str = Field(..., description="Recommendation details")
    potential_savings: Optional[float] = Field(None, description="Estimated monthly savings")

class BudgetAnalysis(BaseModel):
    total_expenses: float = Field(..., description="Total monthly expenses")
    monthly_income: Optional[float] = Field(None, description="Monthly income")
    spending_categories: List[SpendingCategory] = Field(..., description="Breakdown of spending by category")
    recommendations: List[SpendingRecommendation] = Field(..., description="Spending recommendations")

class EmergencyFund(BaseModel):
    recommended_amount: float = Field(..., description="Recommended emergency fund size")
    current_amount: Optional[float] = Field(None, description="Current emergency fund (if any)")
    current_status: str = Field(..., description="Status assessment of emergency fund")

class SavingsRecommendation(BaseModel):
    category: str = Field(..., description="Savings category")
    amount: float = Field(..., description="Recommended monthly amount")
    rationale: Optional[str] = Field(None, description="Explanation for this recommendation")

class AutomationTechnique(BaseModel):
    name: str = Field(..., description="Name of automation technique")
    description: str = Field(..., description="Details of how to implement")

class SavingsStrategy(BaseModel):
    emergency_fund: EmergencyFund = Field(..., description="Emergency fund recommendation")
    recommendations: List[SavingsRecommendation] = Field(..., description="Savings allocation recommendations")
    automation_techniques: Optional[List[AutomationTechnique]] = Field(None, description="Automation techniques to help save")

class Debt(BaseModel):
    name: str = Field(..., description="Name of debt")
    amount: float = Field(..., description="Current balance")
    interest_rate: float = Field(..., description="Annual interest rate (%)")
    min_payment: Optional[float] = Field(None, description="Minimum monthly payment")

class PayoffPlan(BaseModel):
    total_interest: float = Field(..., description="Total interest paid")
    months_to_payoff: int = Field(..., description="Months until debt-free")
    monthly_payment: Optional[float] = Field(None, description="Recommended monthly payment")

class PayoffPlans(BaseModel):
    avalanche: PayoffPlan = Field(..., description="Highest interest first method")
    snowball: PayoffPlan = Field(..., description="Smallest balance first method")

class DebtRecommendation(BaseModel):
    title: str = Field(..., description="Title of recommendation")
    description: str = Field(..., description="Details of recommendation")
    impact: Optional[str] = Field(None, description="Expected impact of this action")

class DebtReduction(BaseModel):
    total_debt: float = Field(..., description="Total debt amount")
    debts: List[Debt] = Field(..., description="List of all debts")
    payoff_plans: PayoffPlans = Field(..., description="Debt payoff strategies")
    recommendations: Optional[List[DebtRecommendation]] = Field(None, description="Recommendations for debt reduction")

class AnomalyInsight(BaseModel):
    transaction_date: str = Field(..., description="Date of anomalous transaction")
    category: str = Field(..., description="Spending category")
    amount: float = Field(..., description="Transaction amount")
    reason: str = Field(..., description="Why this is flagged as anomalous")
    severity: str = Field(..., description="low / medium / high")
    recommendation: str = Field(..., description="What the user should do about it")

class AnomalyAnalysis(BaseModel):
    summary: str = Field(..., description="Overall summary of anomaly findings")
    insights: List[AnomalyInsight] = Field(..., description="Per-transaction insights")
    pattern_warnings: List[str] = Field(..., description="Broader pattern-level warnings")
    total_recoverable_amount: Optional[float] = Field(None, description="Estimated amount that could be recovered")


# ─────────────────────────────────────────────
# ML Anomaly Detector
# ─────────────────────────────────────────────

class SpendingAnomalyDetector:
    def __init__(self, contamination=0.05):
        self.model = IsolationForest(contamination=contamination, random_state=42, n_estimators=150)
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.is_trained = False
        self.feature_columns: List[str] = []

    def preprocess(self, df: pd.DataFrame) -> np.ndarray:
        df = df.copy()
        df['Date'] = pd.to_datetime(df['Date'])
        df['day_of_week']   = df['Date'].dt.dayofweek
        df['day_of_month']  = df['Date'].dt.day
        df['is_weekend']    = df['day_of_week'].isin([5, 6]).astype(int)
        df['month']         = df['Date'].dt.month
        df['category_encoded'] = self.label_encoder.fit_transform(df['Category'].astype(str))
        # rolling mean deviation
        df = df.sort_values('Date')
        df['rolling_mean'] = df.groupby('Category')['Amount'].transform(lambda x: x.rolling(3, min_periods=1).mean())
        df['amount_deviation'] = (df['Amount'] - df['rolling_mean']).abs()
        self.feature_columns = [
            'Amount', 'category_encoded', 'day_of_week',
            'day_of_month', 'is_weekend', 'month', 'amount_deviation'
        ]
        features = df[self.feature_columns].values
        return self.scaler.fit_transform(features)

    def train(self, df: pd.DataFrame):
        if len(df) < 10:
            raise ValueError("Need at least 10 transactions to detect anomalies reliably.")
        features = self.preprocess(df)
        self.model.fit(features)
        self.is_trained = True

    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.is_trained:
            raise ValueError("Model must be trained before detecting anomalies.")
        df = df.copy()
        features = self.preprocess(df)
        predictions = self.model.predict(features)
        scores = self.model.score_samples(features)
        df['is_anomaly']    = predictions == -1
        df['anomaly_score'] = scores
        df['anomaly_rank']  = df['anomaly_score'].rank(ascending=True)
        return df

    def get_anomaly_summary(self, df: pd.DataFrame) -> Dict:
        anomalies = df[df['is_anomaly']]
        normal    = df[~df['is_anomaly']]
        top = anomalies.nsmallest(10, 'anomaly_score')[['Date','Category','Amount','anomaly_score']].copy()
        top['Date'] = top['Date'].astype(str)
        return {
            "total_transactions":     len(df),
            "anomaly_count":          len(anomalies),
            "anomaly_percentage":     round(len(anomalies)/len(df)*100, 1),
            "total_anomaly_amount":   round(float(anomalies['Amount'].sum()), 2),
            "avg_normal_transaction": round(float(normal['Amount'].mean()), 2) if len(normal) > 0 else 0,
            "avg_anomaly_transaction":round(float(anomalies['Amount'].mean()), 2) if len(anomalies) > 0 else 0,
            "top_anomalies":          top.to_dict('records'),
            "anomalies_by_category":  anomalies.groupby('Category')['Amount']
                                        .agg(['count','sum','mean']).round(2).to_dict('index'),
            "normal_patterns":        normal.groupby('Category')['Amount']
                                        .agg(['mean','std','count']).round(2).to_dict('index'),
        }


# ─────────────────────────────────────────────
# Finance Advisor System
# ─────────────────────────────────────────────

class FinanceAdvisorSystem:
    def __init__(self):
        self.session_service = InMemorySessionService()

        self.budget_analysis_agent = LlmAgent(
            name="BudgetAnalysisAgent",
            model="gemini-2.5-flash",
            description="Analyzes financial data to categorize spending patterns and recommend budget improvements",
            instruction="""You are a Budget Analysis Agent specialized in reviewing financial transactions and expenses.
You are the first agent in a sequence of three financial advisor agents.

Your tasks:
1. Analyze income, transactions, and expenses in detail
2. Categorize spending into logical groups with clear breakdown
3. Identify spending patterns and trends across categories
4. Suggest specific areas where spending could be reduced with concrete suggestions
5. Provide actionable recommendations with specific, quantified potential savings amounts

Consider:
- Number of dependants when evaluating household expenses
- Typical spending ratios for the income level (housing 30%, food 15%, etc.)
- Essential vs discretionary spending with clear separation
- Seasonal spending patterns if data spans multiple months

For spending categories, include ALL expenses, ensure percentages add up to 100%,
and make sure every expense is categorized.

For recommendations:
- Provide at least 4-6 specific, actionable recommendations with estimated savings
- Explain the reasoning behind each recommendation
- Consider the impact on quality of life and long-term financial health
- Suggest specific implementation steps

IMPORTANT: Store your analysis in state['budget_analysis'] for use by subsequent agents.""",
            output_schema=BudgetAnalysis,
            output_key="budget_analysis"
        )

        self.savings_strategy_agent = LlmAgent(
            name="SavingsStrategyAgent",
            model="gemini-2.5-flash",
            description="Recommends optimal savings strategies based on income, expenses, and financial goals",
            instruction="""You are a Savings Strategy Agent specialized in creating personalized savings plans.
You are the second agent in the sequence. READ the budget analysis from state['budget_analysis'] first.

Your tasks:
1. Review the budget analysis results from state['budget_analysis']
2. Recommend comprehensive savings strategies based on the analysis
3. Calculate optimal emergency fund size based on expenses and dependants
4. Suggest appropriate savings allocation across different purposes
5. Recommend practical automation techniques for saving consistently

Consider:
- Risk factors based on job stability and dependants
- Balancing immediate needs with long-term financial health
- Progressive savings rates as discretionary income increases
- Multiple savings goals (emergency, retirement, specific purchases)
- Areas of potential savings identified in the budget analysis

IMPORTANT: Store your strategy in state['savings_strategy'] for use by the Debt Reduction Agent.""",
            output_schema=SavingsStrategy,
            output_key="savings_strategy"
        )

        self.debt_reduction_agent = LlmAgent(
            name="DebtReductionAgent",
            model="gemini-2.5-flash",
            description="Creates optimized debt payoff plans to minimize interest paid and time to debt freedom",
            instruction="""You are a Debt Reduction Agent specialized in creating debt payoff strategies.
You are the final agent in the sequence. READ both state['budget_analysis'] and state['savings_strategy'] first.

Your tasks:
1. Review both budget analysis and savings strategy from the state
2. Analyze debts by interest rate, balance, and minimum payments
3. Create prioritized debt payoff plans (avalanche and snowball methods)
4. Calculate total interest paid and time to debt freedom
5. Suggest debt consolidation or refinancing opportunities
6. Provide specific recommendations to accelerate debt payoff

Consider:
- Cash flow constraints from the budget analysis
- Emergency fund and savings goals from the savings strategy
- Psychological factors (quick wins vs mathematical optimization)
- Credit score impact and improvement opportunities

IMPORTANT: Store your final plan in state['debt_reduction'] and ensure it aligns with the previous analyses.""",
            output_schema=DebtReduction,
            output_key="debt_reduction"
        )

        self.anomaly_analysis_agent = LlmAgent(
            name="AnomalyAnalysisAgent",
            model="gemini-2.5-flash",
            description="Explains detected spending anomalies in plain language",
            instruction="""You are a Spending Anomaly Analysis Agent.

You receive a statistical summary of anomalous transactions detected by a machine learning Isolation Forest model
trained on the user's own spending history.

Your job:
1. Explain each flagged transaction in plain English — WHY is it unusual?
2. Consider the normal spending pattern for that category as context
3. Distinguish between genuinely suspicious charges vs. one-time large purchases that are probably fine
   (e.g. annual subscriptions, holiday spending)
4. Assign severity: low (slightly unusual), medium (notably unusual), high (very suspicious or potentially fraudulent)
5. Give a specific recommendation for each: ignore, review, dispute, etc.
6. Identify any broader patterns (e.g. a whole category is running hot)

Be conversational and specific. Reference the actual amounts and dates.
Do NOT be alarmist — most anomalies are innocent, just unusual.""",
            output_schema=AnomalyAnalysis,
            output_key="anomaly_analysis"
        )

        self.coordinator_agent = SequentialAgent(
            name="FinanceCoordinatorAgent",
            description="Coordinates specialized finance agents to provide comprehensive financial advice",
            sub_agents=[
                self.budget_analysis_agent,
                self.savings_strategy_agent,
                self.debt_reduction_agent
            ]
        )

        self.runner = Runner(
            agent=self.coordinator_agent,
            app_name=APP_NAME,
            session_service=self.session_service
        )

    async def _run_anomaly_agent(self, anomaly_summary: Dict) -> Any:
        anomaly_session_id = f"anomaly_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        try:
            self.session_service.create_session(
                app_name=APP_NAME, user_id=USER_ID,
                session_id=anomaly_session_id, state={}
            )
            anomaly_runner = Runner(
                agent=self.anomaly_analysis_agent,
                app_name=APP_NAME,
                session_service=self.session_service
            )
            content = types.Content(
                role='user',
                parts=[types.Part(text=json.dumps(anomaly_summary, default=str))]
            )
            async for event in anomaly_runner.run_async(
                user_id=USER_ID, session_id=anomaly_session_id, new_message=content
            ):
                if event.is_final_response():
                    break
            session = self.session_service.get_session(
                app_name=APP_NAME, user_id=USER_ID, session_id=anomaly_session_id
            )
            return session.state.get("anomaly_analysis")
        except Exception as e:
            logger.warning(f"Anomaly agent failed: {e}")
            return None
        finally:
            try:
                self.session_service.delete_session(
                    app_name=APP_NAME, user_id=USER_ID, session_id=anomaly_session_id
                )
            except Exception:
                pass

    async def analyze_finances(self, financial_data: Dict[str, Any]) -> Dict[str, Any]:
        session_id = f"finance_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        anomaly_results   = None
        anomaly_df_records = None

        if financial_data.get("transactions"):
            try:
                df = pd.DataFrame(financial_data["transactions"])
                if len(df) >= 10:
                    detector = SpendingAnomalyDetector(contamination=0.05)
                    detector.train(df)
                    df_scored = detector.detect(df)
                    summary   = detector.get_anomaly_summary(df_scored)
                    copy_ = df_scored.copy()
                    copy_['Date'] = copy_['Date'].astype(str)
                    copy_['is_anomaly'] = copy_['is_anomaly'].astype(bool)
                    anomaly_df_records = copy_.to_dict('records')
                    anomaly_results = await self._run_anomaly_agent(summary)
            except Exception as e:
                logger.warning(f"Anomaly detection failed: {e}")

        try:
            initial_state = {
                "monthly_income": financial_data.get("monthly_income", 0),
                "dependants":     financial_data.get("dependants", 0),
                "transactions":   financial_data.get("transactions", []),
                "manual_expenses":financial_data.get("manual_expenses", {}),
                "debts":          financial_data.get("debts", [])
            }
            session = self.session_service.create_session(
                app_name=APP_NAME, user_id=USER_ID,
                session_id=session_id, state=initial_state
            )
            if session.state.get("transactions"):
                self._preprocess_transactions(session)
            if session.state.get("manual_expenses"):
                self._preprocess_manual_expenses(session)

            default_results = self._create_default_results(financial_data)

            content = types.Content(
                role='user',
                parts=[types.Part(text=json.dumps(financial_data, default=str))]
            )
            async for event in self.runner.run_async(
                user_id=USER_ID, session_id=session_id, new_message=content
            ):
                if event.is_final_response() and event.author == self.coordinator_agent.name:
                    break

            updated = self.session_service.get_session(
                app_name=APP_NAME, user_id=USER_ID, session_id=session_id
            )
            results = {}
            for key in ["budget_analysis", "savings_strategy", "debt_reduction"]:
                val = updated.state.get(key)
                results[key] = (
                    _parse_json_safely(val, default_results[key]) if val else default_results[key]
                )
            results["anomaly_analysis"] = anomaly_results
            results["anomaly_df"]       = anomaly_df_records
            return results
        except Exception as e:
            logger.exception(f"Error during analysis: {e}")
            raise
        finally:
            try:
                self.session_service.delete_session(
                    app_name=APP_NAME, user_id=USER_ID, session_id=session_id
                )
            except Exception:
                pass

    def _preprocess_transactions(self, session):
        txns = session.state.get("transactions", [])
        if not txns:
            return
        df = pd.DataFrame(txns)
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')
        if 'Category' in df.columns and 'Amount' in df.columns:
            session.state["category_spending"] = df.groupby('Category')['Amount'].sum().to_dict()
            session.state["total_spending"]    = float(df['Amount'].sum())

    def _preprocess_manual_expenses(self, session):
        manual = session.state.get("manual_expenses", {})
        if not manual:
            return
        session.state.update({
            "total_manual_spending":    sum(manual.values()),
            "manual_category_spending": manual
        })

    def _create_default_results(self, fd: Dict[str, Any]) -> Dict[str, Any]:
        income   = fd.get("monthly_income", 0)
        expenses = fd.get("manual_expenses", {}) or {}
        if not expenses and fd.get("transactions"):
            for t in fd["transactions"]:
                cat = t.get("Category","Uncategorized")
                expenses[cat] = expenses.get(cat, 0) + t.get("Amount", 0)
        total = sum(expenses.values())
        debts = fd.get("debts", [])
        total_debt = sum(d.get("amount", 0) for d in debts)
        return {
            "budget_analysis": {
                "total_expenses": total, "monthly_income": income,
                "spending_categories": [
                    {"category": c, "amount": a,
                     "percentage": (a/total*100) if total > 0 else 0}
                    for c, a in expenses.items()
                ],
                "recommendations": [{"category":"General","recommendation":"Review your expenses carefully","potential_savings":total*0.1}]
            },
            "savings_strategy": {
                "emergency_fund": {"recommended_amount": total*6, "current_amount": 0, "current_status": "Not started"},
                "recommendations": [
                    {"category":"Emergency Fund","amount":total*0.1,"rationale":"Build emergency fund first"},
                    {"category":"Retirement","amount":income*0.15,"rationale":"Long-term savings"}
                ],
                "automation_techniques": [{"name":"Automatic Transfer","description":"Set up automatic transfers on payday"}]
            },
            "debt_reduction": {
                "total_debt": total_debt, "debts": debts,
                "payoff_plans": {
                    "avalanche": {"total_interest":total_debt*0.2,"months_to_payoff":24,"monthly_payment":total_debt/24 if total_debt>0 else 0},
                    "snowball":  {"total_interest":total_debt*0.25,"months_to_payoff":24,"monthly_payment":total_debt/24 if total_debt>0 else 0}
                },
                "recommendations": [{"title":"Increase Payments","description":"Increase your monthly payments","impact":"Reduces total interest paid"}]
            }
        }


# ─────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────

def _parse_json_safely(data: Any, default: Any = None) -> Any:
    try:
        return json.loads(data) if isinstance(data, str) else data
    except json.JSONDecodeError:
        return default

def fmt_inr(amount: float) -> str:
    return f"₹{amount:,.0f}"

def _plotly_theme(fig, height=380):
    fig.update_layout(**PLOTLY_LAYOUT, height=height)
    return fig

def validate_csv(file) -> Tuple[bool, str]:
    try:
        content = file.read().decode('utf-8')
        has_header = csv.Sniffer().has_header(content)
        file.seek(0)
        if not has_header:
            return False, "CSV file must have column headers."
        df = pd.read_csv(StringIO(content))
        missing = [c for c in ['Date','Category','Amount'] if c not in df.columns]
        if missing:
            return False, f"Missing columns: {', '.join(missing)}"
        try:
            pd.to_datetime(df['Date'])
        except Exception:
            return False, "Invalid date format — use YYYY-MM-DD."
        try:
            df['Amount'].replace(r'[\$,]','',regex=True).astype(float)
        except Exception:
            return False, "Invalid amount values in 'Amount' column."
        return True, "Valid"
    except Exception as e:
        return False, f"Invalid CSV: {e}"

def parse_csv_transactions(content: bytes) -> Dict:
    df = pd.read_csv(StringIO(content.decode('utf-8')))
    df['Amount'] = df['Amount'].replace(r'[\$,]','',regex=True).astype(float)
    df['Date']   = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')
    return {"transactions": df.to_dict('records'), "columns": list(df.columns)}


# ─────────────────────────────────────────────
# Display helpers (HTML components)
# ─────────────────────────────────────────────

def kpi_html(label: str, value: str, delta: str = "", variant: str = "blue", icon: str = "") -> str:
    delta_html = f'<div class="kpi-delta kpi-{variant}">{delta}</div>' if delta else ""
    return f"""
    <div class="kpi-card kpi-{variant}">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        {delta_html}
        <div class="kpi-icon">{icon}</div>
    </div>"""

def pill(text: str, strong: str = "", savings: float = None, variant: str = "blue") -> str:
    savings_html = f'<span class="savings-badge">Save {fmt_inr(savings)}/mo</span>' if savings else ""
    return f"""
    <div class="insight-pill {variant}">
        <strong>{strong}</strong>
        <span>{text}</span>
        {savings_html}
    </div>"""

def sev_badge(severity: str) -> str:
    icon = {"high":"●","medium":"◆","low":"○"}.get(severity.lower(),"·")
    return f'<span class="severity-badge sev-{severity.lower()}">{icon} {severity.upper()}</span>'

def progress_bar(label_left: str, label_right: str, pct: float, variant: str = "") -> str:
    w = min(max(pct * 100, 0), 100)
    return f"""
    <div class="ss-progress-wrap">
        <div class="ss-progress-label"><span>{label_left}</span><span>{label_right}</span></div>
        <div class="ss-progress-bar"><div class="ss-progress-fill {variant}" style="width:{w:.1f}%"></div></div>
    </div>"""


# ─────────────────────────────────────────────
# Display Functions
# ─────────────────────────────────────────────

def display_budget_analysis(analysis: Dict[str, Any]):
    if isinstance(analysis, str):
        try:
            analysis = json.loads(analysis)
        except json.JSONDecodeError:
            st.error("Failed to parse budget analysis.")
            return
    if not isinstance(analysis, dict):
        st.error("Invalid budget analysis format.")
        return

    income   = analysis.get("monthly_income", 0) or 0
    expenses = analysis.get("total_expenses", 0)  or 0
    surplus  = income - expenses
    sr_pct   = (expenses / income * 100) if income > 0 else 0
    sav_pct  = (surplus / income * 100)  if income > 0 else 0

    # ── KPI row ────────────────────────────────
    kpis = [
        kpi_html("Monthly Income",   fmt_inr(income),   icon="💼", variant="blue"),
        kpi_html("Total Expenses",   fmt_inr(expenses), icon="📊", variant="red"),
        kpi_html(
            "Net Surplus" if surplus >= 0 else "Net Deficit",
            fmt_inr(abs(surplus)),
            delta="↑ Positive cashflow" if surplus >= 0 else "↓ Overspending",
            variant="green" if surplus >= 0 else "red",
            icon="✅" if surplus >= 0 else "⚠️"
        ),
        kpi_html("Expense Ratio",    f"{sr_pct:.1f}%",
                 delta="Recommended ≤ 70%",
                 variant="green" if sr_pct <= 70 else "amber",
                 icon="📉"),
    ]
    st.markdown(f'<div class="kpi-grid">{"".join(kpis)}</div>', unsafe_allow_html=True)

    # cashflow progress bar
    if income > 0:
        st.markdown(
            progress_bar(f"Expenses {fmt_inr(expenses)}", f"Income {fmt_inr(income)}",
                         expenses/income, variant="" if sr_pct<=70 else ""),
            unsafe_allow_html=True
        )

    col_l, col_r = st.columns([1.1, 0.9])

    # Pie chart
    with col_l:
        st.markdown('<p class="ss-section-label">Spending Breakdown</p>', unsafe_allow_html=True)
        cats = analysis.get("spending_categories", [])
        if cats:
            fig = px.pie(
                values=[c["amount"] for c in cats],
                names=[c["category"] for c in cats],
                color_discrete_sequence=PALETTE,
                hole=0.52,
            )
            fig.update_traces(
                textposition='outside', textfont_size=11,
                marker=dict(line=dict(color='#0b0f1a', width=2))
            )
            fig.update_layout(
                **PLOTLY_LAYOUT, height=360,
                showlegend=True,
                legend=dict(orientation="v", x=1.05, y=0.5,
                            font=dict(size=10), bgcolor="rgba(0,0,0,0)")
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # Bar chart income vs expenses
    with col_r:
        st.markdown('<p class="ss-section-label">Income vs Expenses</p>', unsafe_allow_html=True)
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            x=["Monthly Income", "Total Expenses", "Surplus / Deficit"],
            y=[income, expenses, abs(surplus)],
            marker_color=["#6c8aff", "#ef4444", "#34d399" if surplus >= 0 else "#f59e0b"],
            marker_line_width=0,
            text=[fmt_inr(income), fmt_inr(expenses), fmt_inr(abs(surplus))],
            textposition='outside',
            textfont=dict(color="#eef2ff", size=11)
        ))
        fig2.update_layout(**PLOTLY_LAYOUT, height=360,
                           showlegend=False,
                           yaxis=dict(visible=False))
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

    # Category table
    if cats:
        st.markdown('<p class="ss-section-label" style="margin-top:1rem">Category Breakdown</p>', unsafe_allow_html=True)
        cat_df = pd.DataFrame(cats)
        cat_df = cat_df[cat_df['amount'] > 0].sort_values('amount', ascending=False)
        cat_df['amount']     = cat_df['amount'].apply(fmt_inr)
        cat_df['percentage'] = cat_df['percentage'].apply(lambda x: f"{x:.1f}%" if x else "—")
        cat_df.columns       = ['Category','Amount','Share']
        st.dataframe(cat_df, use_container_width=True, hide_index=True)

    # Recommendations
    recs = analysis.get("recommendations", [])
    if recs:
        st.markdown('<hr class="ss-divider">', unsafe_allow_html=True)
        st.markdown('<p class="ss-section-label">AI Recommendations</p>', unsafe_allow_html=True)
        st.markdown('<p class="ss-section-title">Where to Cut Back</p>', unsafe_allow_html=True)
        for rec in recs:
            savings = rec.get("potential_savings")
            st.markdown(
                pill(rec.get("recommendation",""), strong=rec.get("category",""),
                     savings=savings, variant="blue"),
                unsafe_allow_html=True
            )


def display_savings_strategy(strategy: Dict[str, Any]):
    if isinstance(strategy, str):
        try:
            strategy = json.loads(strategy)
        except json.JSONDecodeError:
            st.error("Failed to parse savings strategy.")
            return
    if not isinstance(strategy, dict):
        st.error("Invalid savings strategy format.")
        return

    ef = strategy.get("emergency_fund", {})
    recs = strategy.get("recommendations", [])
    techniques = strategy.get("automation_techniques", [])

    rec_total = sum(r.get("amount",0) for r in recs)

    # ── KPI row ────────────────────────────────
    current = ef.get("current_amount", 0) or 0
    target  = ef.get("recommended_amount", 1) or 1
    ef_pct  = current / target if target > 0 else 0

    kpis = [
        kpi_html("Emergency Fund Target", fmt_inr(target), icon="🏦", variant="blue"),
        kpi_html("Current Saved",         fmt_inr(current),
                 delta=f"{ef_pct*100:.0f}% funded",
                 variant="green" if ef_pct >= 1 else "amber", icon="💰"),
        kpi_html("Recommended Monthly Savings", fmt_inr(rec_total), icon="📈", variant="green"),
        kpi_html("Fund Status", ef.get("current_status","—")[:22], icon="🎯", variant="blue"),
    ]
    st.markdown(f'<div class="kpi-grid">{"".join(kpis)}</div>', unsafe_allow_html=True)

    # Emergency fund progress
    st.markdown('<p class="ss-section-label">Emergency Fund Progress</p>', unsafe_allow_html=True)
    st.markdown(
        progress_bar(f"Current: {fmt_inr(current)}", f"Target: {fmt_inr(target)}",
                     ef_pct, variant="green"),
        unsafe_allow_html=True
    )

    col_l, col_r = st.columns([1, 1])

    with col_l:
        # Savings allocation chart
        if recs:
            st.markdown('<p class="ss-section-label" style="margin-top:0.6rem">Savings Allocation</p>', unsafe_allow_html=True)
            fig = px.bar(
                x=[r['amount'] for r in recs],
                y=[r['category'] for r in recs],
                orientation='h',
                color=[r['amount'] for r in recs],
                color_continuous_scale=["#1e3a5f","#6c8aff","#a78bfa"],
                text=[fmt_inr(r['amount']) for r in recs],
            )
            fig.update_traces(textposition='outside', textfont=dict(color="#eef2ff", size=11), marker_line_width=0)
            fig.update_layout(**PLOTLY_LAYOUT, height=320, showlegend=False,
                              coloraxis_showscale=False,
                              xaxis=dict(visible=False))
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with col_r:
        if recs:
            st.markdown('<p class="ss-section-label" style="margin-top:0.6rem">Allocation Details</p>', unsafe_allow_html=True)
            for rec in recs:
                st.markdown(
                    pill(rec.get("rationale",""), strong=rec.get("category",""),
                         savings=rec.get("amount"), variant="green"),
                    unsafe_allow_html=True
                )

    # Automation techniques
    if techniques:
        st.markdown('<hr class="ss-divider">', unsafe_allow_html=True)
        st.markdown('<p class="ss-section-label">Automation Techniques</p>', unsafe_allow_html=True)
        for t in techniques:
            st.markdown(pill(t.get("description",""), strong=f"⚡ {t.get('name','')}", variant="amber"),
                        unsafe_allow_html=True)


def display_debt_reduction(plan: Dict[str, Any]):
    if isinstance(plan, str):
        try:
            plan = json.loads(plan)
        except json.JSONDecodeError:
            st.error("Failed to parse debt reduction plan.")
            return
    if not isinstance(plan, dict):
        st.error("Invalid debt reduction format.")
        return

    total_debt = plan.get("total_debt", 0)
    debts      = plan.get("debts", [])
    payoff     = plan.get("payoff_plans", {})
    av = payoff.get("avalanche", {})
    sn = payoff.get("snowball", {})

    # ── KPI row ────────────────────────────────
    interest_saved = sn.get("total_interest",0) - av.get("total_interest",0)
    kpis = [
        kpi_html("Total Debt",           fmt_inr(total_debt), icon="💳", variant="red"),
        kpi_html("Avalanche: Pay Off In", f"{av.get('months_to_payoff',0)} mo", icon="🏔️", variant="blue"),
        kpi_html("Snowball: Pay Off In",  f"{sn.get('months_to_payoff',0)} mo", icon="⛄", variant="amber"),
        kpi_html("Interest Saved (Aval.)", fmt_inr(interest_saved),
                 delta="vs Snowball", variant="green", icon="💡"),
    ]
    st.markdown(f'<div class="kpi-grid">{"".join(kpis)}</div>', unsafe_allow_html=True)

    col_l, col_r = st.columns([1.1, 0.9])

    with col_l:
        if debts:
            st.markdown('<p class="ss-section-label">Debt Breakdown</p>', unsafe_allow_html=True)
            debt_df = pd.DataFrame(debts)
            fig = px.bar(
                debt_df, x="name", y="amount",
                color="interest_rate",
                color_continuous_scale=["#1e3a5f","#6c8aff","#ef4444"],
                labels={"name":"Debt","amount":"Balance (₹)","interest_rate":"Rate (%)"},
                text=[fmt_inr(a) for a in debt_df['amount']],
            )
            fig.update_traces(textposition='outside', textfont=dict(color="#eef2ff", size=10), marker_line_width=0)
            fig.update_layout(**PLOTLY_LAYOUT, height=340, showlegend=False,
                              coloraxis_showscale=True,
                              coloraxis_colorbar=dict(title="Rate %", tickfont=dict(size=10)))
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with col_r:
        st.markdown('<p class="ss-section-label">Method Comparison</p>', unsafe_allow_html=True)
        methods = ["Avalanche", "Snowball"]
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            name="Total Interest",
            x=methods,
            y=[av.get("total_interest",0), sn.get("total_interest",0)],
            marker_color=["#6c8aff","#a78bfa"],
            text=[fmt_inr(av.get("total_interest",0)), fmt_inr(sn.get("total_interest",0))],
            textposition='outside', textfont=dict(color="#eef2ff", size=11),
        ))
        fig2.update_layout(**PLOTLY_LAYOUT, height=220, showlegend=False,
                           yaxis=dict(visible=False), bargap=0.4)
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

        # comparison table
        comp = pd.DataFrame({
            "Metric": ["Total Interest", "Months to Payoff", "Monthly Payment"],
            "Avalanche": [
                fmt_inr(av.get("total_interest",0)),
                f"{av.get('months_to_payoff',0)} mo",
                fmt_inr(av.get("monthly_payment",0))
            ],
            "Snowball": [
                fmt_inr(sn.get("total_interest",0)),
                f"{sn.get('months_to_payoff',0)} mo",
                fmt_inr(sn.get("monthly_payment",0))
            ]
        })
        st.dataframe(comp, use_container_width=True, hide_index=True)

    # Debts table
    if debts:
        st.markdown('<p class="ss-section-label" style="margin-top:0.8rem">All Debts</p>', unsafe_allow_html=True)
        df_show = pd.DataFrame(debts).copy()
        if 'amount' in df_show.columns:
            df_show['amount'] = df_show['amount'].apply(fmt_inr)
        if 'interest_rate' in df_show.columns:
            df_show['interest_rate'] = df_show['interest_rate'].apply(lambda x: f"{x:.1f}%")
        if 'min_payment' in df_show.columns:
            df_show['min_payment'] = df_show['min_payment'].apply(lambda x: fmt_inr(x) if x else "—")
        df_show.columns = [c.replace('_',' ').title() for c in df_show.columns]
        st.dataframe(df_show, use_container_width=True, hide_index=True)

    # Recommendations
    recs = plan.get("recommendations", [])
    if recs:
        st.markdown('<hr class="ss-divider">', unsafe_allow_html=True)
        st.markdown('<p class="ss-section-label">Debt Reduction Tactics</p>', unsafe_allow_html=True)
        for rec in recs:
            impact = rec.get("impact","")
            text   = rec.get("description","")
            if impact:
                text += f" — <em style='color:var(--accent3)'>{impact}</em>"
            st.markdown(pill(text, strong=rec.get("title",""), variant="red"),
                        unsafe_allow_html=True)


def display_anomaly_detection(anomaly_df_records, anomaly_analysis):
    if not anomaly_df_records:
        st.info("Upload a CSV with at least 10 transactions to enable ML anomaly detection.")
        return

    df = pd.DataFrame(anomaly_df_records)
    df['Date'] = pd.to_datetime(df['Date'])
    df['is_anomaly'] = df['is_anomaly'].astype(bool)

    total   = len(df)
    flagged = int(df['is_anomaly'].sum())
    flagged_amt = float(df[df['is_anomaly']]['Amount'].sum()) if flagged > 0 else 0.0
    anom_pct = flagged / total * 100 if total > 0 else 0

    # ── KPIs ────────────────────────────────────
    kpis = [
        kpi_html("Total Transactions", str(total), icon="📋", variant="blue"),
        kpi_html("Anomalies Detected", str(flagged),
                 delta="Requires review" if flagged > 0 else "All clear",
                 variant="red" if flagged > 0 else "green", icon="🚨"),
        kpi_html("Anomaly Rate", f"{anom_pct:.1f}%",
                 variant="red" if anom_pct > 10 else "amber", icon="📊"),
        kpi_html("Flagged Amount",  fmt_inr(flagged_amt), icon="💸", variant="amber"),
    ]
    st.markdown(f'<div class="kpi-grid">{"".join(kpis)}</div>', unsafe_allow_html=True)

    # ── Charts ──────────────────────────────────
    col_l, col_r = st.columns([1.3, 0.7])

    with col_l:
        st.markdown('<p class="ss-section-label">Transaction Map</p>', unsafe_allow_html=True)
        fig = px.scatter(
            df, x='Date', y='Amount',
            color='is_anomaly',
            color_discrete_map={True: '#ef4444', False: '#6c8aff'},
            hover_data=['Category', 'anomaly_score'],
            labels={'is_anomaly': 'Anomaly'},
            symbol='is_anomaly',
            symbol_map={True: 'x', False: 'circle'},
        )
        fig.update_traces(marker=dict(size=9, opacity=0.85,
                                      line=dict(width=1, color='rgba(0,0,0,0.4)')))
        fig.update_layout(**PLOTLY_LAYOUT, height=340)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with col_r:
        st.markdown('<p class="ss-section-label">Anomaly Score Distribution</p>', unsafe_allow_html=True)
        fig2 = px.histogram(
            df, x='anomaly_score', color='is_anomaly',
            color_discrete_map={True:'#ef4444', False:'#6c8aff'},
            nbins=30,
            labels={'anomaly_score':'Score','is_anomaly':'Anomaly'}
        )
        fig2.update_layout(**PLOTLY_LAYOUT, height=340,
                           legend=dict(orientation="h", y=-0.15, x=0.3))
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

    # Category anomaly heatmap
    if flagged > 0:
        st.markdown('<p class="ss-section-label" style="margin-top:0.5rem">Anomalies by Category</p>', unsafe_allow_html=True)
        cat_anom = (
            df[df['is_anomaly']]
            .groupby('Category')['Amount']
            .agg(['count','sum'])
            .reset_index()
            .rename(columns={'count':'Anomalies','sum':'Total Amount'})
            .sort_values('Anomalies', ascending=False)
        )
        cat_anom['Total Amount'] = cat_anom['Total Amount'].apply(fmt_inr)
        st.dataframe(cat_anom, use_container_width=True, hide_index=True)

    # Flagged transactions table
    if flagged > 0:
        st.markdown('<p class="ss-section-label" style="margin-top:0.5rem">Flagged Transactions</p>', unsafe_allow_html=True)
        flagged_df = df[df['is_anomaly']][['Date','Category','Amount','anomaly_score']].sort_values('anomaly_score').copy()
        flagged_df['Date'] = flagged_df['Date'].dt.strftime('%Y-%m-%d')
        flagged_df['Amount'] = flagged_df['Amount'].apply(fmt_inr)
        flagged_df['anomaly_score'] = flagged_df['anomaly_score'].apply(lambda x: f"{x:.4f}")
        flagged_df.columns = ['Date','Category','Amount','Anomaly Score ↓']
        st.dataframe(flagged_df, use_container_width=True, hide_index=True)

    st.markdown('<hr class="ss-divider">', unsafe_allow_html=True)

    # ── AI Analysis ──────────────────────────────
    st.markdown('<p class="ss-section-label">AI Analysis</p>', unsafe_allow_html=True)
    st.markdown('<p class="ss-section-title">Anomaly Insights</p>', unsafe_allow_html=True)

    if not anomaly_analysis:
        st.warning("AI explanation unavailable — check your API key or try again.")
        return

    if isinstance(anomaly_analysis, str):
        try:
            anomaly_analysis = json.loads(anomaly_analysis)
        except json.JSONDecodeError:
            st.error("Failed to parse AI anomaly analysis.")
            return

    summary = anomaly_analysis.get("summary","")
    if summary:
        st.markdown(f'<div class="ss-card"><p style="color:var(--text-secondary);font-size:0.92rem;line-height:1.6;margin:0">{summary}</p></div>', unsafe_allow_html=True)

    insights = anomaly_analysis.get("insights", [])
    if insights:
        st.markdown('<p class="ss-section-label" style="margin-top:0.8rem">Per-Transaction Insights</p>', unsafe_allow_html=True)
        for ins in insights:
            sev   = ins.get("severity","low").lower()
            badge = sev_badge(sev)
            with st.expander(
                f"{'🔴' if sev=='high' else '🟡' if sev=='medium' else '🟢'} "
                f"{ins.get('category','?')} — {fmt_inr(ins.get('amount',0))} on {ins.get('transaction_date','?')}"
            ):
                st.markdown(f"{badge}", unsafe_allow_html=True)
                st.markdown(f"**Why flagged:** {ins.get('reason','—')}")
                st.markdown(f"**Recommendation:** {ins.get('recommendation','—')}")

    warnings = anomaly_analysis.get("pattern_warnings", [])
    if warnings:
        st.markdown('<p class="ss-section-label" style="margin-top:0.8rem">Pattern Warnings</p>', unsafe_allow_html=True)
        for w in warnings:
            st.markdown(pill(w, strong="⚠️ Pattern Warning", variant="amber"), unsafe_allow_html=True)

    recoverable = anomaly_analysis.get("total_recoverable_amount")
    if recoverable:
        st.markdown(
            pill(f"Estimated recoverable amount based on flagged transactions.",
                 strong=f"💡 Potential Recovery: {fmt_inr(recoverable)}",
                 variant="green"),
            unsafe_allow_html=True
        )


def display_csv_preview(df: pd.DataFrame):
    total_amt  = df['Amount'].sum()
    dates      = pd.to_datetime(df['Date'])
    date_range = f"{dates.min().strftime('%b %d')} – {dates.max().strftime('%b %d, %Y')}"

    kpis = [
        kpi_html("Transactions",   str(len(df)),        icon="📋", variant="blue"),
        kpi_html("Total Spend",    fmt_inr(total_amt),  icon="💸", variant="red"),
        kpi_html("Date Range",     date_range,          icon="📅", variant="amber"),
        kpi_html("Categories",     str(df['Category'].nunique()), icon="🏷️", variant="green"),
    ]
    st.markdown(f'<div class="kpi-grid">{"".join(kpis)}</div>', unsafe_allow_html=True)

    col_l, col_r = st.columns([1.1, 0.9])
    with col_l:
        cat_totals = df.groupby('Category')['Amount'].sum().sort_values(ascending=False).reset_index()
        fig = px.bar(cat_totals, x='Category', y='Amount',
                     color='Amount',
                     color_continuous_scale=["#1e3a5f","#6c8aff","#a78bfa"],
                     text=[fmt_inr(v) for v in cat_totals['Amount']])
        fig.update_traces(textposition='outside', textfont=dict(color="#eef2ff", size=10), marker_line_width=0)
        fig.update_layout(**PLOTLY_LAYOUT, height=290, showlegend=False,
                          coloraxis_showscale=False, yaxis=dict(visible=False))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with col_r:
        st.markdown("**Sample Transactions**")
        st.dataframe(df[['Date','Category','Amount']].head(8), use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────

def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-logo">
            <div class="logo-icon">💸</div>
            <div class="logo-text">
                <span class="logo-title">SpendSmartAI</span>
                <span class="logo-sub">Finance Advisor</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<p class="ss-section-label">CSV Format</p>', unsafe_allow_html=True)
        st.markdown("""
        <div style="font-size:0.82rem;color:var(--text-secondary);line-height:1.8">
        Required columns:<br>
        <code style="color:var(--accent);background:rgba(108,138,255,0.1);padding:1px 6px;border-radius:4px">Date</code> — YYYY-MM-DD<br>
        <code style="color:var(--accent);background:rgba(108,138,255,0.1);padding:1px 6px;border-radius:4px">Category</code> — text<br>
        <code style="color:var(--accent);background:rgba(108,138,255,0.1);padding:1px 6px;border-radius:4px">Amount</code> — number
        </div>
        """, unsafe_allow_html=True)

        sample_csv = """Date,Category,Amount
2024-01-01,Housing,20000.00
2024-01-02,Food,1500.50
2024-01-03,Transportation,800.00
2024-01-05,Food,1200.20
2024-01-07,Entertainment,2000.00
2024-01-10,Food,1400.30
2024-01-12,Transportation,650.00
2024-01-15,Housing,500.00
2024-01-18,Food,25000.00
2024-01-20,Healthcare,3500.00
2024-01-22,Entertainment,800.00
2024-01-25,Food,1100.00
2024-01-28,Transportation,450.00
2024-01-30,Personal,2200.00"""

        st.download_button(
            label="📥 Download Sample CSV",
            data=sample_csv,
            file_name="spendsmart_template.csv",
            mime="text/csv",
            use_container_width=True
        )

        st.markdown('<hr class="ss-divider">', unsafe_allow_html=True)
        st.markdown('<p class="ss-section-label">How It Works</p>', unsafe_allow_html=True)
        steps = [
            ("🔍", "Budget Agent", "Categorises spending & finds savings"),
            ("💰", "Savings Agent", "Builds emergency fund & savings plan"),
            ("💳", "Debt Agent",   "Avalanche & snowball payoff plans"),
            ("🚨", "Anomaly ML",   "Isolation Forest on your data"),
        ]
        for icon, title, desc in steps:
            st.markdown(f"""
            <div style="display:flex;gap:10px;margin-bottom:0.85rem;align-items:flex-start">
                <div style="font-size:1.2rem;margin-top:1px">{icon}</div>
                <div>
                    <div style="font-size:0.82rem;font-weight:600;color:var(--text-primary)">{title}</div>
                    <div style="font-size:0.75rem;color:var(--text-muted);line-height:1.4">{desc}</div>
                </div>
            </div>""", unsafe_allow_html=True)

        st.markdown('<hr class="ss-divider">', unsafe_allow_html=True)
        st.markdown("""
        <div style="font-size:0.7rem;color:var(--text-muted);text-align:center;line-height:1.5">
            🔒 All data processed locally.<br>Nothing stored or transmitted.
        </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Main App
# ─────────────────────────────────────────────

def main():
    render_sidebar()

    if not GEMINI_API_KEY:
        st.error("🔑 **GOOGLE_API_KEY** not found. Add it to your `.env` file and restart.")
        st.stop()

    # ── Hero ───────────────────────────────────
    st.markdown("""
    <div class="ss-hero">
        <div class="ss-hero-eyebrow">AI-Powered Personal Finance</div>
        <h1>Spend <em>Smart</em>,<br>Save More.</h1>
        <p class="ss-hero-sub">
            Multi-agent AI analysis • ML anomaly detection • Debt payoff modelling •
            Personalised savings strategies — all in one place.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Tabs ───────────────────────────────────
    input_tab, about_tab = st.tabs(["📊 Financial Analysis", "ℹ️ About"])

    with input_tab:

        # ── Income & Household ─────────────────
        st.markdown('<p class="ss-section-label">Step 1</p>', unsafe_allow_html=True)
        st.markdown('<p class="ss-section-title">Income & Household</p>', unsafe_allow_html=True)

        c1, c2, c3 = st.columns([2, 1, 1])
        with c1:
            monthly_income = st.number_input(
                "Monthly Income (₹)", min_value=0.0, step=1000.0, value=50000.0,
                help="Your total monthly income after taxes"
            )
        with c2:
            dependants = st.number_input(
                "Dependants", min_value=0, max_value=20, step=1, value=0,
                help="Number of people financially dependent on you"
            )
        with c3:
            current_savings = st.number_input(
                "Current Savings (₹)", min_value=0.0, step=1000.0, value=0.0,
                help="Existing emergency fund or savings"
            )

        st.markdown('<hr class="ss-divider">', unsafe_allow_html=True)

        # ── Expenses ──────────────────────────
        st.markdown('<p class="ss-section-label">Step 2</p>', unsafe_allow_html=True)
        st.markdown('<p class="ss-section-title">Expenses</p>', unsafe_allow_html=True)

        expense_option = st.radio(
            "Input method",
            ("📤 Upload CSV Transactions", "✍️ Enter Manually"),
            horizontal=True
        )

        transactions_df   = None
        manual_expenses   = {}
        use_manual_expenses = False

        if expense_option == "📤 Upload CSV Transactions":
            col_up, _ = st.columns([2, 1])
            with col_up:
                f = st.file_uploader(
                    "Drop your CSV here",
                    type=["csv"],
                    help="Columns required: Date, Category, Amount"
                )
            if f:
                ok, msg = validate_csv(f)
                if ok:
                    try:
                        f.seek(0)
                        parsed = parse_csv_transactions(f.read())
                        transactions_df = pd.DataFrame(parsed['transactions'])
                        st.success(f"✅ {len(transactions_df)} transactions loaded successfully.")
                        with st.expander("📊 Preview Uploaded Data", expanded=True):
                            display_csv_preview(transactions_df)
                    except Exception as e:
                        st.error(f"❌ Error processing CSV: {e}")
                else:
                    st.error(f"❌ {msg}")

        else:
            use_manual_expenses = True
            st.markdown("Enter your **monthly** expense amounts by category:")
            categories = [
                ("🏠", "Housing"), ("🔌", "Utilities"), ("🍽️", "Food"),
                ("🚗", "Transportation"), ("🏥", "Healthcare"), ("🎭", "Entertainment"),
                ("📚", "Education"), ("👤", "Personal"), ("💰", "Savings"), ("📦", "Other")
            ]
            cols = st.columns(5)
            for i, (emoji, cat) in enumerate(categories):
                with cols[i % 5]:
                    manual_expenses[cat] = st.number_input(
                        f"{emoji} {cat}", min_value=0.0, step=500.0, value=0.0,
                        key=f"manual_{cat}"
                    )

            active = {k: v for k, v in manual_expenses.items() if v > 0}
            if active:
                total_manual = sum(active.values())
                st.markdown('<p class="ss-section-label" style="margin-top:1rem">Summary</p>', unsafe_allow_html=True)
                kpis = [
                    kpi_html("Total Entered", fmt_inr(total_manual), icon="💳", variant="blue"),
                    kpi_html("Expense Ratio",
                             f"{total_manual/monthly_income*100:.1f}%" if monthly_income > 0 else "—",
                             variant="green" if monthly_income > 0 and total_manual/monthly_income < 0.7 else "red",
                             icon="📊"),
                    kpi_html("Categories", str(len(active)), icon="🏷️", variant="amber"),
                    kpi_html("Surplus",
                             fmt_inr(monthly_income - total_manual),
                             variant="green" if monthly_income > total_manual else "red",
                             icon="✨"),
                ]
                st.markdown(f'<div class="kpi-grid">{"".join(kpis)}</div>', unsafe_allow_html=True)

        st.markdown('<hr class="ss-divider">', unsafe_allow_html=True)

        # ── Debt ──────────────────────────────
        st.markdown('<p class="ss-section-label">Step 3 (Optional)</p>', unsafe_allow_html=True)
        st.markdown('<p class="ss-section-title">Debts</p>', unsafe_allow_html=True)
        st.caption("Add your debts to get avalanche & snowball payoff strategies.")

        num_debts = st.number_input("Number of debts", min_value=0, max_value=10, step=1, value=0)
        debts = []
        if num_debts > 0:
            cols = st.columns(min(int(num_debts), 3))
            for i in range(int(num_debts)):
                with cols[i % 3]:
                    with st.container():
                        st.markdown(f'<div class="ss-card"><p class="ss-card-title">💳 Debt #{i+1}</p>', unsafe_allow_html=True)
                        name   = st.text_input("Name", value=f"Debt {i+1}", key=f"dn_{i}")
                        amount = st.number_input("Balance (₹)", min_value=0.01, step=1000.0, value=10000.0, key=f"da_{i}")
                        rate   = st.number_input("Interest Rate (%)", min_value=0.0, max_value=100.0, step=0.1, value=12.0, key=f"dr_{i}")
                        minpay = st.number_input("Min Payment (₹)", min_value=0.0, step=500.0, value=500.0, key=f"dm_{i}")
                        debts.append({"name": name, "amount": amount, "interest_rate": rate, "min_payment": minpay})
                        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<hr class="ss-divider">', unsafe_allow_html=True)

        # ── Analyse Button ────────────────────
        col_c = st.columns([1.5, 2, 1.5])[1]
        with col_c:
            analyse = st.button("🔮 Analyse My Finances", use_container_width=True)

        if analyse:
            if expense_option == "📤 Upload CSV Transactions" and transactions_df is None:
                st.error("Please upload a valid CSV file first.")
                st.stop()
            if use_manual_expenses and not any(manual_expenses.values()):
                st.warning("No expenses entered — analysis will be limited.")

            # store current savings in manual for emergency fund context
            if use_manual_expenses and current_savings > 0:
                manual_expenses["_current_savings"] = current_savings

            financial_data = {
                "monthly_income":    monthly_income,
                "dependants":        dependants,
                "current_savings":   current_savings,
                "transactions":      transactions_df.to_dict('records') if transactions_df is not None else None,
                "manual_expenses":   {k: v for k, v in manual_expenses.items() if not k.startswith("_") and v > 0} if use_manual_expenses else None,
                "debts":             debts
            }

            st.markdown('<hr class="ss-divider">', unsafe_allow_html=True)
            st.markdown("""
            <div class="ss-hero" style="padding:1rem 0 0.5rem">
                <div class="ss-hero-eyebrow">Analysis Results</div>
                <h1 style="font-size:2rem">Your Financial<br><em>Snapshot</em></h1>
            </div>
            """, unsafe_allow_html=True)

            with st.spinner("🤖 AI agents are analysing your data — this may take 30–60 seconds…"):
                system = FinanceAdvisorSystem()
                try:
                    results = asyncio.run(system.analyze_finances(financial_data))
                except Exception as e:
                    st.error(f"Analysis failed: {e}")
                    st.stop()

            rtabs = st.tabs([
                "💰 Budget Analysis",
                "📈 Savings Strategy",
                "💳 Debt Reduction",
                "🚨 Anomaly Detection"
            ])

            with rtabs[0]:
                if results.get("budget_analysis"):
                    display_budget_analysis(results["budget_analysis"])
                else:
                    st.info("No budget analysis available.")

            with rtabs[1]:
                if results.get("savings_strategy"):
                    display_savings_strategy(results["savings_strategy"])
                else:
                    st.info("No savings strategy available.")

            with rtabs[2]:
                if results.get("debt_reduction"):
                    display_debt_reduction(results["debt_reduction"])
                else:
                    st.info("No debt plan available. Add your debts above.")

            with rtabs[3]:
                display_anomaly_detection(
                    results.get("anomaly_df"),
                    results.get("anomaly_analysis")
                )

    # ── About Tab ─────────────────────────────
    with about_tab:
        st.markdown("""
        <div class="ss-hero" style="padding:1.5rem 0 0.5rem">
            <div class="ss-hero-eyebrow">Documentation</div>
            <h1 style="font-size:2rem">How <em>SpendSmartAI</em><br>Works</h1>
        </div>
        """, unsafe_allow_html=True)

        agents = [
            ("🔍", "Budget Analysis Agent",
             "Processes your transactions or manual inputs, categorises all spending, calculates income-to-expense ratios, and identifies at least 4–6 specific, quantified saving opportunities. Uses Gemini 2.5 Flash with a structured Pydantic output schema."),
            ("💰", "Savings Strategy Agent",
             "Reads the budget analysis from shared session state, then builds a personalised savings plan. Calculates your ideal emergency fund (3–6 months of expenses, adjusted for dependants), allocates savings across goals, and suggests automation techniques."),
            ("💳", "Debt Reduction Agent",
             "Accesses both prior analyses from state, then models your debt payoff using the Avalanche method (highest interest first — mathematically optimal) and the Snowball method (smallest balance first — psychologically motivating), comparing total interest and time to debt freedom."),
            ("🚨", "ML Anomaly Detector",
             "Trains a scikit-learn Isolation Forest model exclusively on your personal transaction history — not generic rules. Detects statistically unusual spending by amount, category, day-of-week, rolling deviation, and more. Requires ≥ 10 transactions. The Anomaly Analysis Agent then explains each flagged transaction in plain English with severity ratings."),
        ]
        for icon, title, desc in agents:
            st.markdown(f"""
            <div class="ss-card">
                <p class="ss-card-title">{icon} {title}</p>
                <p style="color:var(--text-secondary);font-size:0.88rem;line-height:1.65;margin:0">{desc}</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
        <div class="ss-card">
            <p class="ss-card-title">🔒 Privacy & Data Security</p>
            <p style="color:var(--text-secondary);font-size:0.88rem;line-height:1.65;margin:0">
                All financial data is processed entirely within your session — nothing is written to disk,
                stored in a database, or transmitted to any third-party service beyond the Gemini API
                (which receives only the anonymised numerical summary needed for analysis).
                The ML model is trained fresh each session and discarded on page refresh.
            </p>
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()