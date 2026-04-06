import os
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Optional, Any
import asyncio
from datetime import datetime
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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

APP_NAME = "finance_advisor"
USER_ID = "default_user"

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

# Anomaly Detection Pydantic Models
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
        self.model = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100
        )
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.is_trained = False
        self.feature_columns = []

    def preprocess(self, df: pd.DataFrame) -> np.ndarray:
        df = df.copy()
        df['Date'] = pd.to_datetime(df['Date'])
        df['day_of_week'] = df['Date'].dt.dayofweek
        df['day_of_month'] = df['Date'].dt.day
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['month'] = df['Date'].dt.month
        df['category_encoded'] = self.label_encoder.fit_transform(df['Category'].astype(str))

        self.feature_columns = [
            'Amount', 'category_encoded', 'day_of_week',
            'day_of_month', 'is_weekend', 'month'
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
        df['is_anomaly'] = predictions == -1
        df['anomaly_score'] = scores
        df['anomaly_rank'] = df['anomaly_score'].rank(ascending=True)
        return df

    def get_anomaly_summary(self, df_with_anomalies: pd.DataFrame) -> Dict:
        anomalies = df_with_anomalies[df_with_anomalies['is_anomaly'] == True]
        normal = df_with_anomalies[df_with_anomalies['is_anomaly'] == False]

        top_anomalies = anomalies.nsmallest(10, 'anomaly_score')[
            ['Date', 'Category', 'Amount', 'anomaly_score']
        ].copy()
        top_anomalies['Date'] = top_anomalies['Date'].astype(str)

        return {
            "total_transactions": len(df_with_anomalies),
            "anomaly_count": len(anomalies),
            "anomaly_percentage": round(len(anomalies) / len(df_with_anomalies) * 100, 1),
            "total_anomaly_amount": round(float(anomalies['Amount'].sum()), 2),
            "avg_normal_transaction": round(float(normal['Amount'].mean()), 2) if len(normal) > 0 else 0,
            "avg_anomaly_transaction": round(float(anomalies['Amount'].mean()), 2) if len(anomalies) > 0 else 0,
            "top_anomalies": top_anomalies.to_dict('records'),
            "anomalies_by_category": anomalies.groupby('Category')['Amount']
                .agg(['count', 'sum', 'mean']).round(2).to_dict('index'),
            "normal_patterns": normal.groupby('Category')['Amount']
                .agg(['mean', 'std', 'count']).round(2).to_dict('index')
        }


# ─────────────────────────────────────────────
# Load env
# ─────────────────────────────────────────────

load_dotenv()
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")


def parse_json_safely(data: str, default_value: Any = None) -> Any:
    try:
        return json.loads(data) if isinstance(data, str) else data
    except json.JSONDecodeError:
        return default_value


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

For spending categories, include ALL expenses from the user's data, ensure percentages add up to 100%,
and make sure every expense is categorized.

For recommendations:
- Provide at least 3-5 specific, actionable recommendations with estimated savings
- Explain the reasoning behind each recommendation
- Consider the impact on quality of life and long-term financial health
- Suggest specific implementation steps for each recommendation

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

        # Anomaly explanation agent (runs independently)
        self.anomaly_analysis_agent = LlmAgent(
            name="AnomalyAnalysisAgent",
            model="gemini-2.5-flash",
            description="Explains detected spending anomalies in plain language",
            instruction="""You are a Spending Anomaly Analysis Agent.

You receive a statistical summary of anomalous transactions detected
by a machine learning Isolation Forest model trained on the user's
own spending history.

Your job:
1. Explain each flagged transaction in plain English — WHY is it unusual?
2. Consider the normal spending pattern for that category as context
3. Distinguish between genuinely suspicious charges vs. one-time large
   purchases that are probably fine (e.g. annual subscriptions, holiday spending)
4. Assign severity: low (slightly unusual), medium (notably unusual),
   high (very suspicious or potentially fraudulent)
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
        """Run the anomaly explanation agent independently."""
        anomaly_session_id = f"anomaly_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        try:
            self.session_service.create_session(
                app_name=APP_NAME,
                user_id=USER_ID,
                session_id=anomaly_session_id,
                state={}
            )

            anomaly_runner = Runner(
                agent=self.anomaly_analysis_agent,
                app_name=APP_NAME,
                session_service=self.session_service
            )

            anomaly_content = types.Content(
                role='user',
                parts=[types.Part(text=json.dumps(anomaly_summary, default=str))]
            )

            async for event in anomaly_runner.run_async(
                user_id=USER_ID,
                session_id=anomaly_session_id,
                new_message=anomaly_content
            ):
                if event.is_final_response():
                    break

            anomaly_session = self.session_service.get_session(
                app_name=APP_NAME,
                user_id=USER_ID,
                session_id=anomaly_session_id
            )
            return anomaly_session.state.get("anomaly_analysis")
        except Exception as e:
            logger.warning(f"Anomaly agent failed: {str(e)}")
            return None
        finally:
            try:
                self.session_service.delete_session(
                    app_name=APP_NAME,
                    user_id=USER_ID,
                    session_id=anomaly_session_id
                )
            except Exception:
                pass

    async def analyze_finances(self, financial_data: Dict[str, Any]) -> Dict[str, Any]:
        session_id = f"finance_session_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"

        anomaly_results = None
        anomaly_df_records = None

        # ── ML Anomaly Detection ──────────────────────
        if financial_data.get("transactions"):
            try:
                df = pd.DataFrame(financial_data["transactions"])
                if len(df) >= 10:
                    detector = SpendingAnomalyDetector(contamination=0.05)
                    detector.train(df)
                    df_scored = detector.detect(df)
                    anomaly_summary = detector.get_anomaly_summary(df_scored)

                    # Serialize for storage
                    df_scored_copy = df_scored.copy()
                    df_scored_copy['Date'] = df_scored_copy['Date'].astype(str)
                    df_scored_copy['is_anomaly'] = df_scored_copy['is_anomaly'].astype(bool)
                    anomaly_df_records = df_scored_copy.to_dict('records')

                    # Run AI explanation agent
                    anomaly_results = await self._run_anomaly_agent(anomaly_summary)
                else:
                    logger.info("Not enough transactions for anomaly detection (need ≥ 10).")
            except Exception as e:
                logger.warning(f"Anomaly detection failed: {str(e)}")

        # ── Main Financial Analysis ───────────────────
        try:
            initial_state = {
                "monthly_income": financial_data.get("monthly_income", 0),
                "dependants": financial_data.get("dependants", 0),
                "transactions": financial_data.get("transactions", []),
                "manual_expenses": financial_data.get("manual_expenses", {}),
                "debts": financial_data.get("debts", [])
            }

            session = self.session_service.create_session(
                app_name=APP_NAME,
                user_id=USER_ID,
                session_id=session_id,
                state=initial_state
            )

            if session.state.get("transactions"):
                self._preprocess_transactions(session)

            if session.state.get("manual_expenses"):
                self._preprocess_manual_expenses(session)

            default_results = self._create_default_results(financial_data)

            user_content = types.Content(
                role='user',
                parts=[types.Part(text=json.dumps(financial_data, default=str))]
            )

            async for event in self.runner.run_async(
                user_id=USER_ID,
                session_id=session_id,
                new_message=user_content
            ):
                if event.is_final_response() and event.author == self.coordinator_agent.name:
                    break

            updated_session = self.session_service.get_session(
                app_name=APP_NAME,
                user_id=USER_ID,
                session_id=session_id
            )

            results = {}
            for key in ["budget_analysis", "savings_strategy", "debt_reduction"]:
                value = updated_session.state.get(key)
                results[key] = parse_json_safely(value, default_results[key]) if value else default_results[key]

            results["anomaly_analysis"] = anomaly_results
            results["anomaly_df"] = anomaly_df_records

            return results

        except Exception as e:
            logger.exception(f"Error during finance analysis: {str(e)}")
            raise
        finally:
            try:
                self.session_service.delete_session(
                    app_name=APP_NAME,
                    user_id=USER_ID,
                    session_id=session_id
                )
            except Exception:
                pass

    def _preprocess_transactions(self, session):
        transactions = session.state.get("transactions", [])
        if not transactions:
            return
        df = pd.DataFrame(transactions)
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')
        if 'Category' in df.columns and 'Amount' in df.columns:
            session.state["category_spending"] = df.groupby('Category')['Amount'].sum().to_dict()
            session.state["total_spending"] = float(df['Amount'].sum())

    def _preprocess_manual_expenses(self, session):
        manual_expenses = session.state.get("manual_expenses", {})
        if not manual_expenses:
            return
        session.state.update({
            "total_manual_spending": sum(manual_expenses.values()),
            "manual_category_spending": manual_expenses
        })

    def _create_default_results(self, financial_data: Dict[str, Any]) -> Dict[str, Any]:
        monthly_income = financial_data.get("monthly_income", 0)
        expenses = financial_data.get("manual_expenses", {}) or {}

        if not expenses and financial_data.get("transactions"):
            for t in financial_data["transactions"]:
                cat = t.get("Category", "Uncategorized")
                amt = t.get("Amount", 0)
                expenses[cat] = expenses.get(cat, 0) + amt

        total_expenses = sum(expenses.values())
        debts = financial_data.get("debts", [])
        total_debt = sum(d.get("amount", 0) for d in debts)

        return {
            "budget_analysis": {
                "total_expenses": total_expenses,
                "monthly_income": monthly_income,
                "spending_categories": [
                    {"category": c, "amount": a,
                     "percentage": (a / total_expenses * 100) if total_expenses > 0 else 0}
                    for c, a in expenses.items()
                ],
                "recommendations": [
                    {"category": "General",
                     "recommendation": "Consider reviewing your expenses carefully",
                     "potential_savings": total_expenses * 0.1}
                ]
            },
            "savings_strategy": {
                "emergency_fund": {
                    "recommended_amount": total_expenses * 6,
                    "current_amount": 0,
                    "current_status": "Not started"
                },
                "recommendations": [
                    {"category": "Emergency Fund", "amount": total_expenses * 0.1,
                     "rationale": "Build emergency fund first"},
                    {"category": "Retirement", "amount": monthly_income * 0.15,
                     "rationale": "Long-term savings"}
                ],
                "automation_techniques": [
                    {"name": "Automatic Transfer",
                     "description": "Set up automatic transfers on payday"}
                ]
            },
            "debt_reduction": {
                "total_debt": total_debt,
                "debts": debts,
                "payoff_plans": {
                    "avalanche": {
                        "total_interest": total_debt * 0.2,
                        "months_to_payoff": 24,
                        "monthly_payment": total_debt / 24 if total_debt > 0 else 0
                    },
                    "snowball": {
                        "total_interest": total_debt * 0.25,
                        "months_to_payoff": 24,
                        "monthly_payment": total_debt / 24 if total_debt > 0 else 0
                    }
                },
                "recommendations": [
                    {"title": "Increase Payments",
                     "description": "Increase your monthly payments",
                     "impact": "Reduces total interest paid"}
                ]
            }
        }


# ─────────────────────────────────────────────
# Display Functions
# ─────────────────────────────────────────────

def display_budget_analysis(analysis: Dict[str, Any]):
    if isinstance(analysis, str):
        try:
            analysis = json.loads(analysis)
        except json.JSONDecodeError:
            st.error("Failed to parse budget analysis results")
            return
    if not isinstance(analysis, dict):
        st.error("Invalid budget analysis format")
        return

    if "spending_categories" in analysis:
        st.subheader("Spending by Category")
        fig = px.pie(
            values=[cat["amount"] for cat in analysis["spending_categories"]],
            names=[cat["category"] for cat in analysis["spending_categories"]],
            title="Your Spending Breakdown"
        )
        st.plotly_chart(fig)

    if "total_expenses" in analysis:
        st.subheader("Income vs. Expenses")
        income = analysis.get("monthly_income", 0)
        expenses = analysis["total_expenses"]
        surplus_deficit = income - expenses

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=["Income", "Expenses"],
            y=[income, expenses],
            marker_color=["green", "red"]
        ))
        fig.update_layout(title="Monthly Income vs. Expenses")
        st.plotly_chart(fig)
        st.metric("Monthly Surplus/Deficit", f"₹{surplus_deficit:,.2f}", delta=f"₹{surplus_deficit:,.2f}")

    if "recommendations" in analysis:
        st.subheader("Spending Reduction Recommendations")
        for rec in analysis["recommendations"]:
            st.markdown(f"**{rec['category']}**: {rec['recommendation']}")
            if "potential_savings" in rec:
                st.metric("Potential Monthly Savings", f"₹{rec['potential_savings']:,.2f}")


def display_savings_strategy(strategy: Dict[str, Any]):
    if isinstance(strategy, str):
        try:
            strategy = json.loads(strategy)
        except json.JSONDecodeError:
            st.error("Failed to parse savings strategy results")
            return
    if not isinstance(strategy, dict):
        st.error("Invalid savings strategy format")
        return

    st.subheader("Savings Recommendations")

    if "emergency_fund" in strategy:
        ef = strategy["emergency_fund"]
        st.markdown("### Emergency Fund")
        st.markdown(f"**Recommended Size**: ₹{ef['recommended_amount']:,.2f}")
        st.markdown(f"**Current Status**: {ef['current_status']}")
        if "current_amount" in ef and "recommended_amount" in ef:
            progress = ef["current_amount"] / ef["recommended_amount"] if ef["recommended_amount"] > 0 else 0
            st.progress(min(progress, 1.0))
            st.markdown(f"₹{ef['current_amount']:,.2f} of ₹{ef['recommended_amount']:,.2f}")

    if "recommendations" in strategy:
        st.markdown("### Recommended Savings Allocations")
        for rec in strategy["recommendations"]:
            st.markdown(f"**{rec['category']}**: ₹{rec['amount']:,.2f}/month")
            st.markdown(f"_{rec['rationale']}_")

    if "automation_techniques" in strategy:
        st.markdown("### Automation Techniques")
        for technique in strategy["automation_techniques"]:
            st.markdown(f"**{technique['name']}**: {technique['description']}")


def display_debt_reduction(plan: Dict[str, Any]):
    if isinstance(plan, str):
        try:
            plan = json.loads(plan)
        except json.JSONDecodeError:
            st.error("Failed to parse debt reduction results")
            return
    if not isinstance(plan, dict):
        st.error("Invalid debt reduction format")
        return

    if "total_debt" in plan:
        st.metric("Total Debt", f"₹{plan['total_debt']:,.2f}")

    if "debts" in plan and plan["debts"]:
        st.subheader("Your Debts")
        debt_df = pd.DataFrame(plan["debts"])
        st.dataframe(debt_df)
        fig = px.bar(
            debt_df, x="name", y="amount", color="interest_rate",
            labels={"name": "Debt", "amount": "Amount (₹)", "interest_rate": "Interest Rate (%)"},
            title="Debt Breakdown"
        )
        st.plotly_chart(fig)

    if "payoff_plans" in plan:
        st.subheader("Debt Payoff Plans")
        tabs = st.tabs(["Avalanche Method", "Snowball Method", "Comparison"])

        with tabs[0]:
            st.markdown("### Avalanche Method (Highest Interest First)")
            if "avalanche" in plan["payoff_plans"]:
                av = plan["payoff_plans"]["avalanche"]
                st.markdown(f"**Total Interest Paid**: ₹{av['total_interest']:,.2f}")
                st.markdown(f"**Time to Debt Freedom**: {av['months_to_payoff']} months")
                if "monthly_payment" in av:
                    st.markdown(f"**Recommended Monthly Payment**: ₹{av['monthly_payment']:,.2f}")

        with tabs[1]:
            st.markdown("### Snowball Method (Smallest Balance First)")
            if "snowball" in plan["payoff_plans"]:
                sn = plan["payoff_plans"]["snowball"]
                st.markdown(f"**Total Interest Paid**: ₹{sn['total_interest']:,.2f}")
                st.markdown(f"**Time to Debt Freedom**: {sn['months_to_payoff']} months")
                if "monthly_payment" in sn:
                    st.markdown(f"**Recommended Monthly Payment**: ₹{sn['monthly_payment']:,.2f}")

        with tabs[2]:
            st.markdown("### Method Comparison")
            if "avalanche" in plan["payoff_plans"] and "snowball" in plan["payoff_plans"]:
                av = plan["payoff_plans"]["avalanche"]
                sn = plan["payoff_plans"]["snowball"]
                comparison_df = pd.DataFrame({
                    "Method": ["Avalanche", "Snowball"],
                    "Total Interest": [av["total_interest"], sn["total_interest"]],
                    "Months to Payoff": [av["months_to_payoff"], sn["months_to_payoff"]]
                })
                st.dataframe(comparison_df)
                fig = go.Figure(data=[
                    go.Bar(name="Total Interest", x=comparison_df["Method"], y=comparison_df["Total Interest"]),
                    go.Bar(name="Months to Payoff", x=comparison_df["Method"], y=comparison_df["Months to Payoff"])
                ])
                fig.update_layout(barmode='group', title="Debt Payoff Method Comparison")
                st.plotly_chart(fig)

    if "recommendations" in plan and plan["recommendations"]:
        st.subheader("Debt Reduction Recommendations")
        for rec in plan["recommendations"]:
            st.markdown(f"**{rec['title']}**: {rec['description']}")
            if "impact" in rec:
                st.markdown(f"_Impact: {rec['impact']}_")


def display_anomaly_detection(anomaly_df_records, anomaly_analysis):
    """Display the full anomaly detection tab."""

    if not anomaly_df_records:
        st.info("Upload a CSV with at least 10 transactions to enable anomaly detection.")
        return

    df_anomaly = pd.DataFrame(anomaly_df_records)
    df_anomaly['Date'] = pd.to_datetime(df_anomaly['Date'])
    df_anomaly['is_anomaly'] = df_anomaly['is_anomaly'].astype(bool)

    total = len(df_anomaly)
    flagged = int(df_anomaly['is_anomaly'].sum())
    flagged_amount = float(df_anomaly[df_anomaly['is_anomaly']]['Amount'].sum()) if flagged > 0 else 0.0

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Transactions", total)
    col2.metric("Anomalies Detected", flagged)
    col3.metric("Anomaly Rate", f"{flagged / total * 100:.1f}%")
    col4.metric("Flagged Amount", f"₹{flagged_amount:,.2f}")

    st.divider()

    # Scatter plot — all transactions, anomalies in red
    st.subheader("📊 Transaction Map — Anomalies Highlighted")
    fig = px.scatter(
        df_anomaly,
        x='Date',
        y='Amount',
        color='is_anomaly',
        color_discrete_map={True: '#ef4444', False: '#3b82f6'},
        hover_data=['Category'],
        title='All Transactions — Red = Flagged as Anomalous',
        labels={'is_anomaly': 'Anomaly'}
    )
    fig.update_traces(marker=dict(size=8, opacity=0.8))
    st.plotly_chart(fig, use_container_width=True)

    # Anomaly score distribution
    st.subheader("🔍 Anomaly Score Distribution")
    fig2 = px.histogram(
        df_anomaly,
        x='anomaly_score',
        color='is_anomaly',
        color_discrete_map={True: '#ef4444', False: '#3b82f6'},
        nbins=40,
        title='Distribution of Anomaly Scores (lower = more anomalous)',
        labels={'anomaly_score': 'Anomaly Score', 'is_anomaly': 'Anomaly'}
    )
    st.plotly_chart(fig2, use_container_width=True)

    # Flagged transactions table
    if flagged > 0:
        st.subheader("🚩 Flagged Transactions")
        flagged_df = df_anomaly[df_anomaly['is_anomaly'] == True][
            ['Date', 'Category', 'Amount', 'anomaly_score']
        ].sort_values('anomaly_score').copy()
        flagged_df['Date'] = flagged_df['Date'].dt.strftime('%Y-%m-%d')
        flagged_df['anomaly_score'] = flagged_df['anomaly_score'].round(4)
        st.dataframe(flagged_df, use_container_width=True)

    st.divider()

    # AI explanation
    st.subheader("🤖 AI Explanation of Anomalies")

    if not anomaly_analysis:
        st.warning("AI explanation unavailable. Check your API key or try again.")
        return

    if isinstance(anomaly_analysis, str):
        try:
            anomaly_analysis = json.loads(anomaly_analysis)
        except json.JSONDecodeError:
            st.error("Failed to parse AI anomaly analysis.")
            return

    # Summary
    st.info(anomaly_analysis.get("summary", "No summary available."))

    # Per-transaction insights
    insights = anomaly_analysis.get("insights", [])
    if insights:
        st.markdown("### Transaction Insights")
        for insight in insights:
            severity = insight.get("severity", "low").lower()
            severity_icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(severity, "⚪")
            with st.expander(
                f"{severity_icon} {insight.get('category', 'Unknown')} — "
                f"₹{insight.get('amount', 0):,.2f} on {insight.get('transaction_date', 'N/A')}"
            ):
                st.markdown(f"**Why flagged:** {insight.get('reason', 'N/A')}")
                st.markdown(f"**Severity:** {severity.capitalize()}")
                st.markdown(f"**Recommendation:** {insight.get('recommendation', 'N/A')}")

    # Pattern warnings
    warnings = anomaly_analysis.get("pattern_warnings", [])
    if warnings:
        st.markdown("### ⚠️ Pattern Warnings")
        for w in warnings:
            st.warning(w)

    # Recoverable amount
    recoverable = anomaly_analysis.get("total_recoverable_amount")
    if recoverable:
        st.success(f"💡 Estimated recoverable amount: **₹{recoverable:,.2f}**")


# ─────────────────────────────────────────────
# CSV Utilities
# ─────────────────────────────────────────────

def parse_csv_transactions(file_content) -> Dict:
    try:
        df = pd.read_csv(StringIO(file_content.decode('utf-8')))
        required_columns = ['Date', 'Category', 'Amount']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")
        df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')
        df['Amount'] = df['Amount'].replace('[\$,]', '', regex=True).astype(float)
        return {
            'transactions': df.to_dict('records'),
            'category_totals': df.groupby('Category')['Amount'].sum().reset_index().to_dict('records')
        }
    except Exception as e:
        raise ValueError(f"Error parsing CSV file: {str(e)}")


def validate_csv_format(file):
    try:
        content = file.read().decode('utf-8')
        has_header = csv.Sniffer().has_header(content)
        file.seek(0)
        if not has_header:
            return False, "CSV file must have headers"
        df = pd.read_csv(StringIO(content))
        required_columns = ['Date', 'Category', 'Amount']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return False, f"Missing required columns: {', '.join(missing_columns)}"
        try:
            pd.to_datetime(df['Date'])
        except Exception:
            return False, "Invalid date format in Date column"
        try:
            df['Amount'].replace('[\$,]', '', regex=True).astype(float)
        except Exception:
            return False, "Invalid amount format in Amount column"
        return True, "CSV format is valid"
    except Exception as e:
        return False, f"Invalid CSV format: {str(e)}"


def display_csv_preview(df: pd.DataFrame):
    st.subheader("CSV Data Preview")
    total_transactions = len(df)
    total_amount = df['Amount'].sum()
    df_dates = pd.to_datetime(df['Date'])
    date_range = f"{df_dates.min().strftime('%Y-%m-%d')} to {df_dates.max().strftime('%Y-%m-%d')}"

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Transactions", total_transactions)
    with col2:
        st.metric("Total Amount", f"₹{total_amount:,.2f}")
    with col3:
        st.metric("Date Range", date_range)

    st.subheader("Spending by Category")
    category_totals = df.groupby('Category')['Amount'].agg(['sum', 'count']).reset_index()
    category_totals.columns = ['Category', 'Total Amount', 'Transaction Count']
    st.dataframe(category_totals)

    st.subheader("Sample Transactions")
    st.dataframe(df.head())


# ─────────────────────────────────────────────
# Main App
# ─────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="SpendSmartAI",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    with st.sidebar:
        st.title("📊 Expense Tracker Guide")
        st.info(
            "Upload your expense data to get insights into your spending habits. "
            "This tool helps you understand where your money goes and how you can manage it better."
        )
        st.divider()
        st.subheader("📂 Upload Format")
        st.write("Your file should include:")
        st.write("- Date (YYYY-MM-DD)")
        st.write("- Category")
        st.write("- Amount")
        st.write("Download a sample file to follow the correct format 👇")

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
2024-01-20,Healthcare,3500.00"""

        st.download_button(
            label="📥 Download Sample CSV",
            data=sample_csv,
            file_name="expense_template.csv",
            mime="text/csv"
        )

        st.divider()
        st.subheader("🚨 Anomaly Detection")
        st.info(
            "The app uses an **Isolation Forest** ML model trained on YOUR own "
            "transaction history to detect unusual spending — not generic rules, "
            "but your personal patterns. Requires at least 10 transactions."
        )

    if not GEMINI_API_KEY:
        st.error("🔑 GOOGLE_API_KEY not found in environment variables. Please add it to your .env file.")
        return

    st.title("SpendSmartAI")
    st.caption("Track your expenses. Understand your spending. Detect anomalies. Improve your savings.")
    st.info(
        "Upload your expense data to get a clear breakdown of where your money is going. "
        "Discover spending patterns, identify unusual transactions with ML-powered anomaly detection, "
        "and get AI-driven suggestions to manage your budget better."
    )
    st.divider()

    input_tab, about_tab = st.tabs(["Financial Data", "Information"])

    with input_tab:
        st.header("Enter Your Financial Information")
        st.caption("All data is processed locally and not stored anywhere.")

        with st.container():
            st.subheader("💰 Income & Household")
            income_col, dependants_col = st.columns([2, 1])
            with income_col:
                monthly_income = st.number_input(
                    "Monthly Income (₹)", min_value=0.0, step=1000.0, value=50000.0,
                    key="income", help="Enter your total monthly income after taxes"
                )
            with dependants_col:
                dependants = st.number_input(
                    "Number of Dependants", min_value=0, step=1, value=0,
                    key="dependants", help="Include all dependants in your household"
                )

        st.divider()

        with st.container():
            st.subheader("💳 Expenses")
            expense_option = st.radio(
                "How would you like to enter your expenses?",
                ("📤 Upload CSV Transactions", "✍️ Enter Manually"),
                key="expense_option", horizontal=True
            )

            transaction_file = None
            manual_expenses = {}
            use_manual_expenses = False
            transactions_df = None

            if expense_option == "📤 Upload CSV Transactions":
                col1, _ = st.columns([2, 1])
                with col1:
                    st.markdown("""
                    #### Upload your transaction data
                    Your CSV file should have these columns:
                    - 📅 Date (YYYY-MM-DD)
                    - 📝 Category
                    - 💲 Amount
                    """)
                    transaction_file = st.file_uploader(
                        "Choose your CSV file", type=["csv"], key="transaction_file",
                        help="Upload a CSV file containing your transactions"
                    )

                if transaction_file is not None:
                    is_valid, message = validate_csv_format(transaction_file)
                    if is_valid:
                        try:
                            transaction_file.seek(0)
                            file_content = transaction_file.read()
                            parsed_data = parse_csv_transactions(file_content)
                            transactions_df = pd.DataFrame(parsed_data['transactions'])
                            display_csv_preview(transactions_df)
                            st.success("✅ Transaction file uploaded and validated successfully!")
                        except Exception as e:
                            st.error(f"❌ Error processing CSV file: {str(e)}")
                            transactions_df = None
                    else:
                        st.error(message)
                        transactions_df = None
            else:
                use_manual_expenses = True
                st.markdown("#### Enter your monthly expenses by category")
                categories = [
                    ("🏠 Housing", "Housing"), ("🔌 Utilities", "Utilities"),
                    ("🍽️ Food", "Food"), ("🚗 Transportation", "Transportation"),
                    ("🏥 Healthcare", "Healthcare"), ("🎭 Entertainment", "Entertainment"),
                    ("👤 Personal", "Personal"), ("💰 Savings", "Savings"),
                    ("📦 Other", "Other")
                ]
                col1, col2, col3 = st.columns(3)
                cols = [col1, col2, col3]
                for i, (emoji_cat, cat) in enumerate(categories):
                    with cols[i % 3]:
                        manual_expenses[cat] = st.number_input(
                            emoji_cat, min_value=0.0, step=500.0, value=0.0,
                            key=f"manual_{cat}", help=f"Enter your monthly {cat.lower()} expenses"
                        )

                if manual_expenses and any(manual_expenses.values()):
                    st.markdown("#### 📊 Summary of Entered Expenses")
                    manual_df_disp = pd.DataFrame({
                        'Category': list(manual_expenses.keys()),
                        'Amount': list(manual_expenses.values())
                    })
                    manual_df_disp = manual_df_disp[manual_df_disp['Amount'] > 0]
                    if not manual_df_disp.empty:
                        col1, col2 = st.columns([2, 1])
                        with col1:
                            st.dataframe(
                                manual_df_disp,
                                column_config={
                                    "Amount": st.column_config.NumberColumn("Amount", format="₹%.2f")
                                },
                                hide_index=True
                            )
                        with col2:
                            st.metric("Total Monthly Expenses", f"₹{manual_df_disp['Amount'].sum():,.2f}")

        st.divider()

        with st.container():
            st.subheader("🏦 Debt Information")
            st.info("Enter your debts to get personalized payoff strategies using both avalanche and snowball methods.")
            num_debts = st.number_input(
                "How many debts do you have?", min_value=0, max_value=10,
                step=1, value=0, key="num_debts"
            )
            debts = []
            if num_debts > 0:
                cols = st.columns(min(num_debts, 3))
                for i in range(num_debts):
                    col_idx = i % 3
                    with cols[col_idx]:
                        st.markdown(f"##### Debt #{i+1}")
                        debt_name = st.text_input("Name", value=f"Debt {i+1}", key=f"debt_name_{i}")
                        debt_amount = st.number_input("Amount (₹)", min_value=0.01, step=1000.0, value=10000.0, key=f"debt_amount_{i}")
                        interest_rate = st.number_input("Interest Rate (%)", min_value=0.0, max_value=100.0, step=0.1, value=12.0, key=f"debt_rate_{i}")
                        min_payment = st.number_input("Minimum Payment (₹)", min_value=0.0, step=500.0, value=500.0, key=f"debt_min_payment_{i}")
                        debts.append({
                            "name": debt_name, "amount": debt_amount,
                            "interest_rate": interest_rate, "min_payment": min_payment
                        })
                        if col_idx == 2 or i == num_debts - 1:
                            st.markdown("---")

        st.divider()

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            analyze_button = st.button(
                "🔄 Analyze My Finances",
                key="analyze_button",
                use_container_width=True,
                help="Click to get your personalized financial analysis"
            )

        if analyze_button:
            if expense_option == "📤 Upload CSV Transactions" and transactions_df is None:
                st.error("Please upload a valid transaction CSV file or choose manual entry.")
                return
            if use_manual_expenses and (not manual_expenses or not any(manual_expenses.values())):
                st.warning("No manual expenses entered. Analysis might be limited.")

            st.header("Financial Analysis Results")

            with st.spinner("🤖 AI agents are analyzing your financial data..."):
                financial_data = {
                    "monthly_income": monthly_income,
                    "dependants": dependants,
                    "transactions": transactions_df.to_dict('records') if transactions_df is not None else None,
                    "manual_expenses": manual_expenses if use_manual_expenses else None,
                    "debts": debts
                }

                finance_system = FinanceAdvisorSystem()

                try:
                    results = asyncio.run(finance_system.analyze_finances(financial_data))

                    tabs = st.tabs([
                        "💰 Budget Analysis",
                        "📈 Savings Strategy",
                        "💳 Debt Reduction",
                        "🚨 Anomaly Detection"
                    ])

                    with tabs[0]:
                        st.subheader("Budget Analysis")
                        if results.get("budget_analysis"):
                            display_budget_analysis(results["budget_analysis"])
                        else:
                            st.write("No budget analysis available.")

                    with tabs[1]:
                        st.subheader("Savings Strategy")
                        if results.get("savings_strategy"):
                            display_savings_strategy(results["savings_strategy"])
                        else:
                            st.write("No savings strategy available.")

                    with tabs[2]:
                        st.subheader("Debt Reduction Plan")
                        if results.get("debt_reduction"):
                            display_debt_reduction(results["debt_reduction"])
                        else:
                            st.write("No debt reduction plan available.")

                    with tabs[3]:
                        st.subheader("🚨 Spending Anomaly Detection")
                        display_anomaly_detection(
                            results.get("anomaly_df"),
                            results.get("anomaly_analysis")
                        )

                except Exception as e:
                    st.error(f"An error occurred during analysis: {str(e)}")

    with about_tab:
        st.markdown("""
        ### About SpendSmartAI

        This application uses Google's Agent Development Kit (ADK) and scikit-learn ML
        to provide comprehensive financial analysis through multiple specialized components:

        1. **🔍 Budget Analysis Agent**
           - Analyzes spending patterns
           - Identifies areas for cost reduction
           - Provides actionable recommendations

        2. **💰 Savings Strategy Agent**
           - Creates personalized savings plans
           - Calculates emergency fund requirements
           - Suggests automation techniques

        3. **💳 Debt Reduction Agent**
           - Develops optimal debt payoff strategies
           - Compares avalanche vs snowball methods
           - Provides actionable debt reduction tips

        4. **🚨 Anomaly Detection (NEW)**
           - Trains an **Isolation Forest** ML model on YOUR transaction history
           - Detects unusual spending that deviates from your personal patterns
           - AI agent explains each flagged transaction in plain English
           - Assigns severity levels (low / medium / high) and recommendations
           - Requires at least 10 transactions for reliable detection

        ### Privacy & Security
        - All data is processed locally
        - No financial information is stored or transmitted
        - The ML model is trained fresh each session on your data only
        """)


if __name__ == "__main__":
    main()