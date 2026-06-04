"""
charts.py — All Plotly visualisation functions
Warehouse Rental Management Analytics — Data Analytics MGB Project
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from utils import COLORS, PALETTE, PLOTLY_TEMPLATE

# ── SHARED LAYOUT ─────────────────────────────────────────────────────────────
def _base_layout(**kwargs):
    base = dict(
        template=PLOTLY_TEMPLATE,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans, Arial, sans-serif", color="#2D3A47"),
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="#E5E0D8",
            borderwidth=1,
            font=dict(size=11),
        ),
    )
    base.update(kwargs)
    return base


# ════════════════════════════════════════════════════════════════════════════
# DESCRIPTIVE CHARTS
# ════════════════════════════════════════════════════════════════════════════

def monthly_revenue_trend(df: pd.DataFrame) -> go.Figure:
    monthly = (
        df.groupby(df["Month"].dt.to_period("M"))
        .agg(
            Invoiced=("Total_Invoice_INR", "sum"),
            Collected=("Revenue_Collected_INR", "sum"),
            Outstanding=("Balance_Due_INR", "sum"),
        )
        .reset_index()
    )
    monthly["Month"] = monthly["Month"].dt.strftime("%b %Y")

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(
            x=monthly["Month"], y=monthly["Invoiced"],
            name="Invoiced", marker_color=COLORS["blue"],
            opacity=0.35, offsetgroup=0,
        ), secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=monthly["Month"], y=monthly["Collected"],
            name="Collected", line=dict(color=COLORS["orange"], width=2.8),
            mode="lines+markers", marker=dict(size=6),
        ), secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=monthly["Month"], y=monthly["Outstanding"],
            name="Outstanding", line=dict(color=COLORS["red"], width=1.8, dash="dot"),
            mode="lines+markers", marker=dict(size=5),
        ), secondary_y=False,
    )
    fig.update_layout(
        title="Monthly Revenue — Invoiced vs Collected vs Outstanding",
        xaxis=dict(tickangle=-45, showgrid=False),
        yaxis=dict(title="Amount (₹)", tickformat=","),
        **_base_layout(),
    )
    return fig


def owner_revenue_bar(df: pd.DataFrame) -> go.Figure:
    own = (
        df.groupby("Owner_Name")
        .agg(Invoiced=("Total_Invoice_INR", "sum"),
             Collected=("Revenue_Collected_INR", "sum"))
        .sort_values("Invoiced", ascending=False)
        .reset_index()
    )
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=own["Owner_Name"], y=own["Invoiced"],
        name="Invoiced", marker_color=COLORS["blue"], opacity=0.45,
        marker_line_width=0,
    ))
    fig.add_trace(go.Bar(
        x=own["Owner_Name"], y=own["Collected"],
        name="Collected", marker_color=COLORS["teal"],
        marker_line_width=0,
    ))
    fig.update_layout(
        title="Owner-wise Revenue — Invoiced vs Collected",
        barmode="group",
        xaxis=dict(showgrid=False),
        yaxis=dict(title="Amount (₹)", tickformat=","),
        **_base_layout(),
    )
    return fig


def owner_revenue_pie(df: pd.DataFrame) -> go.Figure:
    own = (
        df.groupby("Owner_Name")["Revenue_Collected_INR"]
        .sum().reset_index()
    )
    fig = px.pie(
        own, names="Owner_Name", values="Revenue_Collected_INR",
        title="Revenue Share by Owner",
        color_discrete_sequence=PALETTE,
        hole=0.45,
    )
    fig.update_traces(textinfo="label+percent", pull=[0.05] + [0] * (len(own) - 1))
    fig.update_layout(**_base_layout())
    return fig


def location_bar(df: pd.DataFrame) -> go.Figure:
    loc = (
        df.groupby("Warehouse_Location")
        .agg(Revenue=("Revenue_Collected_INR", "sum"),
             Outstanding=("Balance_Due_INR", "sum"),
             Paid_Rate=("Is_Paid_Binary", "mean"))
        .sort_values("Revenue", ascending=True)
        .reset_index()
    )
    loc["Paid_Pct"] = (loc["Paid_Rate"] * 100).round(1)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=loc["Warehouse_Location"], x=loc["Revenue"],
        name="Collected", marker_color=COLORS["blue"],
        orientation="h", text=loc["Revenue"].apply(lambda v: f"₹{v/1e5:.1f}L"),
        textposition="outside",
    ))
    fig.add_trace(go.Bar(
        y=loc["Warehouse_Location"], x=loc["Outstanding"],
        name="Outstanding", marker_color=COLORS["orange"],
        orientation="h",
    ))
    fig.update_layout(
        title="Location-wise Revenue vs Outstanding",
        barmode="stack",
        xaxis=dict(title="Amount (₹)", tickformat=",", showgrid=True),
        yaxis=dict(showgrid=False),
        **_base_layout(),
    )
    return fig


def payment_status_donut(df: pd.DataFrame) -> go.Figure:
    counts = df["Payment_Status"].value_counts().reset_index()
    counts.columns = ["Status", "Count"]
    fig = px.pie(
        counts, names="Status", values="Count",
        title="Payment Status Distribution",
        color="Status",
        color_discrete_map={"Paid": COLORS["teal"], "Not Paid": COLORS["red"]},
        hole=0.55,
    )
    fig.update_traces(textinfo="label+percent+value")
    fig.update_layout(**_base_layout())
    return fig


def payment_behaviour_bar(df: pd.DataFrame) -> go.Figure:
    order = ["On-time", "Slightly Delayed", "Delayed", "Defaulted"]
    counts = (
        df["Payment_Behavior"].value_counts()
        .reindex(order, fill_value=0)
        .reset_index()
    )
    counts.columns = ["Behavior", "Count"]
    colour_map = {
        "On-time": COLORS["green"],
        "Slightly Delayed": COLORS["gold"],
        "Delayed": COLORS["orange"],
        "Defaulted": COLORS["red"],
    }
    fig = px.bar(
        counts, x="Behavior", y="Count",
        title="Payment Behaviour Distribution",
        color="Behavior",
        color_discrete_map=colour_map,
        text="Count",
    )
    fig.update_traces(textposition="outside", marker_line_width=0)
    fig.update_layout(
        showlegend=False,
        xaxis=dict(showgrid=False),
        yaxis=dict(title="Number of Transactions"),
        **_base_layout(),
    )
    return fig


def quarterly_revenue(df: pd.DataFrame) -> go.Figure:
    df2 = df.copy()
    df2["Quarter"] = df2["Month"].dt.to_period("Q").astype(str)
    qtr = (
        df2.groupby("Quarter")["Revenue_Collected_INR"]
        .sum().reset_index()
    )
    fig = px.bar(
        qtr, x="Quarter", y="Revenue_Collected_INR",
        title="Quarterly Revenue Collected",
        color="Quarter",
        color_discrete_sequence=PALETTE,
        text=qtr["Revenue_Collected_INR"].apply(lambda v: f"₹{v/1e5:.1f}L"),
    )
    fig.update_traces(textposition="outside", marker_line_width=0)
    fig.update_layout(
        showlegend=False,
        xaxis=dict(showgrid=False),
        yaxis=dict(title="Revenue (₹)", tickformat=","),
        **_base_layout(),
    )
    return fig


def wh_type_revenue(df: pd.DataFrame) -> go.Figure:
    t = (
        df.groupby("Warehouse_Type")
        .agg(Revenue=("Revenue_Collected_INR", "sum"),
             Paid_Rate=("Is_Paid_Binary", "mean"))
        .sort_values("Revenue", ascending=False)
        .reset_index()
    )
    t["Paid_Pct"] = (t["Paid_Rate"] * 100).round(1)
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(
        x=t["Warehouse_Type"], y=t["Revenue"],
        name="Revenue", marker_color=COLORS["blue"],
        marker_line_width=0,
    ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=t["Warehouse_Type"], y=t["Paid_Pct"],
        name="Paid Rate %", mode="lines+markers+text",
        line=dict(color=COLORS["orange"], width=2.5),
        marker=dict(size=9),
        text=[f"{v}%" for v in t["Paid_Pct"]],
        textposition="top center",
    ), secondary_y=True)
    fig.update_layout(
        title="Warehouse Type — Revenue & Payment Rate",
        xaxis=dict(showgrid=False),
        **_base_layout(),
    )
    fig.update_yaxes(title_text="Revenue (₹)", secondary_y=False)
    fig.update_yaxes(title_text="Paid Rate (%)", range=[0, 120], secondary_y=True)
    return fig


def automation_summary(df: pd.DataFrame) -> go.Figure:
    cols = ["Invoice_Sent", "Reminder_Sent", "Email_Sent", "WhatsApp_Sent"]
    labels, vals = [], []
    for c in cols:
        if c in df.columns:
            pct = round((df[c] == "Yes").mean() * 100, 1)
            labels.append(c.replace("_", " "))
            vals.append(pct)
    fig = go.Figure(go.Bar(
        x=vals, y=labels,
        orientation="h",
        marker_color=[COLORS["blue"], COLORS["orange"], COLORS["teal"], COLORS["gold"]],
        text=[f"{v}%" for v in vals],
        textposition="inside",
    ))
    fig.update_layout(
        title="Operational Automation — Coverage Rate",
        xaxis=dict(title="Coverage (%)", range=[0, 115], showgrid=False),
        yaxis=dict(showgrid=False),
        **_base_layout(),
    )
    return fig


# ════════════════════════════════════════════════════════════════════════════
# CORRELATION / DIAGNOSTIC CHARTS
# ════════════════════════════════════════════════════════════════════════════

def correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    cols = {
        "Monthly_Rent_INR":       "Monthly Rent",
        "Size_Encoded":           "WH Size",
        "Type_Encoded":           "WH Type",
        "Delay_Days":             "Delay Days",
        "TenantType_Encoded":     "Tenant Type",
        "Customer_Tenure_Months": "Tenure",
        "Risk_Score":             "Risk Score",
        "Is_Paid_Binary":         "Is Paid",
        "Lease_Duration_Months":  "Lease Duration",
        "Month_Num":              "Month No.",
    }
    available = {k: v for k, v in cols.items() if k in df.columns}
    sub = df[list(available.keys())].rename(columns=available)
    corr = sub.corr().round(2)

    fig = go.Figure(go.Heatmap(
        z=corr.values,
        x=corr.columns.tolist(),
        y=corr.index.tolist(),
        colorscale="RdYlGn",
        zmin=-1, zmax=1,
        text=corr.values.round(2),
        texttemplate="%{text}",
        textfont=dict(size=10),
        colorbar=dict(title="Pearson r"),
        hoverongaps=False,
    ))
    fig.update_layout(
        title="Correlation Heatmap — All Key Variables",
        xaxis=dict(tickangle=-40),
        **_base_layout(margin=dict(l=10, r=10, t=60, b=80)),
    )
    return fig


def rent_vs_size(df: pd.DataFrame) -> go.Figure:
    size_order = ["Small", "Medium", "Large"]
    size_avg = (
        df.groupby("Warehouse_Size")["Monthly_Rent_INR"]
        .mean().reindex(size_order).reset_index()
    )
    fig = px.bar(
        size_avg, x="Warehouse_Size", y="Monthly_Rent_INR",
        title="Avg Monthly Rent by Warehouse Size",
        color="Warehouse_Size",
        color_discrete_sequence=[COLORS["teal"], COLORS["blue"], COLORS["orange"]],
        text=size_avg["Monthly_Rent_INR"].apply(lambda v: f"₹{v/1e3:.0f}K"),
    )
    fig.update_traces(textposition="outside", marker_line_width=0)
    fig.update_layout(
        showlegend=False,
        xaxis=dict(showgrid=False, title="Warehouse Size"),
        yaxis=dict(title="Avg Monthly Rent (₹)", tickformat=","),
        **_base_layout(),
    )
    return fig


def rent_vs_type(df: pd.DataFrame) -> go.Figure:
    type_order = ["General", "Dry Warehouse", "Cold Storage", "Distribution Hub"]
    type_avg = (
        df.groupby("Warehouse_Type")["Monthly_Rent_INR"]
        .mean().reindex(type_order).reset_index()
    )
    fig = px.bar(
        type_avg, x="Warehouse_Type", y="Monthly_Rent_INR",
        title="Avg Monthly Rent by Warehouse Type",
        color="Warehouse_Type",
        color_discrete_sequence=PALETTE,
        text=type_avg["Monthly_Rent_INR"].apply(lambda v: f"₹{v/1e3:.0f}K"),
    )
    fig.update_traces(textposition="outside", marker_line_width=0)
    fig.update_layout(
        showlegend=False,
        xaxis=dict(showgrid=False, title="Warehouse Type"),
        yaxis=dict(title="Avg Monthly Rent (₹)", tickformat=","),
        **_base_layout(),
    )
    return fig


def delay_by_industry(df: pd.DataFrame) -> go.Figure:
    ind = (
        df.groupby("Industry_Type")["Delay_Days"]
        .mean().sort_values(ascending=True).reset_index()
    )
    colors = [
        COLORS["red"] if v > 20 else COLORS["gold"] if v > 10 else COLORS["teal"]
        for v in ind["Delay_Days"]
    ]
    fig = go.Figure(go.Bar(
        y=ind["Industry_Type"], x=ind["Delay_Days"].round(1),
        orientation="h", marker_color=colors,
        text=ind["Delay_Days"].round(1),
        textposition="outside",
    ))
    fig.update_layout(
        title="Avg Delay Days by Industry",
        xaxis=dict(title="Avg Delay Days", showgrid=True),
        yaxis=dict(showgrid=False),
        **_base_layout(),
    )
    return fig


def delay_by_tenant_type(df: pd.DataFrame) -> go.Figure:
    tt = (
        df.groupby("Tenant_Type")["Delay_Days"]
        .mean().reset_index()
    )
    fig = px.bar(
        tt, x="Tenant_Type", y="Delay_Days",
        title="Avg Delay Days — Business vs Individual",
        color="Tenant_Type",
        color_discrete_map={"Business": COLORS["blue"], "Individual": COLORS["orange"]},
        text=tt["Delay_Days"].round(1),
    )
    fig.update_traces(textposition="outside", marker_line_width=0)
    fig.update_layout(
        showlegend=False,
        xaxis=dict(showgrid=False),
        yaxis=dict(title="Avg Delay Days"),
        **_base_layout(),
    )
    return fig


def risk_score_histogram(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=df["Risk_Score"], nbinsx=20,
        marker_color=COLORS["blue"], opacity=0.8,
        name="Risk Score",
    ))
    fig.add_vline(x=df["Risk_Score"].mean(), line_dash="dash",
                  line_color=COLORS["orange"],
                  annotation_text=f"Mean: {df['Risk_Score'].mean():.1f}",
                  annotation_position="top right")
    fig.add_vline(x=70, line_dash="dot",
                  line_color=COLORS["red"],
                  annotation_text="High Risk (70)",
                  annotation_position="top left")
    fig.update_layout(
        title="Tenant Risk Score Distribution",
        xaxis=dict(title="Risk Score (0–100)", showgrid=False),
        yaxis=dict(title="Frequency"),
        **_base_layout(),
    )
    return fig


def location_type_heatmap(df: pd.DataFrame) -> go.Figure:
    pivot = df.pivot_table(
        index="Warehouse_Type",
        columns="Warehouse_Location",
        values="Monthly_Rent_INR",
        aggfunc="mean",
    ).fillna(0)
    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=pivot.columns.tolist(),
        y=pivot.index.tolist(),
        colorscale="YlOrBr",
        text=np.round(pivot.values / 1000, 0),
        texttemplate="₹%{text}K",
        textfont=dict(size=10),
        colorbar=dict(title="Avg Rent (₹)"),
    ))
    fig.update_layout(
        title="Location × WH Type — Avg Monthly Rent Heatmap",
        xaxis=dict(tickangle=-30),
        **_base_layout(margin=dict(l=10, r=10, t=60, b=80)),
    )
    return fig


# ════════════════════════════════════════════════════════════════════════════
# CLASSIFICATION CHARTS
# ════════════════════════════════════════════════════════════════════════════

def roc_curve_chart(fpr, tpr, auc: float) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr,
        mode="lines",
        name=f"ROC Curve (AUC = {auc:.3f})",
        line=dict(color=COLORS["blue"], width=2.5),
        fill="tozeroy",
        fillcolor=f"rgba(26,60,94,0.08)",
    ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode="lines",
        name="Random Classifier",
        line=dict(color=COLORS["grey"], width=1.5, dash="dash"),
    ))
    fig.update_layout(
        title=f"ROC Curve — Payment Classification (AUC = {auc:.3f})",
        xaxis=dict(title="False Positive Rate", showgrid=True),
        yaxis=dict(title="True Positive Rate", showgrid=True),
        **_base_layout(),
    )
    return fig


def confusion_matrix_chart(cm) -> go.Figure:
    labels = ["Not Paid", "Paid"]
    pct = cm / cm.sum() * 100
    text = [[f"{cm[i][j]}<br>({pct[i][j]:.1f}%)" for j in range(2)] for i in range(2)]
    fig = go.Figure(go.Heatmap(
        z=cm[::-1], x=labels, y=labels[::-1],
        colorscale=[[0, "#FEF0EA"], [1, COLORS["blue"]]],
        text=text[::-1],
        texttemplate="%{text}",
        textfont=dict(size=13),
        showscale=False,
    ))
    fig.update_layout(
        title="Confusion Matrix",
        xaxis=dict(title="Predicted", showgrid=False),
        yaxis=dict(title="Actual", showgrid=False),
        **_base_layout(),
    )
    return fig


def feature_importance_chart(names: list, importances) -> go.Figure:
    fi = pd.Series(importances, index=names).sort_values(ascending=True)
    colors = [
        COLORS["orange"] if v == fi.max() else COLORS["blue"]
        for v in fi.values
    ]
    fig = go.Figure(go.Bar(
        x=fi.values, y=fi.index.tolist(),
        orientation="h",
        marker_color=colors,
        text=[f"{v:.3f}" for v in fi.values],
        textposition="outside",
    ))
    fig.update_layout(
        title="Feature Importance — Random Forest",
        xaxis=dict(title="Importance Score", showgrid=False),
        yaxis=dict(showgrid=False),
        **_base_layout(),
    )
    return fig


# ════════════════════════════════════════════════════════════════════════════
# REGRESSION CHARTS
# ════════════════════════════════════════════════════════════════════════════

def regression_actual_vs_predicted(y_test, y_pred, title: str, unit: str = "₹") -> go.Figure:
    mn = float(min(y_test.min(), y_pred.min()))
    mx = float(max(y_test.max(), y_pred.max()))
    residuals = np.array(y_test) - np.array(y_pred)

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["Actual vs Predicted", "Residuals"])
    fig.add_trace(go.Scatter(
        x=list(y_test), y=list(y_pred),
        mode="markers",
        marker=dict(color=COLORS["blue"], size=7, opacity=0.7),
        name="Predictions",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=[mn, mx], y=[mn, mx],
        mode="lines",
        line=dict(color=COLORS["orange"], dash="dash", width=2),
        name="Perfect Fit",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=list(y_pred), y=list(residuals),
        mode="markers",
        marker=dict(color=COLORS["teal"], size=6, opacity=0.65),
        name="Residuals",
    ), row=1, col=2)
    fig.add_hline(y=0, line_dash="dash", line_color=COLORS["grey"], row=1, col=2)

    fig.update_layout(title=title, **_base_layout())
    return fig


# ════════════════════════════════════════════════════════════════════════════
# CLUSTERING CHARTS
# ════════════════════════════════════════════════════════════════════════════

def elbow_chart(inertias: list) -> go.Figure:
    k_range = list(range(2, 2 + len(inertias)))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=k_range, y=inertias,
        mode="lines+markers",
        line=dict(color=COLORS["blue"], width=2.5),
        marker=dict(size=9, color=COLORS["orange"]),
        name="Inertia (WCSS)",
    ))
    fig.add_vline(x=4, line_dash="dash", line_color=COLORS["red"],
                  annotation_text="Optimal K=4", annotation_position="top right")
    fig.update_layout(
        title="Elbow Method — Optimal K Selection",
        xaxis=dict(title="Number of Clusters (K)", showgrid=False, dtick=1),
        yaxis=dict(title="Inertia (WCSS)"),
        **_base_layout(),
    )
    return fig


def cluster_scatter(df_clustered: pd.DataFrame) -> go.Figure:
    colour_map = {
        "High-Value Regular": COLORS["orange"],
        "Regular Payer":      COLORS["teal"],
        "Late Payer":         COLORS["gold"],
        "At-Risk":            COLORS["red"],
    }
    fig = px.scatter(
        df_clustered,
        x="Monthly_Rent_INR", y="Delay_Days",
        color="Cluster_Label",
        size="Risk_Score",
        hover_data=["Tenant_Name", "Payment_Behavior", "Customer_Tenure_Months"]
        if "Tenant_Name" in df_clustered.columns else None,
        title="K-Means Clusters — Rent vs Delay Days",
        color_discrete_map=colour_map,
        size_max=20,
    )
    fig.update_layout(
        xaxis=dict(title="Monthly Rent (₹)", tickformat=","),
        yaxis=dict(title="Delay Days"),
        **_base_layout(),
    )
    return fig


def cluster_profile_radar(profile: pd.DataFrame) -> go.Figure:
    cats = ["Avg_Rent", "Avg_Delay", "Paid_Rate"]
    cat_labels = ["Avg Rent (norm)", "Avg Delay (norm)", "Paid Rate (norm)"]
    colour_map = {
        "High-Value Regular": COLORS["orange"],
        "Regular Payer":      COLORS["teal"],
        "Late Payer":         COLORS["gold"],
        "At-Risk":            COLORS["red"],
    }
    norm = profile[cats].copy()
    for c in cats:
        rng = norm[c].max() - norm[c].min()
        norm[c] = (norm[c] - norm[c].min()) / rng if rng > 0 else 0

    fig = go.Figure()
    for _, row in profile.iterrows():
        lbl = row.get("Label", "Unknown")
        vals = [norm.loc[row.name, c] for c in cats]
        vals += [vals[0]]
        fig.add_trace(go.Scatterpolar(
            r=vals,
            theta=cat_labels + [cat_labels[0]],
            fill="toself",
            name=lbl,
            line_color=colour_map.get(lbl, COLORS["grey"]),
            fillcolor=colour_map.get(lbl, COLORS["grey"]) + "44",
        ))
    fig.update_layout(
        title="Cluster Profile Radar Chart",
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        **_base_layout(),
    )
    return fig


# ════════════════════════════════════════════════════════════════════════════
# ASSOCIATION RULES CHARTS
# ════════════════════════════════════════════════════════════════════════════

def association_heatmap(rules_df: pd.DataFrame) -> go.Figure:
    paid_rules = rules_df[rules_df["Consequent"] == "Paid"].copy()
    if paid_rules.empty:
        paid_rules = rules_df.copy()

    pivot = paid_rules.pivot_table(
        index="WH_Type", columns="Tenant_Type",
        values="Confidence", aggfunc="mean"
    ).fillna(0)

    text_vals = (pivot * 100).round(1).astype(str) + "%"

    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=pivot.columns.tolist(),
        y=pivot.index.tolist(),
        colorscale="YlGn",
        zmin=0, zmax=1,
        text=text_vals.values,
        texttemplate="%{text}",
        textfont=dict(size=12),
        colorbar=dict(title="Confidence", tickformat=".0%"),
    ))
    fig.update_layout(
        title="Association Rules — Confidence Heatmap<br>(Tenant Type × WH Type → Paid)",
        xaxis=dict(title="Tenant Type"),
        yaxis=dict(title="WH Type"),
        **_base_layout(margin=dict(l=10, r=10, t=80, b=80)),
    )
    return fig


def association_bubble(rules_df: pd.DataFrame) -> go.Figure:
    top = rules_df.nlargest(20, "Lift").copy()
    fig = px.scatter(
        top,
        x="Support", y="Confidence",
        size="Lift", color="Lift",
        hover_data=["Antecedent", "Consequent", "Support", "Confidence", "Lift", "Count"],
        title="Association Rules — Support vs Confidence (Bubble = Lift)",
        color_continuous_scale="Oranges",
        size_max=35,
        text="Antecedent",
    )
    fig.update_traces(textposition="top center", textfont_size=9)
    fig.update_layout(
        xaxis=dict(title="Support"),
        yaxis=dict(title="Confidence"),
        **_base_layout(),
    )
    return fig


# ════════════════════════════════════════════════════════════════════════════
# PROSPECT SCORING CHARTS
# ════════════════════════════════════════════════════════════════════════════

def prospect_gauge(prob: float, name: str) -> go.Figure:
    color = (
        COLORS["green"] if prob >= 0.75
        else COLORS["gold"] if prob >= 0.5
        else COLORS["red"]
    )
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=round(prob * 100, 1),
        title={"text": f"Pay Probability — {name}", "font": {"size": 14}},
        delta={"reference": 75},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": color},
            "steps": [
                {"range": [0, 40],  "color": "#FCEAEA"},
                {"range": [40, 70], "color": "#FDF6E3"},
                {"range": [70, 100],"color": "#E8F6F0"},
            ],
            "threshold": {
                "line": {"color": COLORS["blue"], "width": 3},
                "thickness": 0.75,
                "value": 75,
            },
        },
        number={"suffix": "%"},
    ))
    fig.update_layout(**_base_layout(margin=dict(l=20, r=20, t=60, b=20)))
    return fig


def prospect_risk_bar(df_scored: pd.DataFrame) -> go.Figure:
    df_s = df_scored.sort_values("Pay_Probability", ascending=True)
    colors = [
        COLORS["green"] if p >= 0.75
        else COLORS["gold"] if p >= 0.5
        else COLORS["red"]
        for p in df_s["Pay_Probability"]
    ]
    name_col = "Tenant_Name" if "Tenant_Name" in df_s.columns else df_s.index.astype(str)
    fig = go.Figure(go.Bar(
        y=df_s[name_col] if "Tenant_Name" in df_s.columns else df_s.index.astype(str),
        x=(df_s["Pay_Probability"] * 100).round(1),
        orientation="h",
        marker_color=colors,
        text=[f"{v*100:.1f}%" for v in df_s["Pay_Probability"]],
        textposition="outside",
    ))
    fig.update_layout(
        title="Prospect Payment Probability Ranking",
        xaxis=dict(title="Pay Probability (%)", range=[0, 120], showgrid=False),
        yaxis=dict(showgrid=False),
        **_base_layout(),
    )
    return fig
