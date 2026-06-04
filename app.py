"""
app.py — WareHub Analytics v4
All 8 upgrades: SQLite storage, grouped nav, action list, rent calendar,
WhatsApp-first flow, tenant risk tool, vacancy tracker, P&L export.
Run: streamlit run app.py
"""
# ── Path fix for Streamlit Cloud (files may run from a parent directory) ──────
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# ─────────────────────────────────────────────────────────────────────────────
import io, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import streamlit as st

from notifications import render_notification_page
from settings import render_settings_page, init_settings
from admin import render_admin_page
from data_manager import get_transaction_df, get_business, get_expiring_leases, get_vacancy_stats
from decisions import render_action_list, render_tenant_risk_tool, render_vacancy_tracker
from rent_calendar import render_rent_calendar
from pl_export import render_pl_export
from utils import (
    load_data, apply_filters, compute_kpis, inr_fmt,
    train_classifier, train_rent_regressor, train_delay_regressor,
    train_clustering, compute_association_rules, score_prospects,
    COLORS, PALETTE, PROSPECT_SCHEMA, encode_prospect, SIZE_MAP, TYPE_MAP,
)
from charts import (
    monthly_revenue_trend, owner_revenue_bar, owner_revenue_pie,
    location_bar, payment_status_donut, payment_behaviour_bar,
    quarterly_revenue, wh_type_revenue, automation_summary,
    correlation_heatmap, rent_vs_size, rent_vs_type,
    delay_by_industry, delay_by_tenant_type, risk_score_histogram,
    location_type_heatmap,
    roc_curve_chart, confusion_matrix_chart, feature_importance_chart,
    regression_actual_vs_predicted,
    elbow_chart, cluster_scatter, cluster_profile_radar,
    association_heatmap, association_bubble,
    prospect_gauge, prospect_risk_bar,
)

st.set_page_config(page_title="WareHub Analytics",page_icon="🏭",layout="wide",initial_sidebar_state="expanded")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Serif+Display&display=swap');
html,body,[class*="css"]{font-family:'DM Sans',sans-serif;}
[data-testid="stSidebar"]{background:#0F1923;border-right:1px solid #1E2D3D;}
[data-testid="stSidebar"] *{color:#C8D8E8 !important;}
[data-testid="stSidebar"] h1,h2,h3{color:#FFFFFF !important;}
[data-testid="stSidebar"] .stMarkdown p{color:#90A4AE !important;font-size:12px;}
.nav-section{font-size:9px;font-weight:600;color:#546E7A !important;text-transform:uppercase;letter-spacing:.1em;padding:10px 4px 2px;margin:0;}
.kpi-card{background:#FFFFFF;border:1px solid #E5E0D8;border-radius:14px;padding:14px 16px;}
.kpi-label{font-size:10px;font-weight:600;color:#64748B;text-transform:uppercase;letter-spacing:.07em;margin-bottom:4px;}
.kpi-value{font-size:22px;font-weight:700;color:#0F1923;letter-spacing:-.03em;line-height:1.1;}
.kpi-footer{font-size:10px;color:#94A3B8;margin-top:4px;}
.kpi-tag{display:inline-block;font-size:9px;font-weight:600;padding:2px 8px;border-radius:10px;margin-top:6px;text-transform:uppercase;letter-spacing:.04em;}
.tag-green{background:#E8F6F0;color:#1D8A5F;}.tag-red{background:#FCEAEA;color:#C0392B;}
.tag-amber{background:#FDF6E3;color:#D97706;}.tag-blue{background:#EBF2FA;color:#1A3C5E;}
.alert-danger{background:#FCEAEA;border:1px solid #F5C4C4;border-radius:10px;padding:10px 14px;margin-bottom:14px;font-size:13px;color:#C0392B;}
.alert-info{background:#EBF2FA;border:1px solid #BDD5EE;border-radius:10px;padding:10px 14px;margin-bottom:14px;font-size:13px;color:#1A3C5E;}
.alert-success{background:#E8F6F0;border:1px solid #A8DDD8;border-radius:10px;padding:10px 14px;margin-bottom:14px;font-size:13px;color:#1D8A5F;}
.section-header{font-family:'DM Serif Display',serif;font-size:22px;color:#0F1923;letter-spacing:-.02em;margin-bottom:4px;}
.section-sub{font-size:13px;color:#64748B;margin-bottom:18px;}
.divider{border:none;border-top:1px solid #E5E0D8;margin:20px 0;}
</style>""", unsafe_allow_html=True)

# ─── DATA LOADING ─────────────────────────────────────────────────────────────
df_main = get_transaction_df()
init_settings()
biz = get_business()
if biz.get("phone") and not st.session_state.get("_phone_set"):
    import re as _re
    ph=_re.sub(r"\D","",biz["phone"])
    if not ph.startswith("91"): ph="91"+ph
    st.session_state["sender_phone"]=ph; st.session_state["_phone_set"]=True

# ─── SIDEBAR NAV ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏭 WareHub")
    st.markdown("<span style='font-size:11px;color:#90A4AE'>Data as of Dec 2024 · v4</span>",unsafe_allow_html=True)
    st.markdown("---")

    exp_n  = len(get_expiring_leases(30))
    vac_n  = sum(1 for v in get_vacancy_stats() if v["is_vacant"])
    ov_n   = int((df_main["Payment_Status"]=="Not Paid").sum()) if not df_main.empty else 0
    urg    = ov_n+exp_n
    badge  = "  🔴" if urg>0 else ""

    st.markdown('<p class="nav-section">Day-to-day</p>',unsafe_allow_html=True)
    pg1=st.radio("d",[f"📋 Action list{badge}","📅 Rent calendar","📨 Invoice & reminders"],label_visibility="collapsed",key="k1")
    st.markdown('<p class="nav-section">Analytics</p>',unsafe_allow_html=True)
    pg2=st.radio("a",["📊 Overview","💰 Revenue analysis","🔍 Why did it happen?"],label_visibility="collapsed",key="k2")
    st.markdown('<p class="nav-section">AI tools</p>',unsafe_allow_html=True)
    pg3=st.radio("i",["🔮 Should I sign this tenant?","🏭 Vacancy & renewals","🎯 Tenant segments","🔗 Patterns & rules","🚀 Score a prospect","📈 Rent & delay forecasts"],label_visibility="collapsed",key="k3")
    st.markdown('<p class="nav-section">Management</p>',unsafe_allow_html=True)
    pg4=st.radio("m",["📊 Annual P&L report","⚙️ Settings","🗂️ Admin — manage data"],label_visibility="collapsed",key="k4")

    # Track active page across groups
    _M={f"📋 Action list{badge}":"action","📅 Rent calendar":"calendar","📨 Invoice & reminders":"invoice","📊 Overview":"overview","💰 Revenue analysis":"revenue","🔍 Why did it happen?":"diagnostic","🔮 Should I sign this tenant?":"risk","🏭 Vacancy & renewals":"vacancy","🎯 Tenant segments":"clustering","🔗 Patterns & rules":"association","🚀 Score a prospect":"prospect","📈 Rent & delay forecasts":"regression","📊 Annual P&L report":"pl","⚙️ Settings":"settings","🗂️ Admin — manage data":"admin"}
    if "pg" not in st.session_state: st.session_state.update({"pg":"action","p1":pg1,"p2":pg2,"p3":pg3,"p4":pg4})
    if pg1!=st.session_state["p1"]: st.session_state["pg"]=_M.get(pg1,"action"); st.session_state["p1"]=pg1
    elif pg2!=st.session_state["p2"]: st.session_state["pg"]=_M.get(pg2,"overview"); st.session_state["p2"]=pg2
    elif pg3!=st.session_state["p3"]: st.session_state["pg"]=_M.get(pg3,"risk"); st.session_state["p3"]=pg3
    elif pg4!=st.session_state["p4"]: st.session_state["pg"]=_M.get(pg4,"pl"); st.session_state["p4"]=pg4
    page=st.session_state["pg"]

    # Filters (analytics pages only)
    df=df_main
    if page in {"overview","revenue","diagnostic"} and not df_main.empty:
        st.markdown("---")
        st.markdown("**Filters**")
        ow=["All"]+sorted(df_main["Owner_Name"].dropna().unique().tolist())
        lc=["All"]+sorted(df_main["Warehouse_Location"].dropna().unique().tolist())
        wt=["All"]+sorted(df_main["Warehouse_Type"].dropna().unique().tolist())
        yr=["All"]+sorted(df_main["Month"].dt.year.dropna().unique().astype(str).tolist())
        so=st.selectbox("Owner",ow); sl=st.selectbox("Location",lc); sw=st.selectbox("WH Type",wt)
        sy=st.selectbox("Year",yr); sp=st.selectbox("Payment",["All","Paid","Not Paid"])
        df=apply_filters(df_main,owner=so,location=sl,wh_type=sw,pay_status=sp,year=sy)
        n=len(df)
        if n<len(df_main): st.caption(f"🔍 {n}/{len(df_main)} records")

    st.markdown("---")
    from data_manager import get_tenants as _gt, get_warehouses as _gw
    st.caption(f"🏭 {len(_gw())} warehouses · {len(_gt())} tenants · {len(df_main)} rows")

# ─── KPI HELPERS ─────────────────────────────────────────────────────────────
def kpi_card(label,value,footer="",tag="",tc="tag-blue"):
    return f'<div class="kpi-card"><div class="kpi-label">{label}</div><div class="kpi-value">{value}</div><div class="kpi-footer">{footer}</div>{"<span class=\"kpi-tag "+tc+"\">"+tag+"</span>" if tag else ""}</div>'

def render_kpis(dk):
    kpis=compute_kpis(dk)
    c1,c2,c3,c4,c5,c6=st.columns(6)
    with c1: st.markdown(kpi_card("Total Invoiced",inr_fmt(kpis["total_inv"]),"incl. 18% GST","Info","tag-blue"),unsafe_allow_html=True)
    with c2:
        tc="tag-green" if kpis["col_rate"]>=85 else "tag-amber"
        st.markdown(kpi_card("Revenue Collected",inr_fmt(kpis["total_col"]),f"{kpis['col_rate']}% collection rate","On track" if kpis["col_rate"]>=85 else "Watch",tc),unsafe_allow_html=True)
    with c3:
        oc="tag-red" if kpis["total_out"]>100000 else "tag-amber"
        st.markdown(kpi_card("Outstanding",inr_fmt(kpis["total_out"]),f"{kpis['high_risk']} high-risk records","Collect now" if kpis["total_out"]>100000 else "Low",oc),unsafe_allow_html=True)
    with c4:
        pc="tag-green" if kpis["paid_rate"]>=85 else "tag-amber"
        st.markdown(kpi_card("Payment Rate",f"{kpis['paid_rate']}%","of all transactions","On track" if kpis["paid_rate"]>=85 else "Review",pc),unsafe_allow_html=True)
    with c5: st.markdown(kpi_card("Avg Monthly Rent",inr_fmt(kpis["avg_rent"]),"active contracts","Info","tag-blue"),unsafe_allow_html=True)
    with c6:
        dc="tag-red" if kpis["avg_delay"]>20 else "tag-amber"
        st.markdown(kpi_card("Avg Delay (late)",f"{kpis['avg_delay']}d","delayed records only","High" if kpis["avg_delay"]>20 else "Moderate",dc),unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# PAGES
# ═══════════════════════════════════════════════════════════════════════════

if page=="action":
    render_action_list()

elif page=="calendar":
    render_rent_calendar()

elif page=="invoice":
    render_notification_page(df)

elif page=="overview":
    st.markdown('<div class="section-header">📊 Portfolio Overview</div>',unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Family warehouse rental business — summary of all operations</div>',unsafe_allow_html=True)
    kpis=compute_kpis(df)
    if kpis["total_out"]>0:
        ot=df[df["Balance_Due_INR"]>0]["Tenant_Name"].unique()
        st.markdown(f'<div class="alert-danger">⚠️ <strong>Collect now:</strong> {inr_fmt(kpis["total_out"])} outstanding across {len(ot)} tenant(s): {", ".join(ot[:3])}{"..." if len(ot)>3 else ""}. Go to Invoice & Reminders.</div>',unsafe_allow_html=True)
    render_kpis(df)
    st.markdown("<hr class='divider'>",unsafe_allow_html=True)
    c1,c2,c3=st.columns([2,1,1])
    with c1: st.plotly_chart(monthly_revenue_trend(df),use_container_width=True)
    with c2: st.plotly_chart(payment_status_donut(df),use_container_width=True)
    with c3: st.plotly_chart(wh_type_revenue(df),use_container_width=True)
    c1,c2=st.columns([3,2])
    with c1: st.plotly_chart(location_bar(df),use_container_width=True)
    with c2: st.plotly_chart(payment_behaviour_bar(df),use_container_width=True)
    st.markdown("#### Tenant segments — with prescribed actions")
    seg_counts=df["Tenant_Segment"].value_counts(); total_seg=len(df)
    seg_cfg={"High-Value Regular":("⭐","#EBF2FA","#1A3C5E","Offer 3-yr lock-in"),"Regular Payer":("✅","#E8F6F0","#1D8A5F","Revise rent at CPI+2%"),"Late Payer":("⏰","#FDF6E3","#D97706","Offer early-pay discount"),"Defaulter":("⚠️","#FCEAEA","#C0392B","Require 3-month advance")}
    cols=st.columns(4)
    for i,(seg,cnt) in enumerate(seg_counts.items()):
        icon,bg,fg,act=seg_cfg.get(seg,("•","#F7F4EF","#0F1923",""))
        pct=round(cnt/total_seg*100,1)
        with cols[i%4]: st.markdown(f'<div style="background:{bg};border-radius:12px;padding:14px;text-align:center"><div style="font-size:18px">{icon}</div><div style="font-size:11px;font-weight:600;color:{fg};text-transform:uppercase;letter-spacing:.06em;margin:4px 0">{seg}</div><div style="font-size:28px;font-weight:700;color:{fg}">{cnt}</div><div style="font-size:10px;color:{fg};opacity:.7">{pct}% of records</div><div style="font-size:10px;font-weight:600;color:{fg};margin-top:6px">{act} →</div></div>',unsafe_allow_html=True)
    st.markdown("<hr class='divider'>",unsafe_allow_html=True)
    c1,c2=st.columns(2)
    with c1: st.plotly_chart(quarterly_revenue(df),use_container_width=True)
    with c2: st.plotly_chart(automation_summary(df),use_container_width=True)

elif page=="revenue":
    st.markdown('<div class="section-header">💰 Revenue Analysis</div>',unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Owner-wise, location-wise and quarterly revenue breakdown</div>',unsafe_allow_html=True)
    render_kpis(df)
    st.markdown("<hr class='divider'>",unsafe_allow_html=True)
    c1,c2=st.columns(2)
    with c1: st.plotly_chart(owner_revenue_bar(df),use_container_width=True)
    with c2: st.plotly_chart(owner_revenue_pie(df),use_container_width=True)
    c1,c2=st.columns(2)
    with c1: st.plotly_chart(quarterly_revenue(df),use_container_width=True)
    with c2: st.plotly_chart(location_type_heatmap(df),use_container_width=True)
    st.markdown("#### Location performance")
    lt=df.groupby("Warehouse_Location").agg(Revenue=("Revenue_Collected_INR","sum"),Invoiced=("Total_Invoice_INR","sum"),Outstanding=("Balance_Due_INR","sum"),Avg_Rent=("Monthly_Rent_INR","mean"),Transactions=("Row_ID","count"),Paid_Rate=("Is_Paid_Binary","mean")).reset_index()
    lt["Paid_Rate"]=(lt["Paid_Rate"]*100).round(1); lt["Revenue"]=lt["Revenue"].apply(inr_fmt); lt["Invoiced"]=lt["Invoiced"].apply(inr_fmt); lt["Outstanding"]=lt["Outstanding"].apply(inr_fmt); lt["Avg_Rent"]=lt["Avg_Rent"].apply(lambda v:inr_fmt(round(v)))
    lt.columns=["Location","Revenue","Invoiced","Outstanding","Avg Rent","Txns","Paid%"]
    st.dataframe(lt,use_container_width=True,hide_index=True)

elif page=="diagnostic":
    st.markdown('<div class="section-header">🔍 Why did it happen?</div>',unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Correlation analysis, delay drivers, and risk patterns</div>',unsafe_allow_html=True)
    st.plotly_chart(correlation_heatmap(df),use_container_width=True)
    st.markdown('<div class="alert-info">📌 <strong>Key findings:</strong> WH Type vs Rent (strong positive), Risk Score vs Delay (strong positive), Tenure vs Delay (negative — longer tenants pay faster).</div>',unsafe_allow_html=True)
    c1,c2=st.columns(2)
    with c1: st.plotly_chart(rent_vs_size(df),use_container_width=True)
    with c2: st.plotly_chart(rent_vs_type(df),use_container_width=True)
    c1,c2=st.columns(2)
    with c1: st.plotly_chart(delay_by_industry(df),use_container_width=True)
    with c2: st.plotly_chart(delay_by_tenant_type(df),use_container_width=True)
    c1,c2=st.columns(2)
    with c1: st.plotly_chart(risk_score_histogram(df),use_container_width=True)
    with c2: st.plotly_chart(location_type_heatmap(df),use_container_width=True)
    st.markdown("#### Pearson correlations")
    from scipy.stats import pearsonr
    pairs=[("Monthly_Rent_INR","Size_Encoded","Rent vs WH Size"),("Monthly_Rent_INR","Type_Encoded","Rent vs WH Type"),("Delay_Days","TenantType_Encoded","Delay vs Tenant Type"),("Delay_Days","Customer_Tenure_Months","Delay vs Tenure"),("Risk_Score","Delay_Days","Risk Score vs Delay"),("Is_Paid_Binary","Customer_Tenure_Months","Paid vs Tenure"),("Monthly_Rent_INR","Lease_Duration_Months","Rent vs Lease Duration")]
    rows=[]
    for c1n,c2n,lbl in pairs:
        if c1n in df.columns and c2n in df.columns:
            sub=df[[c1n,c2n]].dropna()
            if len(sub)>5:
                r,p=pearsonr(sub[c1n],sub[c2n])
                rows.append({"Variables":lbl,"Pearson r":round(r,3),"p-value":round(p,4),"Strength":"Strong" if abs(r)>0.7 else("Moderate" if abs(r)>0.4 else "Weak"),"Direction":"Positive ↑" if r>0 else "Negative ↓","Significant":"Yes ✓" if p<0.05 else "No"})
    st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)

elif page=="risk":
    render_tenant_risk_tool()

elif page=="vacancy":
    render_vacancy_tracker()

elif page=="clustering":
    st.markdown('<div class="section-header">🎯 Tenant Segments</div>',unsafe_allow_html=True)
    st.markdown('<div class="section-sub">K-Means clustering — who are your tenants and what should you do with each group?</div>',unsafe_allow_html=True)
    k_val=st.slider("Number of segments",2,8,4)
    with st.spinner("Segmenting tenants..."): km,kms,dfc,profile,inertias,feat_c=train_clustering(df_main,k=k_val)
    # Persona cards
    st.markdown("#### Segment personas — with prescribed actions")
    pcfg={"High-Value Regular":("⭐","#EBF2FA","#1A3C5E","Offer 3-year lock-in"),"Regular Payer":("✅","#E8F6F0","#1D8A5F","Annual revision CPI+2%"),"Late Payer":("⏰","#FDF6E3","#D97706","1% discount if paid by 5th"),"At-Risk":("⚠️","#FCEAEA","#C0392B","Legal notice + 3-month advance")}
    cols=st.columns(min(k_val,4))
    for i,(cid,row) in enumerate(profile.iterrows()):
        lbl=row.get("Label",f"Cluster {cid}"); icon,bg,fg,act=pcfg.get(lbl,("•","#F7F4EF","#444",""))
        with cols[i%min(k_val,4)]: st.markdown(f'<div style="background:{bg};border-radius:14px;padding:16px;margin-bottom:10px"><div style="font-size:22px">{icon}</div><div style="font-size:13px;font-weight:600;color:{fg};margin:6px 0">{lbl}</div><div style="font-size:11px;color:{fg};opacity:.8">Count: {int(row.get("Count",0))}</div><div style="font-size:11px;color:{fg};opacity:.8">Avg rent: {inr_fmt(row.get("Avg_Rent",0))}</div><div style="font-size:11px;color:{fg};opacity:.8">Paid rate: {row.get("Paid_Rate",0)*100:.0f}%</div><div style="font-size:11px;color:{fg};margin-top:8px;font-weight:500">{act}</div></div>',unsafe_allow_html=True)
    c1,c2=st.columns(2)
    with c1: st.plotly_chart(elbow_chart(inertias),use_container_width=True)
    with c2: st.plotly_chart(cluster_profile_radar(profile),use_container_width=True)
    st.plotly_chart(cluster_scatter(dfc),use_container_width=True)
    with st.expander("View transactions by segment"):
        cc=st.selectbox("Filter",["All"]+sorted(dfc["Cluster_Label"].unique().tolist()))
        vd=dfc if cc=="All" else dfc[dfc["Cluster_Label"]==cc]
        sc=[c for c in["Tenant_Name","Warehouse_Location","Warehouse_Type","Monthly_Rent_INR","Delay_Days","Payment_Behavior","Risk_Score","Cluster_Label"] if c in dfc.columns]
        st.dataframe(vd[sc].head(50),use_container_width=True,hide_index=True)

elif page=="association":
    st.markdown('<div class="section-header">🔗 Patterns & Rules</div>',unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Which tenant types + warehouse types predict late payment? Plain-English findings.</div>',unsafe_allow_html=True)
    min_sup=st.slider("Sensitivity (min support)",0.01,0.20,0.05,0.01)
    rules=compute_association_rules(df_main,min_support=min_sup)
    if rules.empty: st.warning("No patterns found. Try lowering the slider.")
    else:
        st.markdown("#### What the data says")
        for _,rule in rules.head(6).iterrows():
            conf=rule["Confidence"]; sup=rule["Support"]; lift=rule["Lift"]; outcome=rule["Consequent"]
            bg="#E8F6F0" if outcome=="Paid" else "#FCEAEA"; fg="#1D8A5F" if outcome=="Paid" else "#A32D2D"; icon="✅" if outcome=="Paid" else "⚠️"
            st.markdown(f'<div style="background:{bg};border-radius:12px;padding:13px 16px;margin-bottom:8px"><div style="font-size:13px;font-weight:500;color:{fg}">{icon} When <strong>{rule["Antecedent"]}</strong> → <strong>{conf*100:.0f}%</strong> chance of <strong>{outcome}</strong></div><div style="font-size:11px;color:{fg};opacity:.8;margin-top:3px">{lift:.1f}× more likely than average · seen in {sup*100:.1f}% of records ({int(rule["Count"])} transactions)</div></div>',unsafe_allow_html=True)
        with st.expander("View all rules (technical table)"):
            dr=rules.copy(); dr["Support"]=(dr["Support"]*100).round(2).astype(str)+"%"; dr["Confidence"]=(dr["Confidence"]*100).round(1).astype(str)+"%"; dr["Lift"]=dr["Lift"].round(3)
            st.dataframe(dr[["Antecedent","Consequent","Support","Confidence","Lift","Count"]],use_container_width=True,hide_index=True)
        c1,c2=st.columns(2)
        with c1: st.plotly_chart(association_heatmap(rules),use_container_width=True)
        with c2: st.plotly_chart(association_bubble(rules),use_container_width=True)

elif page=="prospect":
    st.markdown('<div class="section-header">🚀 Score a Prospect</div>',unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Upload a list of prospects — get them ranked by payment reliability</div>',unsafe_allow_html=True)
    with st.spinner("Loading model..."): model_clf,scaler_clf,metrics_clf,feat_clf=train_classifier(df_main)
    st.markdown('<div class="alert-info">💡 Upload a CSV of prospects. The model scores each one and recommends whether to proceed.</div>',unsafe_allow_html=True)
    tpl=pd.DataFrame([{"Tenant_Name":"ABC Logistics","Tenant_Type":"Business","Industry_Type":"Logistics","Warehouse_Type":"Distribution Hub","Warehouse_Size":"Large","Monthly_Rent_INR":90000,"Customer_Tenure_Months":0,"Lease_Duration_Months":24}])
    buf=io.BytesIO(); tpl.to_csv(buf,index=False); buf.seek(0)
    st.download_button("⬇️ Download template CSV",data=buf,file_name="prospect_template.csv",mime="text/csv")
    uploaded=st.file_uploader("Upload prospect CSV",type=["csv"])
    use_m=st.checkbox("Enter one prospect manually")
    prospects_df=None
    if uploaded:
        try: prospects_df=pd.read_csv(uploaded); st.success(f"✅ {len(prospects_df)} prospects loaded.")
        except Exception as e: st.error(f"Error: {e}")
    elif use_m:
        with st.form("mp"):
            mc1,mc2,mc3=st.columns(3)
            with mc1: mn=st.text_input("Name","New Prospect Ltd"); mt=st.selectbox("Type",["Business","Individual"]); mi=st.text_input("Industry","Logistics")
            with mc2: mwt=st.selectbox("WH Type",["Distribution Hub","Cold Storage","Dry Warehouse","General"]); mws=st.selectbox("WH Size",["Large","Medium","Small"]); mr=st.number_input("Monthly Rent (₹)",5000,300000,50000,step=1000)
            with mc3: mtn=st.slider("Tenure (months)",0,60,0); ml=st.slider("Lease (months)",6,36,12)
            if st.form_submit_button("Score this prospect",type="primary"):
                prospects_df=pd.DataFrame([{"Tenant_Name":mn,"Tenant_Type":mt,"Industry_Type":mi,"Warehouse_Type":mwt,"Warehouse_Size":mws,"Monthly_Rent_INR":mr,"Customer_Tenure_Months":mtn,"Lease_Duration_Months":ml}])
    if prospects_df is not None and not prospects_df.empty:
        try:
            scored=score_prospects(prospects_df,model_clf,scaler_clf,feat_clf)
            if len(scored)<=3:
                gcols=st.columns(len(scored))
                for i,(_,row) in enumerate(scored.iterrows()):
                    with gcols[i]: st.plotly_chart(prospect_gauge(float(row["Pay_Probability"]),row.get("Tenant_Name",f"P{i+1}")),use_container_width=True)
            else: st.plotly_chart(prospect_risk_bar(scored),use_container_width=True)
            od=scored[["Tenant_Name","Tenant_Type","Warehouse_Type","Warehouse_Size","Monthly_Rent_INR","Pay_Probability","Risk_Tier","Recommended_Action"]].copy()
            od["Pay_Probability"]=(od["Pay_Probability"]*100).round(1).astype(str)+"%"
            st.dataframe(od,use_container_width=True,hide_index=True)
        except Exception as e: st.error(f"Scoring error: {e}")

elif page=="regression":
    st.markdown('<div class="section-header">📈 Rent & Delay Forecasts</div>',unsafe_allow_html=True)
    st.markdown('<div class="section-sub">What should I charge? How long will a late payer take to pay?</div>',unsafe_allow_html=True)
    with st.spinner("Training models..."): mr,metr,fr=train_rent_regressor(df_main); md,metd,fd=train_delay_regressor(df_main)
    st.markdown("#### Rent estimator — what should I charge?")
    with st.form("rf"):
        rc1,rc2,rc3=st.columns(3)
        with rc1: rs=st.selectbox("WH Size",["Small","Medium","Large"])
        with rc2: rt=st.selectbox("WH Type",["General","Dry Warehouse","Cold Storage","Distribution Hub"])
        with rc3: rl=st.slider("Lease (months)",6,36,12)
        rtn=st.slider("Tenant tenure (months)",1,60,12)
        if st.form_submit_button("Estimate fair rent",type="primary"):
            Xn=pd.DataFrame([{"Size_Encoded":SIZE_MAP.get(rs,2),"Type_Encoded":TYPE_MAP.get(rt,2),"Month_Num":25,"Customer_Tenure_Months":rtn,"Lease_Duration_Months":rl}])[fr]
            pred=mr.predict(Xn)[0]
            st.success(f"💰 Estimated fair rent: **{inr_fmt(max(5000,pred))}** / month")
    with st.expander("Technical model details"):
        r1,r2,r3=st.columns(3)
        with r1: st.metric("R²",f"{metr['r2']:.3f}")
        with r2: st.metric("MAE",inr_fmt(metr['mae']))
        with r3: st.metric("RMSE",inr_fmt(metr['rmse']))
        st.plotly_chart(regression_actual_vs_predicted(metr["y_test"],metr["y_pred"],"Rent — Actual vs Predicted"),use_container_width=True)

elif page=="pl":
    render_pl_export()

elif page=="settings":
    render_settings_page()

elif page=="admin":
    render_admin_page()
