"""
decisions.py — Action-first decision engine (Upgrades #1 + #4)
  • "What needs attention today" — prioritised action list
  • "Should I sign this tenant?" — single verdict ML tool
  • Vacancy & renewal tracker
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import streamlit as st
import pandas as pd
from datetime import datetime
from data_manager import get_vacancy_stats, get_expiring_leases, get_transaction_df
from utils import inr_fmt, train_classifier, encode_prospect, SIZE_MAP, TYPE_MAP, TTYPE_MAP
from charts import roc_curve_chart, confusion_matrix_chart, feature_importance_chart

# ─── ACTION LIST ──────────────────────────────────────────────────────────────
def render_action_list():
    st.markdown('<div class="section-header">📋 What needs your attention today?</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Everything urgent, sorted by priority — act on each one with a single tap</div>', unsafe_allow_html=True)
    df=get_transaction_df(); actions=[]
    # Overdue payments
    if not df.empty:
        for _,row in df[df["Payment_Status"]=="Not Paid"].sort_values("Delay_Days",ascending=False).iterrows():
            actions.append({"p":1,"icon":"💰","label":f"Collect from **{row['Tenant_Name']}**","detail":f"{inr_fmt(row['Balance_Due_INR'])} overdue · {int(row['Delay_Days'])}d late · {row.get('Warehouse_Location','')}","phone":str(row.get("Tenant_Phone","")),"type":"collect"})
    # Expiring leases
    for l in get_expiring_leases(60):
        p=1 if l["days_left"]<=14 else(2 if l["days_left"]<=30 else 3)
        actions.append({"p":p,"icon":"📄","label":f"Renew lease for **{l['tenant_name']}**","detail":f"{l['wh_name']} · expires {l['end_date']} · {l['days_left']}d left","phone":l.get("tenant_phone",""),"type":"renewal"})
    # Vacancies
    for v in [x for x in get_vacancy_stats() if x["is_vacant"]]:
        actions.append({"p":2,"icon":"🏭","label":f"Find tenant for **{v['name']}**","detail":f"{v['location']} · vacant {v.get('days_vacant',0)}d · {inr_fmt(v.get('revenue_lost',0))} lost","phone":"","type":"vacancy"})

    if not actions:
        st.markdown('<div class="alert-success">✅ <strong>All clear!</strong> No urgent actions today. Portfolio is healthy.</div>', unsafe_allow_html=True)
        return

    actions.sort(key=lambda x:x["p"])
    ur=sum(1 for a in actions if a["p"]==1); so=sum(1 for a in actions if a["p"]==2); pl=sum(1 for a in actions if a["p"]==3)
    c1,c2,c3=st.columns(3)
    for col,n,lbl,bg,fg in [(c1,ur,"Urgent — act today","#FCEAEA","#A32D2D"),(c2,so,"Due this week","#FDF6E3","#854F0B"),(c3,pl,"Plan ahead","#E8F6F0","#1D8A5F")]:
        with col: st.markdown(f'<div style="background:{bg};border-radius:12px;padding:14px;text-align:center"><div style="font-size:28px;font-weight:600;color:{fg}">{n}</div><div style="font-size:10px;color:{fg};text-transform:uppercase;letter-spacing:.06em">{lbl}</div></div>',unsafe_allow_html=True)
    st.markdown("<br>",unsafe_allow_html=True)
    clr={"collect":("#FCEAEA","#A32D2D"),"renewal":("#FDF6E3","#854F0B"),"vacancy":("#EBF2FA","#185FA5")}
    for a in actions:
        bg,fg=clr.get(a["type"],("#F7F4EF","#444"))
        wa=""
        if a["phone"]: wa=f'<a href="https://wa.me/91{a["phone"]}?text=WareHub+reminder" target="_blank" style="background:#25D366;color:white;font-size:11px;font-weight:600;padding:6px 14px;border-radius:20px;text-decoration:none;white-space:nowrap;margin-left:auto">📱 WhatsApp</a>'
        st.markdown(f'<div style="background:{bg};border-radius:12px;padding:13px 16px;margin-bottom:8px;display:flex;align-items:center;gap:12px"><div style="font-size:20px">{a["icon"]}</div><div style="flex:1"><div style="font-size:13px;font-weight:500;color:{fg}">{a["label"]}</div><div style="font-size:11px;color:{fg};opacity:.8;margin-top:2px">{a["detail"]}</div></div>{wa}</div>',unsafe_allow_html=True)

# ─── TENANT RISK TOOL ─────────────────────────────────────────────────────────
def render_tenant_risk_tool():
    st.markdown('<div class="section-header">🔮 Should I sign this tenant?</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Enter the prospect\'s details — get a plain-English risk verdict in seconds</div>', unsafe_allow_html=True)
    df=get_transaction_df()
    with st.spinner("Loading risk model..."): model,scaler,metrics,feat=train_classifier(df)
    acc=round(metrics["accuracy"]*100,1)
    st.markdown(f'<div class="alert-info">🤖 Model trained on your rental history. Correctly identifies <strong>{acc}%</strong> of payment outcomes.</div>',unsafe_allow_html=True)
    with st.form("risk_form"):
        c1,c2,c3=st.columns(3)
        with c1: name=st.text_input("Prospect name","New Prospect Ltd"); ttype=st.selectbox("Type",["Business","Individual"]); rent=st.number_input("Monthly rent (₹)",5000,500000,45000,step=1000)
        with c2: size=st.selectbox("Warehouse size",["Small","Medium","Large"]); wtype=st.selectbox("Warehouse type",["General","Dry Warehouse","Cold Storage","Distribution Hub"]); lease=st.slider("Lease (months)",6,36,12)
        with c3: tenure=st.slider("Prior tenancy (months)",0,60,0); risk_sc=st.slider("Risk score estimate",0,100,30)
        sub=st.form_submit_button("Get Verdict",type="primary",use_container_width=True)
    if sub:
        enc=encode_prospect({"Monthly_Rent_INR":rent,"Warehouse_Size":size,"Warehouse_Type":wtype,"Tenant_Type":ttype,"Customer_Tenure_Months":tenure,"Lease_Duration_Months":lease,"Risk_Score":risk_sc},month_num=25)
        row=pd.DataFrame([enc])[feat]; row_s=scaler.transform(row)
        prob=model.predict_proba(row_s)[0][1]
        if prob>=0.75: bg,fg,icon,v,act="#E8F6F0","#1D8A5F","✅","Low risk — proceed with standard contract","Standard 1-month security deposit."
        elif prob>=0.5: bg,fg,icon,v,act="#FDF6E3","#854F0B","⚠️","Medium risk — proceed with caution","2-month security deposit. Add penalty clause."
        else: bg,fg,icon,v,act="#FCEAEA","#A32D2D","🚫","High risk — do not proceed without protection","Require 3-month advance OR decline."
        st.markdown(f'<div style="background:{bg};border-radius:16px;padding:24px;text-align:center"><div style="font-size:44px">{icon}</div><div style="font-size:20px;font-weight:600;color:{fg};margin:8px 0">{name}</div><div style="font-size:15px;color:{fg}">{v}</div><div style="font-size:26px;font-weight:700;color:{fg};margin-top:8px">{prob*100:.0f}% chance of paying on time</div></div>',unsafe_allow_html=True)
        st.markdown(f'<div style="background:#F7F4EF;border-radius:10px;padding:12px 16px;margin-top:10px;font-size:13px;color:#444"><strong>Recommended action:</strong> {act}</div>',unsafe_allow_html=True)
        with st.expander("View technical details (for academic review)"):
            c1,c2=st.columns(2)
            with c1: st.plotly_chart(roc_curve_chart(metrics["fpr"],metrics["tpr"],metrics["roc_auc"]),use_container_width=True)
            with c2: st.plotly_chart(confusion_matrix_chart(metrics["cm"]),use_container_width=True)
            st.plotly_chart(feature_importance_chart(metrics["feature_names"],metrics["feature_importance"]),use_container_width=True)

# ─── VACANCY & RENEWALS ───────────────────────────────────────────────────────
def render_vacancy_tracker():
    st.markdown('<div class="section-header">🏭 Vacancy & Renewal Tracker</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Empty warehouses and expiring leases — your two biggest revenue risks</div>', unsafe_allow_html=True)
    vacancies=get_vacancy_stats(); vacant=[v for v in vacancies if v["is_vacant"]]; occupied=[v for v in vacancies if not v["is_vacant"]]
    c1,c2,c3=st.columns(3)
    with c1: st.markdown(f'<div style="background:#E8F6F0;border-radius:12px;padding:14px;text-align:center"><div style="font-size:28px;font-weight:600;color:#1D8A5F">{len(occupied)}</div><div style="font-size:10px;color:#1D8A5F;text-transform:uppercase">Occupied</div></div>',unsafe_allow_html=True)
    with c2: st.markdown(f'<div style="background:#FCEAEA;border-radius:12px;padding:14px;text-align:center"><div style="font-size:28px;font-weight:600;color:#A32D2D">{len(vacant)}</div><div style="font-size:10px;color:#A32D2D;text-transform:uppercase">Vacant</div></div>',unsafe_allow_html=True)
    with c3:
        lost=sum(v.get("revenue_lost",0) for v in vacant)
        st.markdown(f'<div style="background:#FDF6E3;border-radius:12px;padding:14px;text-align:center"><div style="font-size:22px;font-weight:600;color:#854F0B">{inr_fmt(lost)}</div><div style="font-size:10px;color:#854F0B;text-transform:uppercase">Revenue lost to vacancy</div></div>',unsafe_allow_html=True)
    st.markdown("<br>",unsafe_allow_html=True)
    cols=st.columns(min(len(vacancies),3))
    for i,v in enumerate(vacancies):
        with cols[i%min(len(vacancies),3)]:
            iv=v["is_vacant"]; bg="#FCEAEA" if iv else "#E8F6F0"; fg="#A32D2D" if iv else "#1D8A5F"
            status=f"🔴 Vacant · {v['days_vacant']}d" if iv else "🟢 Occupied"
            st.markdown(f'<div style="background:{bg};border-radius:12px;padding:14px;margin-bottom:10px"><div style="font-size:13px;font-weight:500;color:{fg}">{v["name"]}</div><div style="font-size:11px;color:{fg};opacity:.8">{v["location"]} · {v["type"]}</div><div style="font-size:11px;color:{fg};margin-top:6px">{status}</div>'+( f'<div style="font-size:11px;color:{fg};margin-top:2px">Lost: {inr_fmt(v["revenue_lost"])}</div>' if iv and v.get("revenue_lost",0)>0 else "")+f'</div>',unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("#### Leases expiring within 60 days")
    expiring=get_expiring_leases(60)
    if not expiring: st.markdown('<div class="alert-success">✅ No leases expiring in the next 60 days.</div>',unsafe_allow_html=True)
    else:
        for l in expiring:
            bg="#FCEAEA" if l["days_left"]<=14 else("#FDF6E3" if l["days_left"]<=30 else "#E8F6F0")
            fg="#A32D2D" if l["days_left"]<=14 else("#854F0B" if l["days_left"]<=30 else "#1D8A5F")
            lbl="Act now" if l["days_left"]<=14 else("Soon" if l["days_left"]<=30 else "Plan ahead")
            phone=l.get("tenant_phone","")
            wa=f'<a href="https://wa.me/91{phone}?text=Hello+{l["tenant_name"].replace(" ","+")}%2C+let+us+discuss+lease+renewal." target="_blank" style="background:#25D366;color:white;font-size:10px;font-weight:600;padding:5px 12px;border-radius:20px;text-decoration:none;white-space:nowrap">📱 WhatsApp</a>' if phone else ""
            st.markdown(f'<div style="background:{bg};border-radius:12px;padding:13px 16px;margin-bottom:8px;display:flex;align-items:center;gap:12px"><div style="flex:1"><div style="font-size:13px;font-weight:500;color:{fg}">{l["tenant_name"]} — {l["wh_name"]}</div><div style="font-size:11px;color:{fg};opacity:.8">Expires {l["end_date"]} · {l["days_left"]}d left · ₹{int(l["monthly_rent"]):,}/mo</div></div><div style="font-size:10px;font-weight:600;padding:3px 10px;border-radius:10px;background:{fg};color:white">{lbl}</div>{wa}</div>',unsafe_allow_html=True)
