"""
rent_calendar.py — Visual monthly rent grid (Upgrade #3)
Replaces 48-column data table with colour-coded calendar.
Green=paid, Red=overdue, Amber=pending, Grey=no tenancy.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import streamlit as st
import pandas as pd
from data_manager import get_warehouses, get_transaction_df, ensure_payments_for_month, mark_payment_paid
from utils import inr_fmt

def render_rent_calendar():
    st.markdown('<div class="section-header">📅 Rent Calendar</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Every warehouse, every month — see who\'s paid and who hasn\'t at a glance</div>', unsafe_allow_html=True)
    df=get_transaction_df(); whs=get_warehouses()
    if df.empty: st.info("No data yet. Add rentals in Admin."); return

    avail=sorted(df["Month"].dt.to_period("M").unique(),reverse=True)
    c1,c2=st.columns([2,3])
    with c1: sel=st.selectbox("Month",[ str(m) for m in avail],index=0)
    with c2: view=st.radio("View",["Grid","Detail list"],horizontal=True)
    per=pd.Period(sel,"M"); month_df=df[df["Month"].dt.to_period("M")==per].copy()

    paid_r=month_df[month_df["Payment_Status"]=="Paid"]; unpaid_r=month_df[month_df["Payment_Status"]=="Not Paid"]
    c1,c2,c3,c4=st.columns(4)
    for col,val,lbl,bg,fg in[(c1,f"{len(month_df)}","Invoices","#EBF2FA","#185FA5"),(c2,inr_fmt(paid_r["Revenue_Collected_INR"].sum()),"Collected","#E8F6F0","#1D8A5F"),(c3,inr_fmt(unpaid_r["Balance_Due_INR"].sum()),"Outstanding","#FCEAEA","#A32D2D"),(c4,f"{len(paid_r)}/{len(month_df)}","Paid","#FDF6E3","#854F0B")]:
        with col: st.markdown(f'<div style="background:{bg};border-radius:10px;padding:11px 12px;text-align:center"><div style="font-size:18px;font-weight:600;color:{fg}">{val}</div><div style="font-size:10px;color:{fg};text-transform:uppercase;letter-spacing:.06em">{lbl}</div></div>',unsafe_allow_html=True)
    st.markdown("<br>",unsafe_allow_html=True)

    if view=="Grid":
        wh_data={row["WH_ID"]:row for _,row in month_df.iterrows()}
        cols=st.columns(min(len(whs),3))
        for i,wh in enumerate(whs):
            with cols[i%min(len(whs),3)]:
                row=wh_data.get(wh["id"])
                if row is None:
                    st.markdown(f'<div style="background:#F7F4EF;border-radius:12px;padding:16px;margin-bottom:12px;border:1px dashed #D3D1C7;text-align:center"><div style="font-size:12px;font-weight:500;color:#888">{wh["name"]}</div><div style="font-size:10px;color:#B4B2A9;margin-top:4px">{wh["location"]}</div><div style="font-size:10px;color:#B4B2A9;margin-top:6px">No tenancy</div></div>',unsafe_allow_html=True)
                    continue
                paid=row["Payment_Status"]=="Paid"; delay=int(row["Delay_Days"]); bal=row["Balance_Due_INR"]
                if paid: bg,fg,status="#E8F6F0","#1D8A5F",f"✅ Paid"+(" ("+str(delay)+"d late)" if delay>0 else "")
                elif delay>0: bg,fg,status="#FCEAEA","#A32D2D",f"🔴 {delay}d overdue"
                else: bg,fg,status="#FDF6E3","#854F0B","🟡 Pending"
                phone=str(row.get("Tenant_Phone",""))
                wa=f'<a href="https://wa.me/91{phone}?text=Dear+{str(row.get("Tenant_Name","")).replace(" ","+")}%2C+your+rent+of+{inr_fmt(bal)}+is+pending." target="_blank" style="display:inline-block;margin-top:8px;background:#25D366;color:white;font-size:10px;font-weight:600;padding:4px 10px;border-radius:20px;text-decoration:none">📱 Send reminder</a>' if phone and not paid else ""
                st.markdown(f'<div style="background:{bg};border-radius:12px;padding:16px;margin-bottom:12px"><div style="font-size:13px;font-weight:500;color:{fg}">{wh["name"]}</div><div style="font-size:11px;color:{fg};opacity:.8">{row.get("Tenant_Name","—")}</div><div style="font-size:18px;font-weight:600;color:{fg};margin:6px 0">{inr_fmt(row["Total_Invoice_INR"])}</div><div style="font-size:11px;color:{fg}">{status}</div>{wa}</div>',unsafe_allow_html=True)
    else:
        y,m=per.year,per.month; payments=ensure_payments_for_month(y,m); pay_map={p["tenant_id"]:p for p in payments}
        for _,row in month_df.sort_values("Balance_Due_INR",ascending=False).iterrows():
            paid=row["Payment_Status"]=="Paid"; delay=int(row["Delay_Days"]); total=row["Total_Invoice_INR"]; tnt_id=row["Tenant_ID"]
            icon="✅" if paid else("🔴" if delay>0 else "🟡")
            with st.expander(f"{icon} {row['Tenant_Name']} — {row['Warehouse_Location']} — {inr_fmt(total)}"):
                c1,c2=st.columns([2,1])
                with c1:
                    st.markdown(f"**Warehouse:** {row.get('WH_ID','')} · {row.get('Warehouse_Location','')}")
                    st.markdown(f"**Invoice:** {inr_fmt(row.get('Monthly_Rent_INR',0))} + {inr_fmt(row.get('GST_18pct_INR',0))} GST = **{inr_fmt(total)}**")
                    st.markdown(f"**Status:** {row['Payment_Status']}{' · '+str(delay)+'d delay' if delay>0 else ''}")
                    st.markdown(f"**Contact:** {row.get('Contact_Person','')} · +91-{row.get('Tenant_Phone','')}")
                with c2:
                    pr=pay_map.get(tnt_id,{}); pid=pr.get("id","")
                    if not paid and pid:
                        if st.button("✅ Mark paid",key=f"p_{pid}"): mark_payment_paid(pid,total); st.success("Marked as paid!"); st.rerun()
                    phone=str(row.get("Tenant_Phone",""))
                    if phone and not paid:
                        msg=f"Dear {row.get('Tenant_Name','')}, rent of {inr_fmt(row['Balance_Due_INR'])} for {row.get('Month_Name','')} is pending."
                        import urllib.parse
                        st.markdown(f'<a href="https://wa.me/91{phone}?text={urllib.parse.quote(msg)}" target="_blank" style="display:block;text-align:center;background:#25D366;color:white;font-size:12px;font-weight:600;padding:8px;border-radius:8px;text-decoration:none;margin-top:6px">📱 WhatsApp reminder</a>',unsafe_allow_html=True)
