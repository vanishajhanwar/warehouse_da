"""
pl_export.py — Annual P&L Excel export (Upgrade #8)
4-sheet workbook: Summary, Monthly, Tenant Ledger, GST Statement.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import streamlit as st
import pandas as pd
import io
from datetime import datetime
from data_manager import get_business, get_transaction_df
from utils import inr_fmt

def _n(v):
    try: return round(float(v),2)
    except: return 0.0

def generate_pl_excel(df,year):
    try: import openpyxl; from openpyxl.styles import Font,PatternFill,Alignment,Border,Side; from openpyxl.utils import get_column_letter
    except: st.error("openpyxl not installed."); return b""
    biz=get_business(); ydf=df[df["Month"].dt.year==year].copy()
    if ydf.empty: return b""
    wb=openpyxl.Workbook()
    MID="FF1A3C5E"; HF=Font(name="Calibri",bold=True,color="FFFFFF",size=11); TF=Font(name="Calibri",bold=True,color="FF0F1923",size=14); BF=Font(name="Calibri",size=10); BF2=Font(name="Calibri",bold=True,size=10)
    HFill=PatternFill("solid",fgColor="1A3C5E"); AFill=PatternFill("solid",fgColor="F7F4EF"); INR='#,##0.00'
    def hrow(ws,row,vals,widths=None):
        for c,v in enumerate(vals,1):
            cell=ws.cell(row=row,column=c,value=v); cell.font=HF; cell.fill=HFill; cell.alignment=Alignment(horizontal="center",vertical="center")
            if widths and c<=len(widths): ws.column_dimensions[get_column_letter(c)].width=widths[c-1]
    def drow(ws,row,vals,ncols=None,alt=False):
        fill=AFill if alt else None
        for c,v in enumerate(vals,1):
            cell=ws.cell(row=row,column=c,value=v); cell.font=BF; cell.alignment=Alignment(horizontal="right" if(ncols and c in ncols) else "left",vertical="center")
            if ncols and c in ncols and isinstance(v,(int,float)): cell.number_format=INR
            if fill: cell.fill=fill

    # Sheet 1 — Summary
    ws1=wb.active; ws1.title="P&L Summary"
    ws1.merge_cells("A1:G1"); t=ws1["A1"]; t.value=f"{biz.get('business_name','WareHub')} — Annual P&L {year}"; t.font=TF
    ws1.merge_cells("A2:G2"); s=ws1["A2"]; s.value=f"Generated {datetime.today().strftime('%d %b %Y')} | GSTIN: {biz.get('gstin','')} | PAN: {biz.get('pan','')}"; s.font=Font(name="Calibri",color="FF64748B",size=10)
    ws1.append([])
    ti=ydf["Total_Invoice_INR"].sum(); tc=ydf["Revenue_Collected_INR"].sum(); to=ydf["Balance_Due_INR"].sum(); tg=ydf["GST_18pct_INR"].sum(); tr=ydf["Monthly_Rent_INR"].sum(); pr=ydf["Is_Paid_Binary"].mean()*100
    ws1.append(["Key Metrics",""]); ws1[f"A{ws1.max_row}"].font=Font(name="Calibri",bold=True,color="FF1A3C5E",size=11)
    for lbl,val in[("Total Invoiced (incl GST)",_n(ti)),("Revenue Collected",_n(tc)),("Outstanding",_n(to)),("Base Rent",_n(tr)),("GST Collected (18%)",_n(tg)),("Payment Rate",f"{pr:.1f}%")]:
        ws1.append([lbl,val]); r=ws1.max_row; ws1.cell(r,1).font=BF; ws1.cell(r,2).font=BF2
        if isinstance(val,float): ws1.cell(r,2).number_format=INR; ws1.cell(r,2).alignment=Alignment(horizontal="right")
    ws1.append([])
    ws1.append(["Owner-wise Breakdown"]); ws1[f"A{ws1.max_row}"].font=Font(name="Calibri",bold=True,color="FF1A3C5E",size=11); ws1.append([])
    ot=ydf.groupby("Owner_Name").agg(Inv=("Total_Invoice_INR","sum"),Col=("Revenue_Collected_INR","sum"),Out=("Balance_Due_INR","sum"),GST=("GST_18pct_INR","sum"),Txn=("Row_ID","count"),PR=("Is_Paid_Binary","mean")).reset_index()
    hrow(ws1,ws1.max_row+1,["Owner","Invoiced","Collected","Outstanding","GST","Transactions","Pay Rate"],widths=[22,16,16,16,14,14,12])
    for i,(_,row) in enumerate(ot.iterrows()): drow(ws1,ws1.max_row+1,[row["Owner_Name"],_n(row["Inv"]),_n(row["Col"]),_n(row["Out"]),_n(row["GST"]),int(row["Txn"]),f"{row['PR']*100:.1f}%"],ncols={2,3,4,5},alt=i%2==1)
    ws1.column_dimensions["A"].width=28; ws1.column_dimensions["B"].width=18

    # Sheet 2 — Monthly
    ws2=wb.create_sheet("Monthly Breakdown"); ws2.merge_cells("A1:F1"); ws2["A1"].value=f"Monthly Revenue — {year}"; ws2["A1"].font=TF
    mt=ydf.groupby("Month_Name").agg(Inv=("Total_Invoice_INR","sum"),Col=("Revenue_Collected_INR","sum"),Out=("Balance_Due_INR","sum"),GST=("GST_18pct_INR","sum"),PC=("Is_Paid_Binary","sum"),TC=("Row_ID","count")).reset_index()
    mo=[f"{m}-{year}" for m in["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]]
    mt["_s"]=mt["Month_Name"].map(lambda x:mo.index(x) if x in mo else 99); mt=mt.sort_values("_s").drop(columns=["_s"])
    ws2.append([]); hrow(ws2,3,["Month","Invoiced","Collected","Outstanding","GST","Pay Rate"],widths=[14,16,16,16,14,12])
    for i,(_,row) in enumerate(mt.iterrows()): drow(ws2,ws2.max_row+1,[row["Month_Name"],_n(row["Inv"]),_n(row["Col"]),_n(row["Out"]),_n(row["GST"]),f"{row['PC']/max(row['TC'],1)*100:.1f}%"],ncols={2,3,4,5},alt=i%2==1)
    lr=ws2.max_row+1; ws2.append(["TOTAL",_n(mt["Inv"].sum()),_n(mt["Col"].sum()),_n(mt["Out"].sum()),_n(mt["GST"].sum()),""])
    for c in range(1,7):
        cell=ws2.cell(lr,c); cell.font=BF2
        if c in{2,3,4,5}: cell.number_format=INR; cell.alignment=Alignment(horizontal="right")

    # Sheet 3 — Tenant Ledger
    ws3=wb.create_sheet("Tenant Ledger"); ws3.merge_cells("A1:I1"); ws3["A1"].value=f"Tenant Ledger — {year}"; ws3["A1"].font=TF
    ws3.append([])
    hrow(ws3,3,["Tenant","Warehouse","Month","Rent","GST","Total","Status","Paid","Balance"],widths=[22,18,12,12,10,14,12,14,12])
    for i,(_,row) in enumerate(ydf.sort_values(["Tenant_Name","Month"]).iterrows()): drow(ws3,ws3.max_row+1,[row["Tenant_Name"],row.get("Warehouse_Location",""),row["Month_Name"],_n(row["Monthly_Rent_INR"]),_n(row["GST_18pct_INR"]),_n(row["Total_Invoice_INR"]),row["Payment_Status"],_n(row["Amount_Paid_INR"]),_n(row["Balance_Due_INR"])],ncols={4,5,6,8,9},alt=i%2==1)

    # Sheet 4 — GST
    ws4=wb.create_sheet("GST Statement"); ws4.merge_cells("A1:D1"); ws4["A1"].value=f"GST Collected — {year}"; ws4["A1"].font=TF
    ws4.merge_cells("A2:D2"); ws4["A2"].value=f"GSTIN: {biz.get('gstin','')} | Rate: {biz.get('gst_rate',18)}%"; ws4["A2"].font=Font(name="Calibri",color="FF64748B",size=10)
    gt=ydf.groupby("Quarter").agg(BR=("Monthly_Rent_INR","sum"),GST=("GST_18pct_INR","sum"),Txn=("Row_ID","count")).reset_index()
    ws4.append([]); ws4.append([])
    hrow(ws4,5,["Quarter","Taxable Value","GST @ 18%","Transactions"],widths=[14,22,18,16])
    for i,(_,row) in enumerate(gt.iterrows()): drow(ws4,ws4.max_row+1,[row["Quarter"],_n(row["BR"]),_n(row["GST"]),int(row["Txn"])],ncols={2,3},alt=i%2==1)
    lr=ws4.max_row+1; ws4.append(["TOTAL",_n(gt["BR"].sum()),_n(gt["GST"].sum()),int(gt["Txn"].sum())])
    for c in range(1,5):
        cell=ws4.cell(lr,c); cell.font=BF2
        if c in{2,3}: cell.number_format=INR; cell.alignment=Alignment(horizontal="right")
    for ws in[ws1,ws2,ws3,ws4]: ws.freeze_panes="A4"
    buf=io.BytesIO(); wb.save(buf); buf.seek(0); return buf.read()

def render_pl_export():
    st.markdown('<div class="section-header">📊 Annual P&L Report</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">One-click Excel export for your CA — revenue, GST, tenant ledger, quarterly breakdown</div>', unsafe_allow_html=True)
    df=get_transaction_df(); biz=get_business()
    if df.empty: st.info("No data available."); return
    years=sorted(df["Month"].dt.year.unique(),reverse=True)
    sel=st.selectbox("Financial year",years); ydf=df[df["Month"].dt.year==sel]
    ti=ydf["Total_Invoice_INR"].sum(); tc=ydf["Revenue_Collected_INR"].sum(); tg=ydf["GST_18pct_INR"].sum(); to=ydf["Balance_Due_INR"].sum(); pr=ydf["Is_Paid_Binary"].mean()*100
    st.markdown("<br>",unsafe_allow_html=True)
    c1,c2,c3,c4,c5=st.columns(5)
    for col,val,lbl,bg,fg in[(c1,inr_fmt(ti),"Total invoiced","#EBF2FA","#185FA5"),(c2,inr_fmt(tc),"Collected","#E8F6F0","#1D8A5F"),(c3,inr_fmt(tg),"GST collected","#FDF6E3","#854F0B"),(c4,inr_fmt(to),"Outstanding","#FCEAEA","#A32D2D"),(c5,f"{pr:.1f}%","Payment rate","#F7F4EF","#444")]:
        with col: st.markdown(f'<div style="background:{bg};border-radius:10px;padding:11px;text-align:center"><div style="font-size:16px;font-weight:600;color:{fg}">{val}</div><div style="font-size:10px;color:{fg};text-transform:uppercase;letter-spacing:.06em">{lbl}</div></div>',unsafe_allow_html=True)
    st.markdown("<br>",unsafe_allow_html=True)
    st.markdown('<div class="alert-info">📋 The Excel file includes 4 sheets: <strong>P&L Summary</strong>, <strong>Monthly Breakdown</strong>, <strong>Tenant Ledger</strong>, and <strong>GST Statement</strong> — ready for your CA.</div>',unsafe_allow_html=True)
    if st.button("⬇️ Generate & Download Excel",type="primary"):
        with st.spinner(f"Building P&L report for {sel}..."):
            xl=generate_pl_excel(df,sel)
        if xl:
            fname=f"WareHub_PL_{sel}.xlsx"
            st.download_button(f"📥 Download {fname}",data=xl,file_name=fname,mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    st.markdown("#### Quarterly preview")
    qt=ydf.groupby("Quarter").agg(Inv=("Total_Invoice_INR","sum"),Col=("Revenue_Collected_INR","sum"),GST=("GST_18pct_INR","sum"),Out=("Balance_Due_INR","sum")).reset_index()
    for col in["Inv","Col","GST","Out"]: qt[col]=qt[col].apply(inr_fmt)
    qt.columns=["Quarter","Invoiced","Collected","GST (18%)","Outstanding"]
    st.dataframe(qt,use_container_width=True,hide_index=True)
