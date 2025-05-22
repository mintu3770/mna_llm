import streamlit as st
from core import run_mna_scouting

st.set_page_config(page_title="M&A Scouting Tool", layout="wide")

st.title("🔍 M&A Buyer/Seller Scouting Tool")

profile = st.radio("Are you looking for potential:", ["Buyers", "Sellers"])

with st.form("input_form"):
    st.subheader("Enter Search Parameters")
    industry = st.text_input("Industry (e.g., Fintech, Healthcare, etc.)")
    technology = st.text_input("Technology (e.g., AWS, Salesforce, SAP, etc.)")
    region = st.text_input("Region/Country (optional)")
    deal_size = st.text_input("Estimated Deal Size (optional)")
    additional_keywords = st.text_area("Other Keywords (comma-separated)", "")

    submitted = st.form_submit_button("Run Scouting")

if submitted:
    with st.spinner("Running deep search and LLM analysis..."):
        df = run_mna_scouting(
            profile=profile.lower(),
            industry=industry,
            technology=technology,
            region=region,
            deal_size=deal_size,
            additional_keywords=additional_keywords
        )
        if df is not None:
            st.success("✅ Scouting completed!")
            st.dataframe(df)
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download Results", csv, "mna_results.csv", "text/csv")
        else:
            st.error("No prospects found. Try different parameters.")