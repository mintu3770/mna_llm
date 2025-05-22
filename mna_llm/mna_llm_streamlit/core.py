import os
import pandas as pd
import requests
from bs4 import BeautifulSoup
from googlesearch import search as google_search_unofficial
import google.generativeai as genai
import time
import json

# --- Configuration ---
GOOGLE_API_KEY = "AIzaSyBRXKa_FljVBqfdtnq2tvvAGCVN5k5Nlo0"  # Replace with your actual API key

# Configure Gemini
try:
    genai.configure(api_key=GOOGLE_API_KEY)
    llm_model = genai.GenerativeModel('gemini-1.5-flash-latest')
except Exception as e:
    print(f"Error configuring Gemini API: {e}")
    llm_model = None

def fetch_text_from_url(url: str) -> str | None:
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        for script_or_style in soup(["script", "style"]):
            script_or_style.decompose()
        return ' '.join(t.strip() for t in soup.stripped_strings)
    except Exception as e:
        print(f"Error fetching or parsing {url}: {e}")
        return None

def search_online(query: str, num_results: int = 5) -> list:
    urls = []
    try:
        for j in google_search_unofficial(query, num_results=num_results, lang="en"):
            urls.append(j)
            time.sleep(0.5)
    except Exception as e:
        print(f"Search failed for query '{query}': {e}")
    return urls

def analyze_content_with_llm(text_content: str, company_url: str) -> dict | None:
    if not llm_model or not text_content:
        return None

    prompt = f"""
    Analyze the following text from {company_url} and extract M&A-related information.
    Respond ONLY in proper JSON format with the following fields:
    "company_name", "primary_technology_focus", "country_city", "employee_count_estimation",
    "revenue_estimation", "EBITDA_estimation", "oem_certifications", "foundation_year",
    "is_potential_seller_signals", "is_potential_buyer_signals", "past_acquisitions_mentioned",
    "summary_relevance_to_ma".
    ---
    {text_content[:15000]}
    """

    try:
        response = llm_model.generate_content(prompt)
        json_response_text = response.text.strip()
        if json_response_text.startswith("```json"):
            json_response_text = json_response_text[7:]
        if json_response_text.endswith("```"):
            json_response_text = json_response_text[:-3]
        return json.loads(json_response_text.strip())
    except Exception as e:
        print(f"LLM Error: {e}")
        return None

def build_search_queries(profile: str, industry: str, technology: str, region: str, deal_size: str, additional_keywords: str) -> list:
    queries = []
    keyword_chunk = f"{industry} {technology} {region} {deal_size} {additional_keywords}".strip()
    base_patterns = [
        f"{keyword_chunk} M&A news",
        f"{keyword_chunk} companies {'buying' if profile == 'buyers' else 'for sale'}",
        f"{keyword_chunk} acquisition targets",
        f"{keyword_chunk} strategic investment",
        f"{keyword_chunk} merger discussions"
    ]
    return [q for q in base_patterns if len(q.strip()) > 0]

def run_mna_scouting(
    profile: str,
    industry: str = "",
    technology: str = "",
    region: str = "",
    deal_size: str = "",
    additional_keywords: str = "",
    limit_results_per_query: int = 5
) -> pd.DataFrame:
    if not llm_model:
        raise RuntimeError("LLM model not available. Check your API key.")

    all_company_data = []
    processed_urls = set()

    search_queries = build_search_queries(
        profile=profile,
        industry=industry,
        technology=technology,
        region=region,
        deal_size=deal_size,
        additional_keywords=additional_keywords
    )

    for query in search_queries:
        urls = search_online(query, num_results=limit_results_per_query)
        for url in urls:
            if url in processed_urls:
                continue
            processed_urls.add(url)
            text = fetch_text_from_url(url)
            if text and len(text) > 200:
                data = analyze_content_with_llm(text, url)
                if data:
                    data['source_url'] = url
                    data['search_query_origin'] = query
                    data['llm_status'] = "success"
                    all_company_data.append(data)
                else:
                    fallback_data = {
                        "source_url": url,
                        "search_query_origin": query,
                        "fallback_summary": text[:500],
                        "llm_status": "fallback"
                    }
                    all_company_data.append(fallback_data)
            time.sleep(1)

    return pd.DataFrame(all_company_data)

if __name__ == "__main__":
    df = run_mna_scouting(
        profile="buyers",
        industry="Fintech",
        technology="AWS",
        region="India",
        deal_size="10M-100M",
        additional_keywords="Bangalore, acquisition"
    )
    if not df.empty:
        df.to_csv("ma_prospects_llm_output.csv", index=False)
        print(f"Saved {len(df)} records to ma_prospects_llm_output.csv")
    else:
        print("No prospects found.")