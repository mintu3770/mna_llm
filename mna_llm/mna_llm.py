import os
import pandas as pd
import requests
from bs4 import BeautifulSoup
from googlesearch import search as google_search_unofficial  # ✅ fixed alias
import google.generativeai as genai
import time
import json
# from newspaper import Article  # Uncomment if you want to use newspaper3k for better article text

# --- Configuration ---
GOOGLE_API_KEY = "AIzaSyBRXKa_FljVBqfdtnq2tvvAGCVN5k5Nlo0"  # ⚠️ Replace with your actual API key

TARGET_TECHNOLOGIES = [
    "SAP", "Salesforce", "ServiceNow", "GCP", "AWS", "Azure",
    "Snowflake", "Databricks", "Workday", "BPO"
]

# Configure Gemini
try:
    genai.configure(api_key=GOOGLE_API_KEY)
    llm_model = genai.GenerativeModel('gemini-1.5-flash-latest')
except Exception as e:
    print(f"Error configuring Gemini API: {e}. Ensure API key is correct.")
    llm_model = None

# --- Helper Functions ---
def fetch_text_from_url(url: str) -> str | None:
    """Fetches and extracts cleaned text content from a URL."""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')

        for script_or_style in soup(["script", "style"]):
            script_or_style.decompose()

        text = ' '.join(t.strip() for t in soup.stripped_strings)
        return text
    except Exception as e:
        print(f"Error fetching or parsing {url}: {e}")
        return None

def search_online(query: str, num_results: int = 5) -> list:
    """Performs a web search using googlesearch-python (unofficial)."""
    urls = []
    print(f"🌐 Searching for: {query}")
    try:
        for j in google_search_unofficial(query, num_results=num_results, lang="en"):  # ✅ fixed usage
            urls.append(j)
            time.sleep(0.5)
    except Exception as e:
        print(f"Search failed for query '{query}': {e}")
    return urls

def analyze_content_with_llm(text_content: str, company_url: str) -> dict | None:
    """Analyzes text content using an LLM to extract M&A related information."""
    if not llm_model or not text_content:
        return None

    max_chars = 15000
    truncated_text = text_content[:max_chars]

    prompt = f"""
    Analyze the following text from the URL {company_url}. The company might be involved with one or more of these technologies: {', '.join(TARGET_TECHNOLOGIES)}.
    Extract information relevant to Mergers and Acquisitions (M&A).

    Identify the following and respond in JSON format with ONLY the JSON object:
    1.  "company_name": The official name of the company discussed.
    2.  "primary_technology_focus": Key technologies from the provided list that the company specializes in. (list of strings)
    3.  "country_city": Estimated primary location (Country, City). (string)
    4.  "employee_count_estimation": Any mention of employee numbers. (string or null)
    5.  "revenue_estimation": Any mention of revenue figures. (string or null)
    6.  "EBITDA_estimation": Any mention of EBITDA. (string or null)
    7.  "oem_certifications": Mention of official partnerships or certifications. (list of strings or null)
    8.  "foundation_year": If mentioned. (string or null)
    9.  "is_potential_seller_signals": Phrases suggesting the company might be selling. (list of strings or null)
    10. "is_potential_buyer_signals": Phrases suggesting buying interest. (list of strings or null)
    11. "past_acquisitions_mentioned": Any companies acquired. (list of strings or null)
    12. "summary_relevance_to_ma": A brief summary of M&A relevance. (string)

    Text to analyze:
    ---
    {truncated_text}
    ---

    Respond with ONLY the JSON object. Ensure proper JSON formatting.
    """

    try:
        print(f"🧠 Analyzing content from {company_url} with LLM...")
        response = llm_model.generate_content(prompt)

        json_response_text = response.text.strip()
        if json_response_text.startswith("```json"):
            json_response_text = json_response_text[7:]
        if json_response_text.endswith("```"):
            json_response_text = json_response_text[:-3]

        json_response_text = json_response_text.strip()
        parsed_json = json.loads(json_response_text)
        return parsed_json
    except json.JSONDecodeError as je:
        print(f"Error decoding LLM JSON response: {je}")
        print(f"Problematic LLM Response Text: {response.text}")
        return None
    except Exception as e:
        print(f"Error during LLM analysis for {company_url}: {e}")
        return None

# --- Main Execution Logic ---
def main():
    if not llm_model:
        print("LLM model not available. Please check API key and configuration.")
        return

    all_company_data = []
    processed_urls = set()

    search_queries = []
    for tech in TARGET_TECHNOLOGIES:
        search_queries.append(f"{tech} company news M&A")
        search_queries.append(f"companies acquiring {tech} services")
        search_queries.append(f"{tech} consulting firm for sale")
        search_queries.append(f"{tech} partner 'strategic investment'")

    for query in search_queries[:8]:
        urls = search_online(query, num_results=3)

        for url in urls:
            if url in processed_urls:
                print(f"⏭️ Skipping already processed URL: {url}")
                continue

            print(f"\n🔗 Processing URL: {url}")
            text = fetch_text_from_url(url)
            processed_urls.add(url)

            if text and len(text) > 200:
                llm_extracted_data = analyze_content_with_llm(text, url)
                if llm_extracted_data:
                    llm_extracted_data['source_url'] = url
                    llm_extracted_data['search_query_origin'] = query
                    all_company_data.append(llm_extracted_data)
                    print(f"✅ Data extracted for {llm_extracted_data.get('company_name', url)}")
                else:
                    print(f"⚠️ No data extracted by LLM for {url}")
            else:
                print(f"📄 Content too short or not fetched for {url}")
            time.sleep(1)

    if all_company_data:
        df = pd.DataFrame(all_company_data)
        output_filename = "ma_prospects_llm_output.csv"
        df.to_csv(output_filename, index=False, encoding='utf-8')
        print(f"\n💾 All data saved to {output_filename}")
        print(f"\nFound {len(df)} potential prospects.")
    else:
        print("\nNo M&A prospects identified in this run.")

if __name__ == "__main__":
    if GOOGLE_API_KEY == "YOUR_GEMINI_API_KEY":
        print("🚨 Please replace 'YOUR_GEMINI_API_KEY' with your actual Google Gemini API key in the script.")
    else:
        main()