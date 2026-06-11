import os
import csv
import json
from datetime import datetime
from dotenv import load_dotenv
from tavily import TavilyClient
from google import genai


load_dotenv()

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


class AutomaticNewsSocialMediaScrapingAgent:
    """
    Automatic News & Social Media Scraping Agent

    Tavily API:
    - Finds real companies from the web.
    - Collects public news, website, LinkedIn, X, and blog results.

    Gemini API:
    - Cleans company names.
    - Summarizes results.
    - Extracts business signals.
    - Gives lead score.

    Output:
    - CSV file.
    """

    def __init__(self):
        if not TAVILY_API_KEY:
            raise ValueError("Missing TAVILY_API_KEY. Add it inside .env file.")

        if not GEMINI_API_KEY:
            raise ValueError("Missing GEMINI_API_KEY. Add it inside .env file.")

        self.tavily_client = TavilyClient(api_key=TAVILY_API_KEY)
        self.gemini_client = genai.Client(api_key=GEMINI_API_KEY)

    def tavily_search(self, query, max_results=2):
        """
        Search public web results using Tavily API.
        Quiet version: no long terminal printing.
        """
        try:
            response = self.tavily_client.search(
                query=query,
                search_depth="basic",
                include_answer=False,
                include_raw_content=False,
                max_results=max_results
            )

            return response.get("results", [])

        except Exception:
            return []

    def ask_gemini_json(self, prompt):
        """
        Ask Gemini to return JSON.
        Quiet version: no long terminal printing.
        """
        try:
            response = self.gemini_client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )

            text = response.text.strip()
            text = text.replace("```json", "")
            text = text.replace("```", "")
            text = text.strip()

            return json.loads(text)

        except Exception:
            return {}

    def discover_companies_from_web(self, max_companies=3):
        """
        Discover real company names from web search results.
        Tavily collects raw results.
        Gemini extracts clean company names.
        """

        discovery_queries = [
            "Saudi AI companies official website",
            "Saudi technology companies AI data automation"
        ]

        raw_results = []

        for query in discovery_queries:
            results = self.tavily_search(query, max_results=2)

            for result in results:
                raw_results.append({
                    "title": result.get("title", ""),
                    "url": result.get("url", ""),
                    "content": result.get("content", "")
                })

        if not raw_results:
            return []

        prompt = f"""
You are a strict company-name extraction assistant.

Extract real company names from these web search results.

Rules:
- Return only real company names.
- Do not return generic words such as Top, Best, Verified, Location, Company, Companies, AI, Data, Automation.
- Do not return directory labels such as Clutch, GoodFirms, Premier Verified, Custom Software.
- Do not return duplicate names.
- Prefer companies related to Saudi Arabia, AI, data, automation, or digital transformation.
- Maximum number of companies: {max_companies}.

Return JSON only in this exact format:
{{
  "companies": ["Company 1", "Company 2", "Company 3"]
}}

Search results:
{json.dumps(raw_results, ensure_ascii=False)}
"""

        data = self.ask_gemini_json(prompt)
        companies = data.get("companies", [])

        clean_companies = []

        for company in companies:
            if isinstance(company, str):
                company = company.strip()

                if company and company not in clean_companies:
                    clean_companies.append(company)

        return clean_companies[:max_companies]

    def build_scraping_queries(self, company_name):
        """
        Build search queries for each discovered company.
        """

        return [
            {
                "search_type": "News",
                "query": f'"{company_name}" recent news AI data automation'
            },
            {
                "search_type": "Official Website",
                "query": f'"{company_name}" official website technology AI data'
            },
            {
                "search_type": "LinkedIn Public Results",
                "query": f'site:linkedin.com/company "{company_name}"'
            },
            {
                "search_type": "X / Twitter Public Results",
                "query": f'site:x.com "{company_name}"'
            },
            {
                "search_type": "Company Blog",
                "query": f'"{company_name}" blog announcements technology AI data'
            }
        ]

    def classify_source(self, url):
        """
        Classify source type based on URL.
        """

        url = url.lower()

        if "linkedin.com" in url:
            return "LinkedIn"
        elif "x.com" in url or "twitter.com" in url:
            return "X / Twitter"
        elif (
            "reuters.com" in url
            or "arabnews.com" in url
            or "zawya.com" in url
            or "news" in url
        ):
            return "News"
        elif "blog" in url:
            return "Blog"
        else:
            return "Website"

    def remove_duplicates(self, results):
        """
        Remove duplicated URLs.
        """

        unique_results = []
        seen_urls = set()

        for item in results:
            url = item.get("url", "")

            if url and url not in seen_urls:
                seen_urls.add(url)
                unique_results.append(item)

        return unique_results

    def scrape_company(self, company_name):
        """
        Scrape public web, news, LinkedIn, X, and blog results for one company.
        """

        all_results = []
        queries = self.build_scraping_queries(company_name)

        for query_item in queries:
            results = self.tavily_search(query_item["query"], max_results=2)

            for result in results:
                url = result.get("url", "")

                all_results.append({
                    "company_name": company_name,
                    "search_type": query_item["search_type"],
                    "source_type": self.classify_source(url),
                    "title": result.get("title", "No title"),
                    "url": url,
                    "content": result.get("content", "No content")
                })

        return self.remove_duplicates(all_results)

    def analyze_company_with_gemini(self, company_name, results):
        """
        Use Gemini to summarize the company activity and extract business signals.
        """

        if not results:
            return {
                "summary": "No public results found.",
                "detected_signals": ["General Business Signal"],
                "lead_score": 0,
                "reason": "Not enough public web results were collected."
            }

        prompt = f"""
You are a business intelligence assistant.

Analyze the following public web, news, and social media search results for this company:

Company: {company_name}

Tasks:
1. Write a short summary of the company's recent public activity.
2. Extract business signals from this list only:
   - AI Interest
   - Automation Need
   - Data Analytics Need
   - Digital Transformation
   - Growth / Expansion
   - Hiring / Talent
   - Proposal / RFP Opportunity
   - General Business Signal
3. Give a lead score from 0 to 100.
4. Give a short reason for the score.

Return JSON only in this exact format:
{{
  "summary": "short summary",
  "detected_signals": ["AI Interest"],
  "lead_score": 80,
  "reason": "short reason"
}}

Results:
{json.dumps(results, ensure_ascii=False)}
"""

        data = self.ask_gemini_json(prompt)

        return {
            "summary": data.get("summary", "No summary generated."),
            "detected_signals": data.get("detected_signals", ["General Business Signal"]),
            "lead_score": data.get("lead_score", 0),
            "reason": data.get("reason", "No reason generated.")
        }

    def save_all_results_to_csv(self, all_company_outputs):
        """
        Save all final results into one CSV file.
        """

        file_name = "automatic_scraping_results.csv"

        with open(file_name, mode="w", newline="", encoding="utf-8-sig") as file:
            writer = csv.writer(file)

            writer.writerow([
                "company_name",
                "search_type",
                "source_type",
                "title",
                "url",
                "content",
                "gemini_summary",
                "detected_signals",
                "lead_score",
                "reason",
                "scraped_at"
            ])

            for company_output in all_company_outputs:
                company_name = company_output["company_name"]
                results = company_output["results"]
                analysis = company_output["analysis"]

                summary = analysis["summary"]
                signals = ", ".join(analysis["detected_signals"])
                lead_score = analysis["lead_score"]
                reason = analysis["reason"]

                if results:
                    for item in results:
                        writer.writerow([
                            company_name,
                            item["search_type"],
                            item["source_type"],
                            item["title"],
                            item["url"],
                            item["content"],
                            summary,
                            signals,
                            lead_score,
                            reason,
                            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        ])
                else:
                    writer.writerow([
                        company_name,
                        "No Results",
                        "No Source",
                        "No Title",
                        "No URL",
                        "No Content",
                        summary,
                        signals,
                        lead_score,
                        reason,
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    ])

        return file_name

    def run_auto(self, max_companies=3):
        """
        Full workflow:
        1. Tavily discovers companies.
        2. Gemini cleans company names.
        3. Tavily scrapes each company.
        4. Gemini summarizes and extracts signals.
        5. CSV file is saved.
        """

        companies = self.discover_companies_from_web(max_companies=max_companies)

        all_company_outputs = []

        for company_name in companies:
            results = self.scrape_company(company_name)
            analysis = self.analyze_company_with_gemini(company_name, results)

            all_company_outputs.append({
                "company_name": company_name,
                "results": results,
                "analysis": analysis
            })

        csv_file = self.save_all_results_to_csv(all_company_outputs)

        return {
            "project_name": "Automatic News & Social Media Scraping Agent using Tavily + Gemini",
            "companies": companies,
            "total_companies": len(companies),
            "all_company_outputs": all_company_outputs,
            "csv_file": csv_file
        }