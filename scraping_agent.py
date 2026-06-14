import os
import csv
import json
from datetime import datetime
from dotenv import load_dotenv
from tavily import TavilyClient
from openai import OpenAI


load_dotenv()

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


class AutomaticNewsSocialMediaScrapingAgent:
    """
    Automatic News & Social Media Scraping Agent

    New logic:
    1. Collect real public sources from the web first.
    2. Identify source type: News, Website, LinkedIn, X, Blog.
    3. Extract company name from each real source using OpenAI.
    4. Extract business signals.
    5. Save everything into CSV.

    Output:
    - automatic_scraping_results.csv
    """

    def __init__(self):
        if not TAVILY_API_KEY:
            raise ValueError("Missing TAVILY_API_KEY. Add it inside .env file.")

        if not OPENAI_API_KEY:
            raise ValueError("Missing OPENAI_API_KEY. Add it inside .env file.")

        self.tavily_client = TavilyClient(api_key=TAVILY_API_KEY)
        self.openai_client = OpenAI(api_key=OPENAI_API_KEY)

    def tavily_search(self, query, max_results=5):
        """
        Search public web results using Tavily.
        """
        try:
            response = self.tavily_client.search(
                query=query,
                search_depth="advanced",
                include_answer=False,
                include_raw_content=False,
                max_results=max_results
            )

            return response.get("results", [])

        except Exception:
            return []

    def ask_openai_json(self, prompt):
        """
        Ask OpenAI to return valid JSON.
        """
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                temperature=0,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a strict data extraction assistant. "
                            "Return valid JSON only. No markdown. No explanation."
                        )
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )

            text = response.choices[0].message.content.strip()
            text = text.replace("```json", "").replace("```", "").strip()

            return json.loads(text)

        except Exception:
            return {}

    def classify_source(self, url):
        """
        Classify source type based on URL.
        """
        url = url.lower()

        if "linkedin.com/posts" in url or "linkedin.com/feed/update" in url:
            return "LinkedIn Post"
        elif "linkedin.com/company" in url:
            return "LinkedIn Company Page"
        elif "linkedin.com" in url:
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

    def remove_duplicate_sources(self, sources):
        """
        Remove duplicate URLs.
        """
        unique_sources = []
        seen_urls = set()

        for item in sources:
            url = item.get("url", "")

            if url and url not in seen_urls:
                seen_urls.add(url)
                unique_sources.append(item)

        return unique_sources

    def collect_sources_from_web(self, max_sources=20):
        """
        Collect real public sources first.
        The company name will be extracted later from each source.
        """

        search_queries = [
            "Saudi Arabia AI companies news automation data analytics",
            "Saudi artificial intelligence startups news data automation",
            "Saudi digital transformation companies recent news",
            "Saudi technology companies AI automation announcements",
            "Saudi data analytics companies official website news",
            "Riyadh AI companies automation data analytics",
            "site:linkedin.com/company Saudi AI automation data company",
            "site:linkedin.com/posts Saudi AI automation company",
            "site:x.com Saudi AI automation company",
            "Saudi software companies AI automation digital transformation"
        ]

        collected_sources = []

        for query in search_queries:
            results = self.tavily_search(query, max_results=5)

            for result in results:
                url = result.get("url", "")

                collected_sources.append({
                    "search_query": query,
                    "source_type": self.classify_source(url),
                    "title": result.get("title", "No title"),
                    "url": url,
                    "content": result.get("content", "No content")
                })

        clean_sources = self.remove_duplicate_sources(collected_sources)

        return clean_sources[:max_sources]

    def extract_companies_from_sources(self, sources):
        """
        Extract company name from each real source after collecting the source.
        """

        indexed_sources = []

        for index, source in enumerate(sources, start=1):
            indexed_sources.append({
                "source_index": index,
                "source_type": source["source_type"],
                "title": source["title"],
                "url": source["url"],
                "content": source["content"]
            })

        prompt = f"""
You are a strict business data extraction assistant.

Your task:
Extract the company name from each real source.

Important:
- The company name must come from the source title, URL, or content.
- Do not invent company names.
- Do not return broad phrases like Saudi AI, Saudi Technology, Saudi Arabia, Artificial Intelligence, Digital Transformation.
- Do not return generic words like Top, Best, Verified, Company, Companies, Services, Solutions, Data, Automation.
- If no clear company name exists, set "is_valid_company" to false.
- Extract business signals only from the source content.

Allowed business signals:
- AI Interest
- Automation Need
- Data Analytics Need
- Digital Transformation
- Growth / Expansion
- Hiring / Talent
- Proposal / RFP Opportunity
- General Business Signal

Return JSON only in this exact format:
{{
  "items": [
    {{
      "source_index": 1,
      "company_name": "Company Name",
      "is_valid_company": true,
      "summary": "short summary based on the source",
      "detected_signals": ["AI Interest", "Data Analytics Need"],
      "lead_score": 80,
      "reason": "short reason"
    }}
  ]
}}

Sources:
{json.dumps(indexed_sources, ensure_ascii=False)}
"""

        data = self.ask_openai_json(prompt)
        items = data.get("items", [])

        analysis_by_index = {}

        for item in items:
            source_index = item.get("source_index")

            if source_index:
                analysis_by_index[source_index] = {
                    "company_name": item.get("company_name", "").strip(),
                    "is_valid_company": item.get("is_valid_company", False),
                    "summary": item.get("summary", "No summary generated."),
                    "detected_signals": item.get("detected_signals", ["General Business Signal"]),
                    "lead_score": item.get("lead_score", 0),
                    "reason": item.get("reason", "No reason generated.")
                }

        return analysis_by_index

    def save_results_to_csv(self, sources, analysis_by_index):
        """
        Save final source-based company extraction results to CSV.
        """

        file_name = "automatic_scraping_results.csv"

        with open(file_name, mode="w", newline="", encoding="utf-8-sig") as file:
            writer = csv.writer(file)

            writer.writerow([
                "company_name",
                "source_type",
                "search_query",
                "title",
                "url",
                "content",
                "openai_summary",
                "detected_signals",
                "lead_score",
                "reason",
                "scraped_at"
            ])

            for index, source in enumerate(sources, start=1):
                analysis = analysis_by_index.get(index)

                if not analysis:
                    continue

                if not analysis["is_valid_company"]:
                    continue

                company_name = analysis["company_name"]

                if not company_name:
                    continue

                writer.writerow([
                    company_name,
                    source["source_type"],
                    source["search_query"],
                    source["title"],
                    source["url"],
                    source["content"],
                    analysis["summary"],
                    ", ".join(analysis["detected_signals"]),
                    analysis["lead_score"],
                    analysis["reason"],
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                ])

        return file_name

    def run_auto(self, max_sources=20):
        """
        Full workflow:
        1. Collect real sources from the web.
        2. Extract company name from each source.
        3. Analyze signals.
        4. Save CSV.
        """

        sources = self.collect_sources_from_web(max_sources=max_sources)
        analysis_by_index = self.extract_companies_from_sources(sources)
        csv_file = self.save_results_to_csv(sources, analysis_by_index)

        return {
            "project_name": "Automatic News & Social Media Scraping Agent using Tavily + OpenAI",
            "total_sources": len(sources),
            "csv_file": csv_file
        }