from scraping_agent import AutomaticNewsSocialMediaScrapingAgent


def main():
    print("AUTOMATIC NEWS & SOCIAL MEDIA SCRAPING AGENT")
    print("Using Tavily API + Gemini API")

    agent = AutomaticNewsSocialMediaScrapingAgent()

    output = agent.run_auto(max_companies=3)

    print("CSV file created successfully:")
    print(output["csv_file"])


if __name__ == "__main__":
    main()