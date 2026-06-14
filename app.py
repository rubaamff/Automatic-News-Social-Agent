from scraping_agent import AutomaticNewsSocialMediaScrapingAgent


def main():
    agent = AutomaticNewsSocialMediaScrapingAgent()

    output = agent.run_auto(max_sources=120)

    print("CSV file created successfully:")
    print(output["csv_file"])


if __name__ == "__main__":
    main()