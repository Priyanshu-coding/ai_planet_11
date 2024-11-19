import os
import requests
import streamlit as st
from bs4 import BeautifulSoup
import cohere
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Access API keys from environment variables
HUGGINGFACE_API_KEY = os.getenv('HUGGINGFACE_API_KEY')
COHERE_API_KEY = os.getenv('COHERE_API_KEY')

# Ensure the keys are not missing
if not HUGGINGFACE_API_KEY or not COHERE_API_KEY:
    raise ValueError("One or more API keys are missing. Check your .env file.")

# Initialize Cohere client
co = cohere.Client(COHERE_API_KEY)


class ResourceCollectorAgent:
    def save_to_markdown(self, industry, links):
        """Save the resource links to a markdown file."""
        filename = f"{industry}_resources.md"
        with open(filename, "w") as f:
            f.write(f"# Resource Links for {industry}\n\n")
            for link in links:
                f.write(f"- {link}\n")
        print(f"Resource links saved to {filename}")
        return filename

    def collect_huggingface_datasets(self, query):
        """Fetch top 5 datasets from Hugging Face based on the query."""
        headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}
        url = f"https://huggingface.co/api/datasets?search={query}"
        response = requests.get(url, headers=headers)

        if response.status_code == 200:
            datasets = response.json()[:5]  # Get top 5 datasets
            dataset_links = []
            for dataset in datasets:
                dataset_id = dataset.get('id', 'Unknown')
                # Link to the Hugging Face dataset page
                dataset_link = f"https://huggingface.co/datasets/{dataset_id}"
                dataset_links.append(f"[{dataset_id}]({dataset_link})")
            return dataset_links
        else:
            return ["Failed to fetch Hugging Face data."]

    def collect_all_resources(self, industry):
        """Collect all resources for the given industry."""
        huggingface_links = self.collect_huggingface_datasets(industry)
        self.save_to_markdown(industry, huggingface_links)  # Save to markdown file
        return huggingface_links


class ResearchAgent:
    def research_industry(self, industry):
        search_query = f"{industry} AI applications"
        url = f"https://www.google.com/search?q={search_query}"
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers)

        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            snippets = [item.text for item in soup.find_all('span')]
            return snippets[:5]  # Return top 5 snippets
        else:
            return ["Failed to fetch data from the web."]


class UseCaseGenerationAgent:
    def generate_use_cases(self, industry_analysis):
        # Combine industry insights into one text
        analysis_text = " ".join(industry_analysis)

        # Generate use cases based on industry analysis
        prompt = f"Based on the following industry insights: {analysis_text}, generate a list of relevant AI use cases for this industry. Focus on improving operational efficiency, customer satisfaction, and innovation."

        response = co.generate(
            model='command-xlarge',
            prompt=prompt,
            max_tokens=300
        )
        return response.generations[0].text.strip().split('\n')


def main_workflow(industry):
    # Step 1: Research Industry
    research_agent = ResearchAgent()
    industry_analysis = research_agent.research_industry(industry)

    # Step 2: Generate Use Cases
    use_case_agent = UseCaseGenerationAgent()
    use_cases = use_case_agent.generate_use_cases(industry_analysis)

    # Step 3: Collect Industry-Specific Datasets
    resource_agent = ResourceCollectorAgent()
    datasets = resource_agent.collect_all_resources(industry)

    return use_cases, datasets


# Streamlit UI
st.title("AI Internship Project - Generative AI Use Case Generator")

industry = st.text_input("Enter the Industry or Company Name")

if st.button('Generate AI Use Cases and Resources'):
    if industry:
        use_cases, datasets = main_workflow(industry)

        st.subheader("Generated Use Cases")
        for use_case in use_cases:
            # Only display non-empty use cases (removes any empty lines)
            if use_case.strip():
                st.write(f"- {use_case.strip()}")

        st.subheader("Collected Datasets")
        for dataset in datasets:
            # Only display non-empty datasets (removes any empty lines)
            if dataset.strip():
                st.markdown(f"[{dataset.strip()}]({dataset.strip()})")
    else:
        st.write("Please enter a valid industry or company name.")
