import os
import json
import requests
import subprocess
import streamlit as st
from bs4 import BeautifulSoup
import cohere
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Access API keys from environment variables
HUGGINGFACE_API_KEY = os.getenv('HUGGINGFACE_API_KEY')
KAGGLE_USERNAME = os.getenv("KAGGLE_USERNAME")
KAGGLE_KEY = os.getenv("KAGGLE_KEY")
COHERE_API_KEY = os.getenv('COHERE_API_KEY')

# Ensure the keys are not missing
if not HUGGINGFACE_API_KEY or not KAGGLE_USERNAME or not KAGGLE_KEY or not COHERE_API_KEY:
    raise ValueError("One or more API keys are missing. Check your .env file or deployment settings.")

# Create kaggle.json dynamically
if KAGGLE_USERNAME and KAGGLE_KEY:
    kaggle_config = {"username": KAGGLE_USERNAME, "key": KAGGLE_KEY}
    os.makedirs(os.path.expanduser("~/.kaggle"), exist_ok=True)
    with open(os.path.expanduser("~/.kaggle/kaggle.json"), "w") as f:
        json.dump(kaggle_config, f)
    os.chmod(os.path.expanduser("~/.kaggle/kaggle.json"), 0o600)

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
        headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}
        url = f"https://huggingface.co/api/datasets?search={query}"
        response = requests.get(url, headers=headers)

        if response.status_code == 200:
            datasets = response.json()[:5]  # Get top 5 datasets
            return [
                f"[{dataset.get('id', 'Unknown')}]"
                f"(https://huggingface.co/datasets/{dataset.get('id', 'Unknown')})"
                for dataset in datasets
            ]
        else:
            return ["Failed to fetch Hugging Face data."]

    def collect_kaggle_datasets(self, query):
        try:
            result = subprocess.run(
                ['kaggle', 'datasets', 'list', '--search', query, '--csv'],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                lines = result.stdout.split('\n')[1:6]
                return [
                    f"[{line.split(',')[0]}](https://www.kaggle.com/datasets/{line.split(',')[0]})"
                    for line in lines if line.strip()
                ]
            else:
                return ["Kaggle API request failed."]
        except Exception as e:
            return [f"Error: {str(e)}"]

    def collect_all_resources(self, industry):
        huggingface_links = self.collect_huggingface_datasets(industry)
        kaggle_links = self.collect_kaggle_datasets(industry)
        all_links = huggingface_links + kaggle_links
        self.save_to_markdown(industry, all_links)  # Save to markdown file
        return all_links


class ResearchAgent:
    def research_industry(self, industry):
        search_query = f"{industry} AI applications"
        url = f"https://www.google.com/search?q={search_query}"
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            return [item.text for item in soup.find_all('span')][:5]
        else:
            return ["Failed to fetch data from the web."]


class UseCaseGenerationAgent:
    def generate_use_cases(self, industry_analysis):
        analysis_text = " ".join(industry_analysis)
        prompt = (
            f"Based on the following industry insights: {analysis_text}, generate a list of "
            f"relevant AI use cases for this industry. Focus on operational efficiency and innovation."
        )
        response = co.generate(
            model='command-xlarge',
            prompt=prompt,
            max_tokens=300
        )
        return response.generations[0].text.strip().split('\n')


def main_workflow(industry):
    research_agent = ResearchAgent()
    industry_analysis = research_agent.research_industry(industry)

    use_case_agent = UseCaseGenerationAgent()
    use_cases = use_case_agent.generate_use_cases(industry_analysis)

    resource_agent = ResourceCollectorAgent()
    datasets = resource_agent.collect_all_resources(industry)

    return use_cases, datasets


# Streamlit UI
st.title("AI Internship Project - Generative AI Use Case Generator")

industry = st.text_input("Enter the Industry or Company Name")

if st.button('Generate AI Use Cases and Resources'):
    if industry:
        try:
            use_cases, datasets = main_workflow(industry)
            st.subheader("Generated Use Cases")
            for use_case in use_cases:
                if use_case.strip():
                    st.write(f"- {use_case.strip()}")

            st.subheader("Collected Datasets")
            for dataset in datasets:
                if dataset.strip():
                    st.markdown(dataset, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.error("Please enter a valid industry or company name.")
