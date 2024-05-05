# Required Libraries Installation
'''!pip install sec_api
!pip install langchain
!pip install unstructured
!pip install openai
!pip install tiktoken
!pip install plotly
!pip install kaleido
import plotly.graph_objs as go
from sec_api import ExtractorApi
from langchain import OpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain import PromptTemplate
from langchain.chains import LLMChain'''

# Fetching Item 1A from a 10-K report using the SEC API
sec_api = ExtractorApi("api_key")
api_key_openai = 'api_key'
url_10k_report = "https://www.sec.gov/ix?doc=/Archives/edgar/data/0000320193/000032019323000106/aapl-20230930.htm"
text_section_1A = sec_api.get_section(url_10k_report, "1A", "text")

# Function to write text to a file
def save_text_file(file_path, content):
    try:
        with open(file_path, 'w') as text_file:
            text_file.write(content)
        print("File successfully saved to", file_path)
    except Exception as error:
        print("Failed to save file due to:", error)

    save_path = "summary.txt"
    save_text_file(save_path, text_section_1A)

language_model = OpenAI(api_key=api_key_openai)  # Initialize language model

file_path_summary = '/content/summary.txt'

# Function to read text content from a file
def read_text_file(file_path):
    try:
        with open(file_path, 'r') as file:
            content = file.read()
        return content
    except Exception as error:
        print("Failed to read file due to:", error)
        return None
summary_text = read_text_file(file_path_summary)

# Split text into manageable parts and summarize each part
token_count = language_model.get_num_tokens(summary_text)
document_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n"], chunk_size=10000, overlap=500)
document_chunks = document_splitter.create_documents([summary_text])
document_count = len(document_chunks)
tokens_first_document = language_model.get_num_tokens(document_chunks[0].page_content)
print (f"Split into {document_count} parts; the first contains {tokens_first_document} tokens")
summary_process = load_summarize_chain(llm=language_model, chain_type='map_reduce')
summary_result = summary_process.run(document_chunks)

# Enhance summaries with better structured prompts
prompt_summary_map = """
Provide a concise summary for the following content:
"{text}"
CONCISE SUMMARY:
"""
template_summary_map = PromptTemplate(template=prompt_summary_map, input_variables=["text"])

prompt_summary_combine = """
Summarize the text below, formatted with bullet points to highlight key points:
```{text}```
BULLET POINT SUMMARY:
"""
template_summary_combine = PromptTemplate(template=prompt_summary_combine, input_variables=["text"])

summary_process_enhanced = load_summarize_chain(llm=language_model,
                                                chain_type='map_reduce',
                                                map_prompt=template_summary_map,
                                                combine_prompt=template_summary_combine)
enhanced_summary = summary_process_enhanced.run(document_chunks)
formatted_enhanced_summary = "\n".join(["* " + line for line in enhanced_summary.split("\n")])
save_text_file('enhanced_summary_1A.txt', formatted_enhanced_summary)

# Generate one-word descriptions for risk factors
template_one_word_risks = '''Provide a single word for each one-line explanation of risk factors:
{risk_factors}'''
prompt_one_word_risks = PromptTemplate(template=template_one_word_risks, input_variables=['risk_factors'])
risk_chain = LLMChain(llm=language_model, prompt=prompt_one_word_risks)
one_word_descriptions = risk_chain.run(formatted_enhanced_summary)
formatted_one_word_descriptions = "\n".join(["* " + line for line in one_word_descriptions.split("\n")])
save_text_file('one_word_descriptions.txt', formatted_one_word_descriptions)

# Rating risk factors based on intensity
template_rating_risks = '''Rate each risk factor on a scale from 1 to 10, with 10 being the most severe. Provide diverse ratings for the following risk factors:
{risk_factors}'''
prompt_rating_risks = PromptTemplate(template=template_rating_risks, input_variables=['risk_factors'])
rating_chain = LLMChain(llm=language_model, prompt=prompt_rating_risks)
risk_ratings = rating_chain.run(formatted_enhanced_summary)
formatted_risk_ratings = "\n".join(["* " + line for line in risk_ratings.split("\n")])
save_text_file('risk_ratings.txt', formatted_risk_ratings)

# Plot intensity ratings of risk factors using Plotly
risk_data = read_text_file('/content/one_word_descriptions.txt')
risk_intensity = read_text_file('/content/risk_ratings.txt')

risk_labels = [line.split("* ")[1].strip() for line in risk_data.split("\n")]
intensity_values = [int(line.split("* ")[1]) for line in risk_intensity.split("\n")]

# Creating a bar chart
risk_chart = go.Figure(data=[go.Bar(
    x=risk_labels,
    y=intensity_values,
    marker_color='lightskyblue'
)])
risk_chart.update_layout(
    title="Risk Factor Intensity Ratings",
    xaxis_title="Risk Factors",
    yaxis_title="Intensity",
    yaxis=dict(range=[0, 10])
)
risk_chart.show()
risk_chart.write_image("risk_factors_intensity_chart.png", engine="kaleido")
