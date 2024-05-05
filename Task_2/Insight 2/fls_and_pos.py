#installations and imports
'''!pip install sec-api
!pip install matplotlib
from sec_api import QueryApi,ExtractorApi
import multiprocessing
import time
import re
import nltk
import requests
import matplotlib.pyplot as plt'''

#extraction of 5 recent filings of AAPL at a time using query and extractor api

queryApi = QueryApi(api_key="f5c7ffbbc22e823324ce7468d0f7f602715650023185f18986e2263437147cb2")
extractorApi = ExtractorApi(api_key='f5c7ffbbc22e823324ce7468d0f7f602715650023185f18986e2263437147cb2')

base_query = {
  "query": "ticker:AAPL AND filedAt:[2019-01-01 TO 2023-12-31] AND formType:\"10-K\"",
  "from": "0",
  "size": "5",
  "sort": [{ "filedAt": { "order": "desc" } }]
}

log_file = open("filing_urls.txt", "a")

response = queryApi.get_filings(base_query)

urls_list = list(map(lambda x: x["linkToFilingDetails"], response["filings"]))

urls_string = "\n".join(urls_list) + "\n"

log_file.write(urls_string)
log_file.close()

log_file = open("filing_urls.txt", "r")


def extract_items_10k(filing_url):
    items = ["7"]

    for item in items:
        print("item:", item, "url", filing_url)

        try:
            section_text = extractorApi.get_section(filing_url=filing_url,
                                                     section=item,
                                                     return_type="text")


            timestamp = int(time.time())
            output_file = f"section_{item}_{timestamp}.txt"
            with open(output_file, "w") as f:
                f.write(section_text)

        except Exception as e:
            print(e)

number_of_processes = 5

with multiprocessing.Pool(number_of_processes) as pool:
  pool.map(extract_items_10k, urls_list)

with open('/content/2023.txt', 'r') as file: #reading all the files in the same way
    text_2023 = file.read()

#removing new lines and html tages
def clean(text):
 cleaned_text = re.sub(r"\n|&#[0-9]+;", "", text)
 return cleaned_text
cleaned_2019= clean(text_2019)
cleaned_2020= clean(text_2020)
cleaned_2021= clean(text_2021)
cleaned_2022= clean(text_2022)
cleaned_2023= clean(text_2023)

#removing tables from the text
def remove_tables(text_content, table_start_marker='#TABLE_START', table_end_marker='#TABLE_END'):

    start_index = text_content.find(table_start_marker)

    while start_index != -1:
        end_index = text_content.find(table_end_marker, start_index)
        if end_index != -1:
            text_content = text_content[:start_index] + text_content[end_index + len(table_end_marker):]
            start_index = text_content.find(table_start_marker, start_index)
        else:
            # If end marker not found, break the loop
            break

    return text_content

nt2019= remove_tables(cleaned_2019)
nt2020= remove_tables(cleaned_2020)
nt2021= remove_tables(cleaned_2021)
nt2022= remove_tables(cleaned_2022)
nt2023= remove_tables(cleaned_2023)

# breaking the text into sentences using nltk
nltk.download('punkt')
def break_text_into_sentences(text):
    sentences = nltk.sent_tokenize(text)
    return sentences

# Read text files and break them into sentences and storing them in a dictionary
file_names = ["nt2019.txt", "nt2020.txt", "nt2021.txt", "nt2022.txt", "nt2023.txt"]
sentences_dict={}
for file_name in file_names:
    with open(file_name, 'r') as file:
        text = file.read()
        sentences = break_text_into_sentences(text)
        sentences_dict[file_name]= sentences

        print(f"Sentences in {file_name}: {len(sentences)}")


# Using Hugging Face inference api to store specific and non-specific fls sentences in a list
API_URL = "https://api-inference.huggingface.co/models/yiyanghkust/finbert-fls"
headers = {"Authorization": "Bearer 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"}

def query(sentences):
    payload = {"inputs": sentences}
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

def process_output(output, sentences):
    fls_2019 = []

    for i in range(len(output)):
        label = output[i][0]['label']
        if label == 'Specific FLS':
            fls_2019.append(sentences[i])
        elif label == 'Non-specific FLS':
            fls_2019.append(sentences[i])
    return fls_2019


#calling the output and storing all the lists in a dictionary
output = query(sentences_dict['nt2019.txt'])
fls_2019 = process_output(output, sentences_dict['nt2019.txt'])
print(" FLS:", fls_2019)

texts = ["nt2019.txt", "nt2020.txt", "nt2021.txt", "nt2022.txt", "nt2023.txt"]
fls_lists = [fls_2019, fls_2020, fls_2021, fls_2022, fls_2023]

fls_dict = dict(zip(texts, fls_lists))

# calculating percentage of fls sentences in a particular text
def calculate_fls_percentage(all_sentences, fls_sentences):
    fls_count = len(fls_sentences)
    total_sentences_count = len(all_sentences)
    if total_sentences_count == 0:
        return 0  # Return 0 if there are no sentences in the text
    fls_percentage = (fls_count / total_sentences_count) * 100
    return fls_percentage

def calculate_fls_percentage_for_texts(texts, fls_sentences_per_text):
    fls_percentage_per_text = []
    for text_name, all_sentences in texts.items():
        fls_sentences = fls_sentences_per_text.get(text_name, [])
        fls_percentage = calculate_fls_percentage(all_sentences, fls_sentences)
        fls_percentage_per_text.append(fls_percentage)
    return fls_percentage_per_text

fls_percentage = calculate_fls_percentage_for_texts(sentences_dict, fls_dict)

print("FLS percentage for each text:", fls_percentage)
fls_percentage.reverse()
years = list([2019,2020,2021,2022,2023])


# Create line chart
plt.figure(figsize=(10, 6))
plt.plot(years, fls_percentage, marker='o', linestyle='-')
plt.title('Percentage of Forward-looking Sentences')
plt.xlabel('Year')
plt.ylabel('Percentage')
plt.grid(True)
plt.xticks(years)
plt.show()
plt.savefig('fls_vs_time.png')



''' PROGRAM TO CALCULATE POSITIVE SENTENCES IN THE ITEM 7 SECTION FOR LAST 5 YEARS '''
# FINERT MODEL with the same procedure as above
API_URL = "https://api-inference.huggingface.co/models/ProsusAI/finbert"
headers = {"Authorization": "Bearer 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"}

def query(sentences):
    payload = {"inputs": sentences}
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

def process_output(output, sentences):
    pos_2023 = []

    for i in range(len(output)):
        label = output[i][0]['label']
        if label == 'positive':
            pos_2023.append(sentences[i])
        '''elif label == 'Non-specific FLS':
            fls_2019.append(sentences[i])'''
    return pos_2023


output = query(sentences_dict['nt2023.txt'])
pos_2023 = process_output(output, sentences_dict['nt2023.txt'])
print(" FLS:", pos_2023)
pos_lists = [pos_2019, pos_2020, pos_2021, pos_2022, pos_2023]
pos_dict = dict(zip(texts, pos_lists)len(pos_2023))

def calculate_pos_percentage(all_sentences, pos_sentences):
    pos_count = len(pos_sentences)
    total_sentences_count = len(all_sentences)
    if total_sentences_count == 0:
        return 0  # Return 0 if there are no sentences in the text
    pos_percentage = (pos_count / total_sentences_count) * 100
    return pos_percentage

def calculate_pos_percentage_for_texts(texts, pos_sentences_per_text):
    pos_percentage_per_text = []
    for text_name, all_sentences in texts.items():
        pos_sentences = pos_sentences_per_text.get(text_name, [])
        pos_percentage = calculate_pos_percentage(all_sentences, pos_sentences)
        pos_percentage_per_text.append(pos_percentage)
    return pos_percentage_per_text

pos_percentage = calculate_pos_percentage_for_texts(sentences_dict, pos_dict)

#PLOTTING THE GRAPH OF % POSITIVE SENTENCES VS YEARS
plt.figure(figsize=(10, 6))
plt.plot(years, pos_percentage, marker='o', linestyle='-')
plt.title('Percentage of positive Sentences')
plt.xlabel('Year')
plt.ylabel('Percentage')
plt.grid(True)
plt.xticks(years)
plt.show()
plt.savefig('pos_vs_time.png')

#Combined line graph
plt.plot(years, fls_percentage, label='FLS')
plt.plot(years, pos_percentage, label='Positive')
plt.xlabel('Year')
plt.ylabel('Percentage')
plt.title('FLS and Positive statement analysis')
plt.legend()
plt.show()
plt.savefig('combined.png')
