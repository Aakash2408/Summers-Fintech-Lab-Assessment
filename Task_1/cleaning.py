from bs4 import BeautifulSoup
import os

# Function to parse and clean SEC filings
def clean_sec_filing(input_file_path, output_file_path):
    with open(input_file_path, 'r', encoding='utf-8') as input_file:
        html_content = input_file.read()
        # Parse the HTML content using BeautifulSoup
        soup = BeautifulSoup(html_content, 'lxml')
        # Extract text content
        text_content = soup.get_text(separator='\n')
        
        # Write cleaned text to output file
        with open(output_file_path, 'w', encoding='utf-8') as output_file:
            output_file.write(text_content)

#Example usage
# Directory containing the SEC filings text files
input_directory = './sec-edgar-filings/AAPL/10-K'
# Directory to save cleaned text files
output_directory = './sec-edgar-filings/AAPL/Cleaned 10-K'

# Recursively iterate through all files in the input directory and its subdirectories
for root, dirs, files in os.walk(input_directory):
    for filename in files:
        if filename.endswith(".txt"):
            input_file_path = os.path.join(root, filename)
            # Removing the input directory path and join with the output directory path
            relative_path = os.path.relpath(input_file_path, input_directory)
            output_file_path = os.path.join(output_directory, relative_path)
            # Ensuring the existence the output directory
            os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
            # Parse and clean SEC filing text
            clean_sec_filing(input_file_path, output_file_path)
            

