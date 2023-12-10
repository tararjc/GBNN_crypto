import re

def extract_string(line, pattern):
    match = re.search(pattern, line)
    if match:
        extracted_strings = match.group(0)
    return extracted_strings

pattern = r"([A-Za-z]+-[A-Za-z]+)"