import json
import pandas as pd
import stanza
from tqdm import tqdm

# Load the JSON file
file_path = r'C:\Users\Viktor Batih\Downloads\Telegram Desktop\ChatExport_2024-11-30\result.json'
with open(file_path, 'r', encoding='utf-8') as file:
    data = json.load(file)

# Extract messages or relevant data
messages = data.get("messages", [])  # Adjust the key based on your JSON structure

# Create a DataFrame
df = pd.DataFrame(messages)

# Save to CSV
csv_file_path = 'output.csv'  # Specify the output CSV file path
df.to_csv(csv_file_path, index=False, encoding='utf-8')

print(f"CSV file saved to {csv_file_path}")

# Initialize Stanza pipeline for Ukrainian
stanza.download('uk')
nlp = stanza.Pipeline('uk')

# Annotate the "text" column with Stanza
def annotate_text(text):
    if isinstance(text, str):  # Ensure it's a string before processing
        doc = nlp(text)
        annotations = []
        for sentence in doc.sentences:
            for word in sentence.words:
                annotations.append({
                    "Word": word.text,
                    "POS": word.upos,
                    "Lemma": word.lemma,
                    "Dependency": word.deprel
                })
        return annotations
    return None  # Return None for non-string entries

# Add a progress bar to show processing status
if "text" in df.columns:
    tqdm.pandas(desc="Annotating messages")
    df["annotations"] = df["text"].progress_apply(annotate_text)

# Save the annotated DataFrame to a new CSV file
annotated_csv_file_path = 'annotated_output.csv'
df.to_csv(annotated_csv_file_path, index=False, encoding='utf-8')

print(f"Annotated CSV file saved to {annotated_csv_file_path}")