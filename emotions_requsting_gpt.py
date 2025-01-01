import pandas as pd
import re
import os
import json
import time
import requests
from dotenv import load_dotenv
from math import ceil
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry  # Correct import
import glob  # For file pattern matching

# Load environment variables from .env file
load_dotenv()

# Retrieve OpenAI API key from environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("Please set the OPENAI_API_KEY environment variable in your .env file.")

# Define headers for JSON-based API requests
JSON_HEADERS = {
    "Authorization": f"Bearer {OPENAI_API_KEY}",
    "Content-Type": "application/json"
}

# Define headers for file uploads (do NOT include 'Content-Type')
UPLOAD_HEADERS = {
    "Authorization": f"Bearer {OPENAI_API_KEY}"
}

# Define a cleaning function for text
def clean_text(text):
    if pd.isnull(text):
        return ""
    text = re.sub(r"http\S+", "", text)  # Remove URLs
    text = re.sub(r"[^А-Яа-яІіЇїҐґЄє' ]", "", text)  # Keep Ukrainian characters
    text = text.lower().strip()  # Normalize case and trim whitespace
    return text

# Load the annotated CSV file
annotated_csv_file_path = 'annotated_output.csv'
if not os.path.exists(annotated_csv_file_path):
    raise FileNotFoundError(f"The file '{annotated_csv_file_path}' does not exist.")

# To handle mixed types warning
df_annotated = pd.read_csv(annotated_csv_file_path, encoding='utf-8', low_memory=False)

# Add an 'order' column to maintain message sequence
df_annotated = df_annotated.reset_index(drop=True)
df_annotated['order'] = df_annotated.index + 1

# Clean the text data
df_annotated['clean_text'] = df_annotated['text'].apply(clean_text)

# Prepare the batch input file
model_name = "gpt-4o-mini-2024-07-18"  # Ensure the model is supported by Batch API
system_prompt = "You are an expert in sentiment and emotion analysis. Only respond with valid JSON."
user_instructions = (
    "Analyze the following Ukrainian text and return a JSON object with keys: "
    "\"sentiment_score\", \"emotion_intensity\", \"joy\", \"sadness\", \"fear\", \"anger\", \"surprise\", \"disgust\". "
    "All values must be floats between 0.0 and 1.0. "
    "Return JSON only, no extra text."
)

# Define default batch size limit
DEFAULT_BATCH_SIZE_LIMIT = 15000

# Define a set of chunk numbers that require a reduced batch size
REDUCED_BATCH_CHUNKS = set(range(13, 18))  # Chunks 13 to 17 inclusive
REDUCED_BATCH_SIZE = 10000

# Split the dataset into dynamic chunks
total_rows = len(df_annotated)

# Define output folder for output JSONL files
output_folder = 'output'
os.makedirs(output_folder, exist_ok=True)  # Create the folder if it doesn't exist

# Function to extract chunk numbers from output filenames
def get_processed_chunks_from_files(output_folder):
    pattern = os.path.join(output_folder, "batch_output_chunk_*_output.jsonl")
    output_files = glob.glob(pattern)
    processed_chunks = set()
    for file in output_files:
        basename = os.path.basename(file)
        try:
            chunk_number = int(basename.split("_")[3])
            processed_chunks.add(chunk_number)
        except (IndexError, ValueError):
            print(f"Filename '{basename}' does not match the expected pattern. Skipping.")
    return processed_chunks

# Function to calculate total number of chunks considering dynamic batch sizes
def calculate_num_chunks(total_rows, default_size, reduced_chunks, reduced_size):
    num_reduced_chunks = len(reduced_chunks)
    total_reduced_rows = num_reduced_chunks * reduced_size
    remaining_rows = total_rows - total_reduced_rows
    if remaining_rows < 0:
        raise ValueError("Total rows are less than the rows allocated for reduced chunks.")
    num_default_chunks = ceil(remaining_rows / default_size) if remaining_rows > 0 else 0
    return num_reduced_chunks + num_default_chunks

# Calculate the total number of chunks
num_chunks = calculate_num_chunks(total_rows, DEFAULT_BATCH_SIZE_LIMIT, REDUCED_BATCH_CHUNKS, REDUCED_BATCH_SIZE)

# Determine processed chunks based on existing output files
processed_chunks = get_processed_chunks_from_files(output_folder)
processed_chunks_count = len(processed_chunks)
print(f"Detected {processed_chunks_count} processed chunks based on existing files in '{output_folder}'.")

# Check if output CSV exists and load processed orders
output_csv_file_path = 'annotated_with_emotion_scores_batch.csv'
if os.path.exists(output_csv_file_path):
    df_existing = pd.read_csv(output_csv_file_path, encoding='utf-8')
    processed_orders = set(df_existing['order'])
else:
    df_existing = pd.DataFrame()
    processed_orders = set()

# Function to save progress (optional, can be removed if relying solely on output files)
def save_progress(processed_chunks_set):
    progress_file = os.path.join(output_folder, 'processed_chunks.json')
    with open(progress_file, 'w') as pf:
        json.dump(list(processed_chunks_set), pf)

# Initialize the results DataFrame
if not df_existing.empty:
    results_df = df_existing.copy()
else:
    results_df = pd.DataFrame()

# Configure retries for requests
retry_strategy = Retry(
    total=1000000,  # Total retries
    backoff_factor=1,  # Backoff between retries (1, 2, 4, 8, ...)
    status_forcelist=[429, 500, 502, 503, 504],  # Retry on these statuses
    allowed_methods=["HEAD", "GET", "POST"]  # Retry these methods
)
adapter = HTTPAdapter(max_retries=retry_strategy)
session = requests.Session()
session.mount("https://", adapter)
session.mount("http://", adapter)

# Initialize chunk processing variables
chunk_number = 1
start_idx = 0

while start_idx < total_rows and chunk_number <= num_chunks:
    # Determine the batch size for the current chunk
    if chunk_number in REDUCED_BATCH_CHUNKS:
        batch_size = REDUCED_BATCH_SIZE
    else:
        batch_size = DEFAULT_BATCH_SIZE_LIMIT
    end_idx = min(start_idx + batch_size, total_rows)
    
    # Skip if chunk has already been processed
    if chunk_number in processed_chunks:
        print(f"Chunk {chunk_number}/{num_chunks} (Rows {start_idx + 1}-{end_idx}) already processed. Skipping...")
        start_idx += batch_size
        chunk_number += 1
        continue

    # Define input and output filenames
    input_filename = f"batch_input_chunk_{chunk_number}.jsonl"
    input_filepath = os.path.join(output_folder, input_filename)
    output_filename = f"batch_output_chunk_{chunk_number}_output.jsonl"
    output_filepath = os.path.join(output_folder, output_filename)

    if os.path.exists(input_filepath):
        print(f"Input file '{input_filename}' already exists in '{output_folder}'. Skipping creation...")
    else:
        print(f"\nProcessing chunk {chunk_number}/{num_chunks} (Rows {start_idx + 1}-{end_idx})...")

        # Slice the current chunk
        df_chunk = df_annotated.iloc[start_idx:end_idx].copy()

        # Exclude already processed orders within the chunk
        df_chunk = df_chunk[~df_chunk['order'].isin(processed_orders)]

        if df_chunk.empty:
            print(f"All orders in chunk {chunk_number} have already been processed. Skipping...")
            processed_chunks.add(chunk_number)
            save_progress(processed_chunks)
            start_idx += batch_size
            chunk_number += 1
            continue

        # Create the .jsonl input file for the current chunk
        with open(input_filepath, "w", encoding="utf-8") as f:
            for _, row in df_chunk.iterrows():
                request_id = f"request-{row['order']}"
                text = row['clean_text']

                body = {
                    "model": model_name,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"{user_instructions}\nText:\n{text}"}
                    ],
                    "max_tokens": 100,
                    "temperature": 0
                }

                line = {
                    "custom_id": request_id,
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": body
                }
                f.write(json.dumps(line, ensure_ascii=False) + "\n")

        print(f"Batch input file '{input_filename}' created successfully in '{output_folder}'.")

    # Step 1: Upload the batch input file for the current chunk
    upload_url = "https://api.openai.com/v1/files"
    try:
        with open(input_filepath, "rb") as file:
            files = {
                'file': (input_filename, file),
                'purpose': (None, 'batch')
            }
            response = session.post(upload_url, headers=UPLOAD_HEADERS, files=files)
    except Exception as e:
        print(f"Exception occurred while uploading file '{input_filename}': {e}")
        print("Retrying in 60 seconds...")
        time.sleep(60)
        continue  # Retry uploading the same chunk

    if response.status_code == 200:
        batch_input_file = response.json()
        print("Uploaded input file ID:", batch_input_file['id'])
    else:
        print("Failed to upload input file:", response.text)
        print("Retrying in 60 seconds...")
        time.sleep(60)
        continue  # Retry uploading the same chunk

    # Step 2: Create the batch for the current chunk
    create_batch_url = "https://api.openai.com/v1/batches"
    batch_payload = {
        "input_file_id": batch_input_file['id'],
        "endpoint": "/v1/chat/completions",
        "completion_window": "24h",
        "metadata": {
            "description": f"nightly eval job - chunk {chunk_number}"
        }
    }

    try:
        response = session.post(create_batch_url, headers=JSON_HEADERS, json=batch_payload)
    except Exception as e:
        print(f"Exception occurred while creating batch for chunk {chunk_number}: {e}")
        print("Retrying in 60 seconds...")
        time.sleep(60)
        continue  # Retry creating the batch for the same chunk

    if response.status_code == 200:
        batch = response.json()
        print("Batch created with ID:", batch['id'])
    else:
        print("Failed to create batch:", response.text)
        print("Retrying in 300 seconds...")
        time.sleep(300)
        continue  # Retry creating the batch for the same chunk

    # Step 3: Poll the batch status until completion
    batch_id = batch['id']
    check_batch_url = f"https://api.openai.com/v1/batches/{batch_id}"

    print("Polling batch status...")
    while True:
        try:
            response = session.get(check_batch_url, headers=JSON_HEADERS)
        except Exception as e:
            print(f"Exception occurred while checking batch status for batch ID {batch_id}: {e}")
            print("Retrying in 60 seconds...")
            time.sleep(60)
            continue  # Retry fetching the status

        if response.status_code != 200:
            print("Failed to retrieve batch status:", response.text)
            print("Retrying in 60 seconds...")
            time.sleep(60)
            continue  # Retry fetching the status

        current_batch = response.json()
        status = current_batch.get('status')
        print(f"Batch status: {status}")

        if status in ["completed", "failed", "cancelled", "expired"]:
            break
        time.sleep(300)  # Wait 5 minutes before checking again

    if status == "completed":
        # Step 4: Download the results for the current chunk
        output_file_id = current_batch.get('output_file_id')
        if not output_file_id:
            print("No output file ID found. Skipping this chunk.")
            start_idx += batch_size
            chunk_number += 1
            continue

        download_url = f"https://api.openai.com/v1/files/{output_file_id}/content"
        try:
            response = session.get(download_url, headers=JSON_HEADERS)
        except Exception as e:
            print(f"Exception occurred while downloading output file for chunk {chunk_number}: {e}")
            print("Retrying in 60 seconds...")
            time.sleep(60)
            continue  # Retry downloading the same chunk

        if response.status_code == 200:
            output_text = response.content.decode("utf-8")
            print(f"Downloaded output file for chunk {chunk_number}.")
        else:
            print("Failed to download output file:", response.text)
            print("Retrying in 60 seconds...")
            time.sleep(60)
            continue  # Retry downloading the same chunk

        # Save the output as a file
        with open(output_filepath, "w", encoding="utf-8") as f:
            f.write(output_text)

        print(f"Saved output file as '{output_filename}' in '{output_folder}'.")

        # Parse the output lines
        chunk_results = []
        for line in output_text.strip().split("\n"):
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                print("Invalid JSON line. Skipping.")
                continue

            custom_id = data.get("custom_id")
            error = data.get("error")

            if error is None:
                response_body = data.get("response", {}).get("body", {})
                choices = response_body.get("choices", [])
                if choices:
                    model_output = choices[0].get("message", {}).get("content", "").strip()
                    try:
                        emotion_data = json.loads(model_output)
                    except json.JSONDecodeError:
                        # If parsing fails, default scores
                        emotion_data = {
                            "sentiment_score": 0.0,
                            "emotion_intensity": 0.0,
                            "joy": 0.0,
                            "sadness": 0.0,
                            "fear": 0.0,
                            "anger": 0.0,
                            "surprise": 0.0,
                            "disgust": 0.0
                        }
                else:
                    # No choices returned
                    emotion_data = {
                        "sentiment_score": 0.0,
                        "emotion_intensity": 0.0,
                        "joy": 0.0,
                        "sadness": 0.0,
                        "fear": 0.0,
                        "anger": 0.0,
                        "surprise": 0.0,
                        "disgust": 0.0
                    }
            else:
                # If there's an error, return default scores
                emotion_data = {
                    "sentiment_score": 0.0,
                    "emotion_intensity": 0.0,
                    "joy": 0.0,
                    "sadness": 0.0,
                    "fear": 0.0,
                    "anger": 0.0,
                    "surprise": 0.0,
                    "disgust": 0.0
                }

            emotion_data["custom_id"] = custom_id
            chunk_results.append(emotion_data)

        # Convert chunk results to DataFrame
        chunk_results_df = pd.DataFrame(chunk_results)

        # Extract the order number from custom_id (e.g., "request-23" -> 23)
        chunk_results_df['order'] = chunk_results_df['custom_id'].apply(
            lambda x: int(x.split("-")[1]) if isinstance(x, str) and "-" in x else None
        )
        chunk_results_df = chunk_results_df.drop('custom_id', axis=1)

        # Merge with original df_chunk on 'order'
        df_merged = pd.merge(df_chunk, chunk_results_df, on='order', how='left')

        # Append to the output CSV incrementally
        if not os.path.exists(output_csv_file_path):
            df_merged.to_csv(output_csv_file_path, index=False, encoding='utf-8')
            print(f"Created new output CSV '{output_csv_file_path}'.")
        else:
            df_merged.to_csv(output_csv_file_path, mode='a', index=False, header=False, encoding='utf-8')
            print(f"Appended results of chunk {chunk_number} to '{output_csv_file_path}'.")

        # Update progress tracking
        processed_chunks.add(chunk_number)
        save_progress(processed_chunks)  # Save the updated set of processed chunks

        # Update processed_orders to avoid reprocessing within this run
        processed_orders.update(df_merged['order'].tolist())

    else:
        print(f"Batch for chunk {chunk_number} not completed successfully. Status: {status}")
        print("Retrying in 300 seconds...")
        time.sleep(300)
        continue  # Retry processing this chunk

    # Confirmation step after completing a chunk
    user_confirmation = input(
        f"Chunk {chunk_number}/{num_chunks} completed. "
        "Check your balance or perform other checks. "
        "Press Enter to continue to the next chunk, or type 'exit' to stop: "
    )
    if user_confirmation.strip().lower() == 'exit':
        print("Exiting script. Progress has been saved.")
        break

    # Update indices for the next chunk
    start_idx += batch_size
    chunk_number += 1

print("\nAll specified chunks processed. Finalizing the results.")

# Optional: If you want to ensure that all results are merged into the final CSV
# (This is already handled incrementally, so this step may be redundant)
# df_final = pd.merge(df_annotated, results_df, on='order', how='left')
# df_final.to_csv(output_csv_file_path, index=False, encoding='utf-8')

print(f"Annotated data with sentiment and emotion scores saved to '{output_csv_file_path}'")
