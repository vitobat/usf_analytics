import sys
import pandas as pd
import numpy as np
import ast
import warnings

warnings.filterwarnings('ignore')  # Suppress unnecessary warnings

# ---------------------------
# Step 1: Load and Prepare Data
# ---------------------------

input_file = 'cleaned_text_and_annotations.csv'

try:
    data = pd.read_csv(input_file)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    print(f"Loaded data from {input_file} successfully. Shape: {data.shape}")
except FileNotFoundError:
    print(f"File {input_file} not found.")
    sys.exit(1)
except Exception as e:
    print(f"An error occurred while loading the file: {e}")
    sys.exit(1)

# ---------------------------
# Step 2: Filter and Process Data
# ---------------------------

# Filter rows where 'from_id' starts with 'user'
if 'from_id' in data.columns:
    data['from_id'] = data['from_id'].astype(str)
    data_filtered = data[data['from_id'].str.startswith('user')].copy()
    print(f"Filtered shape: {data_filtered.shape} (kept only 'user' from_id).")
else:
    print("The 'from_id' column is missing.")
    sys.exit(1)

# Ensure required columns are present
required_columns = ['from_id', 'id', 'reply_to_message_id', 'date', 'reactions']
missing_columns = [col for col in required_columns if col not in data_filtered.columns]
if missing_columns:
    print(f"Missing required columns: {missing_columns}")
    sys.exit(1)

data_filtered['date'] = pd.to_datetime(data_filtered['date'], errors='coerce')

# ---------------------------
# Step 3: Define Helper Functions
# ---------------------------

def parse_reactions(rs):
    if pd.isna(rs) or rs.strip() == '':
        return []
    try:
        val = ast.literal_eval(rs)
        if isinstance(val, list):
            return [xx.get('from_id') for r in val for xx in r.get('recent', []) if 'from_id' in xx]
        return []
    except (ValueError, SyntaxError):
        return []

def calculate_reaction_count(rs):
    if pd.isna(rs) or rs.strip() == '':
        return 0
    try:
        val = ast.literal_eval(rs)
        if isinstance(val, list):
            return sum(r.get('count', 0) for r in val)
        return 0
    except (ValueError, SyntaxError):
        return 0

# Process reactions
data_filtered['parsed_reactions'] = data_filtered['reactions'].apply(parse_reactions)
data_filtered['reaction_count'] = data_filtered['reactions'].apply(calculate_reaction_count)

# Map 'reply_to_message_id' to 'from_id'
id_to_from_id = data_filtered.set_index('id')['from_id']
data_filtered['replied_to_from_id'] = data_filtered['reply_to_message_id'].map(id_to_from_id)

# ---------------------------
# Step 4: Compute Metrics
# ---------------------------

# Aggregate metrics
behavior = data_filtered.groupby('from_id').agg(
    message_count=('id', 'count'),
    reaction_count=('reaction_count', 'sum'),
    replies_sent=('reply_to_message_id', lambda x: x.notnull().sum()),
    replies_received=('replied_to_from_id', lambda x: x.notnull().sum())
).reset_index()

# Process Reactions Sent
all_senders = data_filtered['parsed_reactions'].explode().dropna()
reactions_sent_series = all_senders.value_counts()
reactions_sent = reactions_sent_series.reset_index()
reactions_sent.columns = ['from_id', 'reactions_sent']

# Merge Reactions Sent
behavior = behavior.merge(reactions_sent, on='from_id', how='left').fillna(0)

# Merge User Names if available
if 'from' in data_filtered.columns:
    user_names = data_filtered[['from_id', 'from']].drop_duplicates()
    behavior = behavior.merge(user_names, on='from_id', how='left')

# ---------------------------
# Step 5: Add Rankings and Categories
# ---------------------------

# Normalize and rank metrics
metrics_map = {
    'message_count': "High-volume talker",
    'replies_received': "Conversation Starter",
    'replies_sent': "Frequent Replier",
    'reaction_count': "Reaction Magnet",
    'reactions_sent': "Reactor"
}

for metric in metrics_map.keys():
    mi, ma = behavior[metric].min(), behavior[metric].max()
    if ma - mi > 0:
        behavior[f"rank_{metric}"] = (behavior[metric] - mi) / (ma - mi)
    else:
        behavior[f"rank_{metric}"] = 0

priority = list(metrics_map.keys())

def assign_top_category(row):
    rank_vals = {metric: row[f'rank_{metric}'] for metric in metrics_map.keys()}
    max_rank = max(rank_vals.values())
    top_metrics = [m for m in priority if rank_vals[m] == max_rank]
    return ", ".join(metrics_map[m] for m in top_metrics[:2])

behavior['user_groups'] = behavior.apply(assign_top_category, axis=1)

# ---------------------------
# Step 6: Save Metrics
# ---------------------------

output_file = 'behavior_metrics.csv'
behavior.to_csv(output_file, index=False, encoding='utf-8')
print(f"Metrics saved to {output_file}. Shape: {behavior.shape}")
