import pandas as pd
import csv

# Paths 
deepsearch_file = r'C:\Research Work\Research Proposal Preparation\ACC-N-AEC\SearchQueries\DeepSearch\DeepSearch_combined_unique.csv'
scopus_file = r'C:\Research Work\Research Proposal Preparation\ACC-N-AEC\SearchQueries\Scopus\Scopus_combined_unique.csv'
wos_file = r'C:\Research Work\Research Proposal Preparation\ACC-N-AEC\SearchQueries\WebOfScience\WebOfScience_combined_unique.csv'
ieee_file = r'C:\Research Work\Research Proposal Preparation\ACC-N-AEC\SearchQueries\IEEEXplore\IEEEXplore_combined_unique.csv'

# Fix encoding issues like UKâ€™s → UK’s
def fix_encoding_issues(text):
    if isinstance(text, str):
        try:
            return text.encode('latin1').decode('utf-8')
        except:
            return text
    return text

# Safe loader for messy CSVs
def safe_load_csv(file):
    try:
        with open(file, 'r', encoding='utf-8', errors='replace') as f:
            sample = f.read(2048)
            f.seek(0)
            dialect = csv.Sniffer().sniff(sample)
        df = pd.read_csv(file, dtype=str, encoding='utf-8', delimiter=dialect.delimiter, quotechar='"', on_bad_lines='skip')
        # Apply encoding fix
        for col in df.columns:
            df[col] = df[col].map(fix_encoding_issues)
        return df
    except Exception as e:
        print(f"⚠ Error reading {file}: {e}")
        return pd.DataFrame()

# Load all 4 files
deep_df = safe_load_csv(deepsearch_file)
scopus_df = safe_load_csv(scopus_file)
wos_df = safe_load_csv(wos_file)
ieee_df = safe_load_csv(ieee_file)

# Normalize columns and extract only needed ones
def normalize_df(df, mapping, source_name):
    data = {}
    for target_col, source_cols in mapping.items():
        for col in source_cols:
            if col in df.columns:
                data[target_col] = df[col]
                break
        if target_col not in data:
            data[target_col] = None
    data['Source'] = source_name
    return pd.DataFrame(data)

# Column mappings for each source
deep_mapping = {
    'Authors': ['Authors'],
    'Title': ['Title'],
    'Year': ['Year'],
    'Keywords': [],  # DeepSearch has no keywords
    'Abstract': ['Abstract']
}

scopus_mapping = {
    'Authors': ['Authors', 'Author full names'],
    'Title': ['Title'],
    'Year': ['Year'],
    'Keywords': ['Author Keywords', 'Index Keywords'],
    'Abstract': ['Abstract']
}

wos_mapping = {
    'Authors': ['Authors', 'Author Full Names'],
    'Title': ['Article Title'],
    'Year': ['Publication Year'],
    'Keywords': ['Author Keywords', 'Keywords Plus'],
    'Abstract': ['Abstract']
}

ieee_mapping = {
    'Authors': ['Authors'],
    'Title': ['Document Title'],
    'Year': ['Publication Year'],
    'Keywords': ['Author Keywords'],
    'Abstract': ['Abstract']
}

# Create normalized DataFrames
deep_clean = normalize_df(deep_df, deep_mapping, 'DeepSearch')
scopus_clean = normalize_df(scopus_df, scopus_mapping, 'Scopus')
wos_clean = normalize_df(wos_df, wos_mapping, 'WebOfScience')
ieee_clean = normalize_df(ieee_df, ieee_mapping, 'IEEE Xplore')

# Combine all
combined_df = pd.concat([deep_clean, scopus_clean, wos_clean, ieee_clean], ignore_index=True)

# Remove duplicates based on Title
combined_df.drop_duplicates(subset=['Title'], keep='first', inplace=True)

# Clean whitespace
combined_df = combined_df.map(lambda x: x.strip() if isinstance(x, str) else x)

# Save final output
output_file = r'C:\Research Work\Research Proposal Preparation\ACC-N-AEC\AllSources_Combined_Clean_Final.csv'
combined_df.to_csv(output_file, index=False, encoding='utf-8')

print(f"\n Final clean file saved: {output_file}")
print(f"Total unique records: {len(combined_df)}")
