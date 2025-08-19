import pandas as pd
import os
import glob

# Root directory containing subfolders
root_folder = r'C:\Research Work\Research Proposal Preparation\ACC-N-AEC\SearchQueries'  # Change this path

def fix_encoding_issues(text):
    """Fix common encoding issues like UKâ€™s → UK’s"""
    if isinstance(text, str):
        try:
            return text.encode('latin1').decode('utf-8')
        except:
            return text
    return text

for folder_name in os.listdir(root_folder):
    folder_path = os.path.join(root_folder, folder_name)

    if os.path.isdir(folder_path):
        print(f"\n📂 Processing folder: {folder_name}")

        # 1. Convert Excel files (.xls/.xlsx) to CSV
        excel_files = glob.glob(os.path.join(folder_path, '*.xls*'))  # Matches .xls and .xlsx
        for excel_file in excel_files:
            try:
                df = pd.read_excel(excel_file, dtype=str, engine='openpyxl' if excel_file.endswith('xlsx') else None)
                csv_path = os.path.splitext(excel_file)[0] + '.csv'
                df.to_csv(csv_path, index=False, encoding='utf-8')
                print(f"Converted: {os.path.basename(excel_file)} → {os.path.basename(csv_path)}")
            except ImportError:
                print("⚠ Please install dependencies: pip install openpyxl xlrd")
            except Exception as e:
                print(f"⚠ Error converting {excel_file}: {e}")

        # 2. Combine all CSV files in the folder
        csv_files = glob.glob(os.path.join(folder_path, '*.csv'))
        df_list = []
        for file in csv_files:
            try:
                df = pd.read_csv(file, dtype=str, encoding='utf-8', on_bad_lines='skip')
                # Fix encoding issues in all text columns
                for col in df.columns:
                    df[col] = df[col].map(fix_encoding_issues)
                df_list.append(df)
                print(f"Loaded: {os.path.basename(file)} ({df.shape[0]} rows)")
            except Exception as e:
                print(f"⚠ Error reading {file}: {e}")

        if df_list:
            # Combine & remove duplicates
            combined_df = pd.concat(df_list, ignore_index=True, sort=False)
            before_count = combined_df.shape[0]
            combined_df.drop_duplicates(keep='first', inplace=True)
            after_count = combined_df.shape[0]

            # Save combined file for the folder
            output_file = os.path.join(folder_path, f"{folder_name}_combined_unique.csv")
            combined_df.to_csv(output_file, index=False, encoding='utf-8')
            print(f"Saved combined file: {output_file}")
            print(f"Rows before: {before_count}, after removing duplicates: {after_count}")
        else:
            print("No valid CSV files found in this folder.")
