import os
import pandas as pd
from langchain_core.tools import tool
from config.settings import DATA_DIR

@tool
def list_csv_datasets() -> str:
    """List all CSV datasets in the data directory."""
    try:
        csv_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".csv")]
        if not csv_files:
            return "No CSV files found in the data directory."
        return "📁 **Available CSV files:**\n- " + "\n- ".join(csv_files)
    except Exception as e:
        return f"Error listing CSV files: {e}"


@tool
def get_dataset_details(filename: str) -> str:
    """Preview the first 5 rows and show basic info."""
    try:
        file_path = os.path.join(DATA_DIR, filename)
        if not os.path.exists(file_path):
            return f"File '{filename}' not found."
        df = pd.read_csv(file_path)
        info = [
            f"📊 **Dataset:** `{filename}`",
            f"Rows × Columns: {df.shape[0]} × {df.shape[1]}",
            "\n**Preview:**",
            str(df.head()),
        ]
        if df.shape[1] > 5:
            info.append("\n**Column Info:**")
            info.append(str(df.dtypes))
        return "\n".join(info)
    except Exception as e:
        return f"Error reading {filename}: {e}"
