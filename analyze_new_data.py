"""
Quick script to analyze the new Excel file structure
"""
import pandas as pd
import sys

file_path = r'C:\Users\Paul\OneDrive - telecom-paristech.fr\Documents\Pro\Hackathon\JPMorgan Data for Good\V2\D4G Data - Lou working doc.xlsx'

print("=" * 80)
print("ANALYZING NEW EXCEL FILE")
print("=" * 80)

try:
    xl = pd.ExcelFile(file_path)
    print(f"\nFound {len(xl.sheet_names)} sheets:")
    for i, sheet in enumerate(xl.sheet_names, 1):
        print(f"  {i}. {sheet}")

    # Analyze first sheet (schools data)
    print("\n" + "=" * 80)
    print("FIRST SHEET ANALYSIS (Schools Data)")
    print("=" * 80)

    df = pd.read_excel(file_path, sheet_name=xl.sheet_names[0])

    print(f"\nShape: {df.shape} (rows × columns)")
    print(f"\nColumns ({len(df.columns)}):")
    for i, col in enumerate(df.columns, 1):
        print(f"  {i}. {col}")

    print(f"\nFirst few rows:")
    print(df.head(10).to_string())

    print(f"\n\nData types:")
    print(df.dtypes)

    print(f"\n\nNon-null counts:")
    print(df.count())

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
