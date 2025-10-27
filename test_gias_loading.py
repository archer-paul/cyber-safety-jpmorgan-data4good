"""Quick test script to debug GIAS loading issue"""
import pandas as pd

gias_path = r'C:\Users\Paul\OneDrive - telecom-paristech.fr\Documents\Pro\Hackathon\JPMorgan Data for Good\V2\edubasealldata20251024.csv'

print("Loading GIAS...")
try:
    gias_df = pd.read_csv(gias_path, encoding='utf-8', low_memory=False)
except UnicodeDecodeError:
    gias_df = pd.read_csv(gias_path, encoding='latin-1', low_memory=False)

print(f"Loaded: {len(gias_df):,} rows")

# Check columns
cols = list(gias_df.columns)
print(f"\nTotal columns: {len(cols)}")

def find_col(substrings):
    for s in substrings:
        for c in cols:
            if s.lower() in c.lower():
                return c
    return None

phase_col = find_col(['phaseofeducation', 'phase of education'])
status_col = find_col(['status', 'establishmentstatus'])

print(f"\nPhase column: {phase_col}")
print(f"Status column: {status_col}")

# Filter by status
print(f"\nFiltering by status...")
if 'EstablishmentStatus (code)' in gias_df.columns:
    gias_df = gias_df[gias_df['EstablishmentStatus (code)'].astype(str) == '1']
    print(f"After status filter: {len(gias_df):,}")

# Check phase values
if phase_col:
    print(f"\nPhase values:")
    print(gias_df[phase_col].value_counts())

    # Try filtering
    relevant_phases = ['primary', 'secondary', 'all-through', 'middle']
    mask = gias_df[phase_col].astype(str).str.lower().str.contains('|'.join(relevant_phases), na=False, regex=True)
    print(f"\nMatches: {mask.sum():,}")

    gias_df = gias_df[mask]
    print(f"After phase filter: {len(gias_df):,}")
else:
    print("\n[ERROR] Phase column not found!")

# Check FSM
fsm_col = find_col(['percentagefsm', 'fsm'])
print(f"\nFSM column: {fsm_col}")
if fsm_col:
    gias_df[fsm_col] = pd.to_numeric(gias_df[fsm_col], errors='coerce')
    print(f"FSM data available: {gias_df[fsm_col].notna().sum():,} schools")

# Check coordinates
easting_col = find_col(['easting'])
northing_col = find_col(['northing'])
print(f"\nEasting column: {easting_col}")
print(f"Northing column: {northing_col}")

if easting_col and northing_col:
    gias_df[easting_col] = pd.to_numeric(gias_df[easting_col], errors='coerce')
    gias_df[northing_col] = pd.to_numeric(gias_df[northing_col], errors='coerce')
    valid_coords = (gias_df[easting_col].notna() & gias_df[northing_col].notna()).sum()
    print(f"Valid coordinates: {valid_coords:,} schools")

print(f"\n=== FINAL: {len(gias_df):,} schools ready ===")
