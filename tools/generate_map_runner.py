"""Small runner to generate the interactive map using existing outputs.

This script:
- Loads geocoded Breck schools from outputs/schools_with_geocoding.csv
- Runs a KMeans clustering to create a 'cluster' column
- Optionally runs LeadGenerator to create actionable_school_leads.csv if not present
- Builds an interactive folium map using GeographicVisualizer.create_folium_map

Run from repository root:
    python tools/generate_map_runner.py
"""
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs'

# Add repo root to sys.path so we can import modules
sys.path.insert(0, str(ROOT))

from breck_geographic_clustering import GeographicCluster, GeographicVisualizer

# Optional: use LeadGenerator if actionable leads missing
try:
    from breck_lead_generator import LeadGenerator
except Exception:
    LeadGenerator = None


def main():
    print('Generating map from existing outputs...')

    schools_file = OUT / 'schools_with_geocoding.csv'
    if not schools_file.exists():
        print(f'Could not find {schools_file}. Aborting.')
        return 1

    df = pd.read_csv(schools_file)
    # Keep only geocoded rows with lat/lon
    df = df[pd.notna(df['latitude']) & pd.notna(df['longitude'])].copy()
    df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
    df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')

    if df.empty:
        print('No geocoded schools found. Aborting.')
        return 1

    # Run clustering
    clusterer = GeographicCluster(df)
    clustered = clusterer.perform_kmeans_clustering(n_clusters=5)

    # Save clustered schools for reference
    clustered.to_csv(OUT / 'schools_with_clusters.csv', index=False)

    # Optionally generate leads if missing
    leads_file = OUT / 'actionable_school_leads.csv'
    if not leads_file.exists() and LeadGenerator is not None:
        print('actionable leads file not found; attempting to generate via LeadGenerator')
        # Build a minimal breck summary expected by LeadGenerator
        breck_df = df[['school_name', 'visit_count', 'latitude', 'longitude']].copy()
        # Provide path to edubase if present nearby
        possible_gias = ROOT / 'edubasealldata20251024.csv'
        if not possible_gias.exists():
            possible_gias = ROOT / 'edubasealldata20251024.csv'
        if possible_gias.exists():
            cfg = {
                'champion_min_visits': 2,
                'hotspot_radius_km': 5.0,
                'lead_generation_file': str(leads_file)
            }
            lg = LeadGenerator(breck_df, str(possible_gias), cfg)
            leads = lg.run_lead_generation()
            print(f'Generated {len(leads)} leads')
        else:
            print('GIAS edubase not found locally; skipping lead generation')

    # Build map
    visualizer = GeographicVisualizer(clustered)
    out_map = OUT / 'breck_schools_map.html'
    res = visualizer.create_folium_map(str(out_map), cluster_column='cluster', max_connections_per_lead=5)
    if res:
        print('Map generated at:', res)
        return 0
    else:
        print('Map generation failed')
        return 2


if __name__ == '__main__':
    sys.exit(main())
