"""
BRECK FOUNDATION - MODULE: LEAD GENERATOR
=========================================

Author: JP Morgan Data for Good Hackathon Team
Date: October 2025
Version: 1.0

This module provides the LeadGenerator class, which is responsible for:
1.  Loading the official UK Government (GIAS) school database.
2.  Identifying "Trust Leads": Finding unvisited schools within the same
    Multi-Academy Trust (MAT) as a "champion" (repeat) school.
3.  Identifying "Hotspot Leads": Finding unvisited schools within a
    defined radius of a "champion" school.

This module is intended to be imported by the main master runner script.
"""

import pandas as pd
import numpy as np
from geopy.distance import great_circle
from pathlib import Path

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning, module='openpyxl')


class LeadGenerator:
    """
    Generates actionable leads by cross-referencing visited schools
    with the official UK Government (GIAS) school database.
    """
    
    def __init__(self, geocoded_breck_schools, gias_database_path, config):
        """
        Initializes the LeadGenerator.
        
        Args:
            geocoded_breck_schools (pd.DataFrame): The geocoded summary of Breck schools.
            gias_database_path (str or Path): Path to the GIAS establishments CSV file.
            config (object): A configuration object (or dict) containing parameters:
                                - champion_min_visits (int)
                                - hotspot_radius_km (float)
                                - lead_generation_file (str or Path)
        """
        print("LeadGenerator: Initializing...")
        self.breck_schools = geocoded_breck_schools
        self.gias_path = gias_database_path
        self.config = config
        self.gias_schools = self._load_gias_database()
        
        if self.gias_schools is None:
            print("LeadGenerator: !!! CRITICAL: GIAS database not loaded. Lead generation will fail.")

    def _load_gias_database(self):
        """Loads and cleans the GIAS (Get Information About Schools) database."""
        print(f"LeadGenerator: Loading GIAS database from '{self.gias_path}'...")
        try:
            # Try reading with utf-8, fallback to latin-1 which is common
            try:
                gias_df = pd.read_csv(self.gias_path, encoding='utf-8')
            except UnicodeDecodeError:
                gias_df = pd.read_csv(self.gias_path, encoding='latin-1')
            
            print(f"LeadGenerator: Successfully loaded {len(gias_df)} total establishments from GIAS.")
            # Flexible column detection: attempt to find common column name variants
            cols = list(gias_df.columns)

            def find_col(substrings):
                for s in substrings:
                    for c in cols:
                        if s.lower() in c.lower():
                            return c
                return None

            name_col = find_col(['establishment', 'establishmentname', 'establishment name', 'name'])
            status_col = find_col(['status'])
            type_col = find_col(['typeofestablishment', 'type of establishment', 'type'])
            trust_col = find_col(['trust', 'trusts'])
            lat_col = find_col(['latitude', 'lat'])
            lon_col = find_col(['longitude', 'long', 'lon'])
            urn_col = find_col(['urn'])
            postcode_col = find_col(['postcode', 'post code', 'post_code'])

            # Filter to Open establishments if status column exists
            if status_col and 'open' in gias_df[status_col].astype(str).str.lower().unique():
                try:
                    gias_df = gias_df[gias_df[status_col].astype(str).str.lower() == 'open']
                except Exception:
                    pass

            # Filter by relevant types if available (best-effort)
            relevant_types = [
                'primary', 'secondary', 'all-through', 'middle', 'special', 'pupil referral'
            ]
            if type_col:
                try:
                    mask = gias_df[type_col].astype(str).str.lower().str.contains('|'.join(relevant_types), na=False)
                    gias_df = gias_df[mask]
                except Exception:
                    pass

            # Build a cleaned dataframe with standardized column names where possible
            std_map = {}
            if urn_col: std_map[urn_col] = 'gias_urn'
            if name_col: std_map[name_col] = 'gias_name'
            if trust_col: std_map[trust_col] = 'trust_name'
            if lat_col: std_map[lat_col] = 'gias_lat'
            if lon_col: std_map[lon_col] = 'gias_lon'
            if postcode_col: std_map[postcode_col] = 'postcode'

            if not std_map:
                print('LeadGenerator: Warning - Could not detect any useful GIAS columns to map. Returning raw dataframe')
                return gias_df

            gias_std = gias_df[list(std_map.keys())].rename(columns=std_map)

            # If lat/lon missing entirely, warn but keep the data (lead generation will skip distance based hotspot matching)
            if 'gias_lat' not in gias_std.columns or 'gias_lon' not in gias_std.columns:
                print("LeadGenerator: Warning - Missing expected latitude/longitude columns in GIAS. Some lead types (hotspot) will be skipped.")

            # Normalize names
            if 'gias_name' in gias_std.columns:
                gias_std['__norm_name'] = gias_std['gias_name'].astype(str).str.lower().str.strip()

            print(f"LeadGenerator: {len(gias_std)} relevant schools processed from GIAS.")
            return gias_std
            
        except FileNotFoundError:
            print(f"LeadGenerator: !!! ERROR: GIAS database file not found at '{self.gias_path}'.")
            print("  Please download it from https://get-information-about-schools.service.gov.uk/Downloads")
            return None
        except Exception as e:
            print(f"LeadGenerator: !!! ERROR: Could not read GIAS file. {e}")
            return None

    def run_lead_generation(self):
        """Main function to generate and combine all lead types."""
        if self.gias_schools is None:
            return pd.DataFrame() # Return empty if GIAS data is missing

        print("LeadGenerator: Running lead generation...")
        
        # 1. Find "champion" schools (repeat bookers)
        champion_min_visits = self.config.get('champion_min_visits', 2)
        champions = self.breck_schools[
            self.breck_schools['visit_count'] >= champion_min_visits
        ].copy()
        print(f"LeadGenerator: Found {len(champions)} 'champion' schools (>= {champion_min_visits} visits).")

        # 2. Get list of all visited school names for filtering
        # We need to standardize names for matching
        visited_school_names = set(self.breck_schools['school_name'].str.lower().str.strip())
        
        # 3. Generate leads
        trust_leads = self._generate_trust_leads(champions, visited_school_names)
        hotspot_leads = self._generate_hotspot_leads(champions, visited_school_names)
        
        # 4. Combine and de-duplicate
        all_leads = pd.concat([trust_leads, hotspot_leads], ignore_index=True)
        
        # De-duplicate based on URN (unique school ID)
        all_leads = all_leads.drop_duplicates(subset=['gias_urn'])
        
        print(f"LeadGenerator: Total of {len(all_leads)} unique actionable leads generated.")
        
        # 5. Save to file
        output_file = self.config.get('lead_generation_file', 'outputs/actionable_school_leads.csv')
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        all_leads.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"LeadGenerator: Successfully saved leads to '{output_file}'")
        
        return all_leads

    def _generate_trust_leads(self, champions, visited_school_names):
        """Finds unvisited schools in the same trust as champion schools."""
        print("LeadGenerator: ...generating 'Trust' leads...")
        
        # We need to link our 'champions' (from Breck data) to the 'gias_schools'
        # to find their official trust names. A name-based join is unreliable.
        # A better method: merge on coordinates (robust)
        
        # For simplicity and speed, we'll try to find GIAS schools that share
        # a trust name with our geocoded champions.
        
        # This requires a spatial join or complex matching.
        # Let's try a simpler, good-enough-for-hackathon approach:
        # Match champions to GIAS by name (imperfect, but fast)
        
        # Better matching: map champion school names to GIAS entries using fuzzy matching
        champion_trusts = set()

        # Prepare list of GIAS names
        gias_names = []
        if self.gias_schools is not None and 'gias_name' in self.gias_schools.columns:
            gias_names = self.gias_schools['gias_name'].astype(str).tolist()

        # Try rapidfuzz for better matching; fall back to difflib
        try:
            from rapidfuzz import process as rf_process, fuzz as rf_fuzz
            use_rapidfuzz = True
        except Exception:
            import difflib
            use_rapidfuzz = False

        def best_match(name, choices, score_cutoff=80):
            if not name or not choices:
                return None, 0
            if use_rapidfuzz:
                match = rf_process.extractOne(str(name), choices, scorer=rf_fuzz.token_sort_ratio)
                if match and match[1] >= score_cutoff:
                    return match[0], match[1]
                return None, 0
            else:
                matches = difflib.get_close_matches(str(name), choices, n=1, cutoff=0.7)
                if matches:
                    score = int(difflib.SequenceMatcher(None, str(name), matches[0]).ratio() * 100)
                    if score >= score_cutoff:
                        return matches[0], score
                return None, 0

        for _, row in champions.iterrows():
            school_name = row.get('school_name')
            if not school_name:
                continue

            # Try exact match first
            if self.gias_schools is not None and 'gias_name' in self.gias_schools.columns:
                exact = self.gias_schools[self.gias_schools['gias_name'].astype(str).str.lower().str.strip() == str(school_name).lower().strip()]
                if len(exact) > 0 and 'trust_name' in exact.columns:
                    t = exact.iloc[0].get('trust_name')
                    if pd.notna(t):
                        champion_trusts.add(t)
                        continue

            # Fuzzy match
            match_name, score = best_match(school_name, gias_names, score_cutoff=80)
            if match_name:
                matched = self.gias_schools[self.gias_schools['gias_name'].astype(str) == match_name]
                if len(matched) > 0 and 'trust_name' in matched.columns:
                    t = matched.iloc[0].get('trust_name')
                    if pd.notna(t):
                        champion_trusts.add(t)

        if 'Not Applicable' in champion_trusts:
            champion_trusts.remove('Not Applicable')

        print(f"LeadGenerator: Found {len(champion_trusts)} champion trusts via name/fuzzy matching.")
        if not champion_trusts:
            print("LeadGenerator: No champion trusts found. Skipping trust leads.")
            return pd.DataFrame()

        # Find all schools in the GIAS database that are in one of these trusts
        trust_leads_df = self.gias_schools[
            self.gias_schools['trust_name'].isin(champion_trusts)
        ].copy()
        
        # Filter out schools we have already visited
        trust_leads_df = trust_leads_df[
            ~trust_leads_df['gias_name'].str.lower().str.strip().isin(visited_school_names)
        ]
        
        trust_leads_df['lead_reason'] = 'Trust Lead'
        trust_leads_df['lead_source'] = 'In same trust as a champion school'
        
        print(f"LeadGenerator: Found {len(trust_leads_df)} unvisited schools in champion trusts.")
        return trust_leads_df

    def _generate_hotspot_leads(self, champions, visited_school_names):
        """Finds unvisited schools within X km of champion schools."""
        print("LeadGenerator: ...generating 'Hotspot' leads...")
        
        # Ensure champions have valid coordinates
        champions_geo = champions.dropna(subset=['latitude', 'longitude'])
        if champions_geo.empty:
            print("LeadGenerator: No geocoded champion schools. Skipping hotspot leads.")
            return pd.DataFrame()

        # If we don't have any GIAS rows with coordinates, skip hotspot generation
        if self.gias_schools is None or len(self.gias_schools) == 0:
            print("LeadGenerator: No GIAS schools with coordinates available. Skipping hotspot leads.")
            return pd.DataFrame()

        hotspot_leads_indices = set()
        radius_km = self.config.get('hotspot_radius_km', 5.0)

        print(f"LeadGenerator: Calculating distances for {len(champions_geo)} champions against {len(self.gias_schools)} GIAS schools...")
        
        # Iterate over champions
        for i, champ_row in champions_geo.iterrows():
            champ_coord = (champ_row['latitude'], champ_row['longitude'])
            
            # Find distances to ALL GIAS schools at once (vectorized, but needs careful handling)
            # A simple loop is clearer and fine for this size
            for j, gias_row in self.gias_schools.iterrows():
                gias_coord = (gias_row['gias_lat'], gias_row['gias_lon'])
                
                try:
                    distance = great_circle(champ_coord, gias_coord).kilometers
                    if distance <= radius_km:
                        hotspot_leads_indices.add(j) # Add the *index* from gias_schools
                except Exception:
                    continue # Ignore invalid coordinate pairs
        
        # Get the unique GIAS schools from the identified indices
        hotspot_leads_df = self.gias_schools.loc[list(hotspot_leads_indices)].copy()
        
        # Filter out schools we have already visited
        hotspot_leads_df = hotspot_leads_df[
            ~hotspot_leads_df['gias_name'].str.lower().str.strip().isin(visited_school_names)
        ]
        
        hotspot_leads_df['lead_reason'] = 'Hotspot Lead'
        hotspot_leads_df['lead_source'] = f'Within {radius_km}km of a champion school'
        
        print(f"LeadGenerator: Found {len(hotspot_leads_df)} unvisited schools in geographic hotspots.")
        return hotspot_leads_df