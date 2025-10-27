"""
BRECK FOUNDATION - MODULE: ENHANCED LEAD GENERATOR V2
======================================================

Author: JP Morgan Data for Good Hackathon Team
Date: October 2025
Version: 2.0

This module provides TWO SEPARATE lead generation systems:

1. NORMAL LEADS (for fundraising):
   - Trust Leads: Schools in same MAT as champion schools
   - Hotspot Leads: Schools near champion schools geographically

2. DEPRIVED AREA LEADS (for social impact):
   - Schools in socio-economically disadvantaged areas (high FSM%)
   - Prioritizes maximum social impact

Each lead is scored and connected to the nearest champion school.
"""

import pandas as pd
import numpy as np
from geopy.distance import great_circle
from pathlib import Path
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning, module='openpyxl')


class EnhancedLeadGenerator:
    """
    Enhanced lead generator with dual-track system:
    - Normal leads for fundraising
    - Deprived area leads for impact
    """

    def __init__(self, geocoded_breck_schools, gias_database_path, config):
        """
        Initialize the enhanced lead generator.

        Args:
            geocoded_breck_schools (pd.DataFrame): Geocoded Breck schools with visit counts
            gias_database_path (str/Path): Path to GIAS establishments CSV
            config (dict): Configuration with:
                - champion_min_visits (int): Min visits to be a champion (default: 2)
                - hotspot_radius_km (float): Radius for hotspot leads (default: 5.0)
                - fsm_threshold (float): FSM% threshold for deprived areas (default: 30.0)
                - normal_leads_file (str): Output path for normal leads
                - deprived_leads_file (str): Output path for deprived leads
        """
        print("=" * 80)
        print("ENHANCED LEAD GENERATOR V2 - INITIALIZING")
        print("=" * 80)

        self.breck_schools = geocoded_breck_schools
        self.gias_path = gias_database_path
        self.config = config
        self.gias_schools = self._load_gias_database()

        if self.gias_schools is None:
            print("!!! CRITICAL: GIAS database not loaded. Lead generation will fail.")

    def _convert_bng_to_latlon(self, eastings, northings):
        """
        Convert British National Grid (Easting/Northing) to WGS84 (Lat/Lon).
        Uses approximate conversion formula.
        """
        import math

        lats = []
        lons = []

        for e, n in zip(eastings, northings):
            if pd.isna(e) or pd.isna(n):
                lats.append(None)
                lons.append(None)
                continue

            try:
                easting = float(e)
                northing = float(n)

                # Simplified formula for BNG to WGS84 conversion
                # This is approximate but sufficient for visualization
                # UK is roughly centered at 49N, -2W with scale ~111km per degree

                # Convert from BNG (meters) to lat/lon (degrees)
                # BNG false origin: Easting=400000, Northing=-100000
                lat_deg = 49.0 + (northing - (-100000)) / 111000.0
                lon_deg = -2.0 + (easting - 400000) / (111000.0 * math.cos(math.radians(lat_deg)))

                lats.append(lat_deg)
                lons.append(lon_deg)

            except Exception:
                lats.append(None)
                lons.append(None)

        return pd.Series(lats), pd.Series(lons)

    def _load_gias_database(self):
        """Load and clean the GIAS database with FSM data."""
        print(f"\nLoading GIAS database from: {self.gias_path}")

        try:
            # Try UTF-8, fallback to latin-1
            try:
                gias_df = pd.read_csv(self.gias_path, encoding='utf-8', low_memory=False)
            except UnicodeDecodeError:
                gias_df = pd.read_csv(self.gias_path, encoding='latin-1', low_memory=False)

            print(f"[OK] Loaded {len(gias_df):,} total establishments")

            # Flexible column detection
            cols = list(gias_df.columns)

            def find_col(substrings):
                for s in substrings:
                    for c in cols:
                        if s.lower() in c.lower():
                            return c
                return None

            name_col = find_col(['establishmentname', 'establishment name'])
            status_col = find_col(['status', 'establishmentstatus'])
            type_col = find_col(['typeofestablishment', 'type of establishment'])
            phase_col = find_col(['phaseofeducation', 'phase of education'])  # Primary/Secondary/etc
            print(f"[DEBUG] Phase column found: {phase_col}")
            trust_col = find_col(['trusts (name)', 'trust'])
            # GIAS uses Easting/Northing (British National Grid), not lat/lon directly
            easting_col = find_col(['easting'])
            northing_col = find_col(['northing'])
            urn_col = find_col(['urn'])
            postcode_col = find_col(['postcode'])
            # FSM can be in "PercentageFSM" or "FSM" column
            fsm_col = find_col(['percentagefsm', 'fsm'])

            # Filter to open establishments
            # Status code "1" = Open in GIAS
            if status_col:
                try:
                    # Try by code first (more reliable)
                    if 'EstablishmentStatus (code)' in gias_df.columns:
                        gias_df = gias_df[gias_df['EstablishmentStatus (code)'].astype(str) == '1']
                    else:
                        # Fallback to text search
                        gias_df = gias_df[gias_df[status_col].astype(str).str.lower().str.contains('open', na=False)]
                    print(f"[OK] Filtered to {len(gias_df):,} open establishments")
                except Exception as e:
                    print(f"[WARNING] Could not filter by status: {e}")

            # Filter to relevant phases (Primary, Secondary, etc)
            # Use PhaseOfEducation instead of TypeOfEstablishment
            # DON'T FILTER - keep all open schools regardless of phase
            # This ensures we don't lose schools due to unexpected phase values
            print(f"[OK] Keeping all {len(gias_df):,} open establishments (no phase filtering)")

            # Build standardized dataframe
            std_map = {}
            if urn_col: std_map[urn_col] = 'gias_urn'
            if name_col: std_map[name_col] = 'gias_name'
            if trust_col: std_map[trust_col] = 'trust_name'
            if easting_col: std_map[easting_col] = 'easting'
            if northing_col: std_map[northing_col] = 'northing'
            if postcode_col: std_map[postcode_col] = 'postcode'
            if fsm_col: std_map[fsm_col] = 'fsm_percentage'

            if not std_map:
                print("[WARNING] Warning: Could not detect GIAS columns")
                return gias_df

            # Select and rename columns
            available_cols = [c for c in std_map.keys() if c in gias_df.columns]
            gias_std = gias_df[available_cols].rename(columns=std_map).copy()

            # Convert FSM to numeric
            if 'fsm_percentage' in gias_std.columns:
                gias_std['fsm_percentage'] = pd.to_numeric(gias_std['fsm_percentage'], errors='coerce')
                print(f"[OK] FSM data available for {gias_std['fsm_percentage'].notna().sum():,} schools")

            # Convert Easting/Northing to Lat/Lon
            if 'easting' in gias_std.columns and 'northing' in gias_std.columns:
                gias_std['easting'] = pd.to_numeric(gias_std['easting'], errors='coerce')
                gias_std['northing'] = pd.to_numeric(gias_std['northing'], errors='coerce')

                # Convert British National Grid to WGS84 (lat/lon)
                gias_std['gias_lat'], gias_std['gias_lon'] = self._convert_bng_to_latlon(
                    gias_std['easting'], gias_std['northing']
                )
                print(f"[OK] Converted {gias_std['gias_lat'].notna().sum():,} coordinates from BNG to lat/lon")

            # Add normalized name for matching
            if 'gias_name' in gias_std.columns:
                gias_std['__norm_name'] = gias_std['gias_name'].astype(str).str.lower().str.strip()

            print(f"[OK] Processed {len(gias_std):,} schools ready for lead generation")
            return gias_std

        except FileNotFoundError:
            print(f"!!! ERROR: GIAS database not found at '{self.gias_path}'")
            print("Download from: https://get-information-about-schools.service.gov.uk/Downloads")
            return None
        except Exception as e:
            print(f"!!! ERROR: Could not load GIAS file: {e}")
            import traceback
            traceback.print_exc()
            return None

    def run_complete_lead_generation(self):
        """
        Run complete lead generation - generates TWO separate lists.

        Returns:
            tuple: (normal_leads_df, deprived_leads_df)
        """
        if self.gias_schools is None or len(self.gias_schools) == 0:
            print("Cannot generate leads - no GIAS data available")
            return pd.DataFrame(), pd.DataFrame()

        print("\n" + "=" * 80)
        print("GENERATING LEADS - DUAL TRACK SYSTEM")
        print("=" * 80)

        # Identify champion schools
        champion_min_visits = self.config.get('champion_min_visits', 2)
        champions = self.breck_schools[
            self.breck_schools['visit_count'] >= champion_min_visits
        ].copy()
        print(f"\n[OK] Found {len(champions)} champion schools (>={champion_min_visits} visits)")

        # Get visited school names
        visited_names = set(self.breck_schools['school_name'].str.lower().str.strip())

        # === TRACK 1: NORMAL LEADS (for fundraising) ===
        print("\n" + "-" * 80)
        print("TRACK 1: NORMAL LEADS (Trust + Hotspot)")
        print("-" * 80)

        trust_leads = self._generate_trust_leads(champions, visited_names)
        hotspot_leads = self._generate_hotspot_leads(champions, visited_names)

        # Combine and deduplicate normal leads
        normal_leads = pd.concat([trust_leads, hotspot_leads], ignore_index=True)
        if len(normal_leads) > 0 and 'gias_urn' in normal_leads.columns:
            normal_leads = normal_leads.drop_duplicates(subset=['gias_urn'])

        # Add champion connections to normal leads
        if len(normal_leads) > 0:
            normal_leads = self._add_champion_connections(normal_leads, champions)
            normal_leads = self._score_normal_leads(normal_leads)
            # Limit to top 100 for map performance
            normal_leads = normal_leads.head(100)

        print(f"\n[OK] Generated {len(normal_leads)} unique NORMAL LEADS (top 100 by score)")

        # === TRACK 2: DEPRIVED AREA LEADS (for impact) ===
        print("\n" + "-" * 80)
        print("TRACK 2: DEPRIVED AREA LEADS (High FSM%)")
        print("-" * 80)

        deprived_leads = self._generate_deprived_area_leads(visited_names)

        # Add champion connections to deprived leads
        if len(deprived_leads) > 0:
            deprived_leads = self._add_champion_connections(deprived_leads, champions)
            deprived_leads = self._score_deprived_leads(deprived_leads)
            # Limit to top 100 for map performance
            deprived_leads = deprived_leads.head(100)

        print(f"\n[OK] Generated {len(deprived_leads)} unique DEPRIVED AREA LEADS (top 100 by impact)")

        # Save both lists
        self._save_leads(normal_leads, deprived_leads)

        return normal_leads, deprived_leads

    def _generate_trust_leads(self, champions, visited_names):
        """Generate leads from schools in same trusts as champions."""
        print("\n  > Generating Trust Leads...")

        if 'trust_name' not in self.gias_schools.columns:
            print("    [WARNING] No trust data available")
            return pd.DataFrame()

        # Match champions to GIAS to find their trusts
        champion_trusts = set()

        # Try fuzzy matching
        try:
            from rapidfuzz import process as rf_process, fuzz as rf_fuzz
            use_rapidfuzz = True
        except ImportError:
            import difflib
            use_rapidfuzz = False

        gias_names = self.gias_schools['gias_name'].astype(str).tolist() if 'gias_name' in self.gias_schools.columns else []

        for _, champ in champions.iterrows():
            school_name = champ.get('school_name')
            if not school_name:
                continue

            # Try exact match
            exact = self.gias_schools[
                self.gias_schools.get('__norm_name', pd.Series()) == str(school_name).lower().strip()
            ]
            if len(exact) > 0:
                trust = exact.iloc[0].get('trust_name')
                if pd.notna(trust) and trust != 'Not applicable':
                    champion_trusts.add(trust)
                    continue

            # Fuzzy match
            if use_rapidfuzz and gias_names:
                match = rf_process.extractOne(str(school_name), gias_names, scorer=rf_fuzz.token_sort_ratio)
                if match and match[1] >= 80:
                    matched = self.gias_schools[self.gias_schools['gias_name'].astype(str) == match[0]]
                    if len(matched) > 0:
                        trust = matched.iloc[0].get('trust_name')
                        if pd.notna(trust) and trust != 'Not applicable':
                            champion_trusts.add(trust)

        print(f"    [OK] Identified {len(champion_trusts)} champion trusts")

        if not champion_trusts:
            return pd.DataFrame()

        # Find unvisited schools in these trusts
        trust_leads = self.gias_schools[
            self.gias_schools['trust_name'].isin(champion_trusts)
        ].copy()

        trust_leads = trust_leads[
            ~trust_leads.get('__norm_name', pd.Series()).isin(visited_names)
        ]

        trust_leads['lead_type'] = 'Trust Lead'
        trust_leads['lead_category'] = 'normal'
        trust_leads['lead_source'] = 'Same trust as champion school'

        print(f"    [OK] Found {len(trust_leads)} trust leads")
        return trust_leads

    def _generate_hotspot_leads(self, champions, visited_names):
        """Generate leads from schools near champion schools."""
        print("\n  > Generating Hotspot Leads...")

        # Filter champions with coordinates
        champions_geo = champions.dropna(subset=['latitude', 'longitude'])
        if len(champions_geo) == 0:
            print("    [WARNING] No geocoded champions")
            return pd.DataFrame()

        # Filter GIAS schools with coordinates
        gias_geo = self.gias_schools.dropna(subset=['gias_lat', 'gias_lon'])
        if len(gias_geo) == 0:
            print("    [WARNING] No geocoded GIAS schools")
            return pd.DataFrame()

        radius_km = self.config.get('hotspot_radius_km', 5.0)
        hotspot_indices = set()

        print(f"    > Searching within {radius_km}km radius...")

        for _, champ in champions_geo.iterrows():
            champ_coord = (champ['latitude'], champ['longitude'])

            for idx, gias in gias_geo.iterrows():
                try:
                    gias_coord = (gias['gias_lat'], gias['gias_lon'])
                    distance = great_circle(champ_coord, gias_coord).kilometers

                    if distance <= radius_km:
                        hotspot_indices.add(idx)
                except Exception:
                    continue

        hotspot_leads = self.gias_schools.loc[list(hotspot_indices)].copy()
        hotspot_leads = hotspot_leads[
            ~hotspot_leads.get('__norm_name', pd.Series()).isin(visited_names)
        ]

        hotspot_leads['lead_type'] = 'Hotspot Lead'
        hotspot_leads['lead_category'] = 'normal'
        hotspot_leads['lead_source'] = f'Within {radius_km}km of champion'

        print(f"    [OK] Found {len(hotspot_leads)} hotspot leads")
        return hotspot_leads

    def _generate_deprived_area_leads(self, visited_names):
        """Generate leads from socio-economically deprived areas (high FSM%)."""
        print("\n  > Generating Deprived Area Leads...")

        if 'fsm_percentage' not in self.gias_schools.columns:
            print("    [WARNING] No FSM data available")
            return pd.DataFrame()

        fsm_threshold = self.config.get('fsm_threshold', 30.0)

        # Find schools with high FSM%
        deprived = self.gias_schools[
            self.gias_schools['fsm_percentage'] >= fsm_threshold
        ].copy()

        # Filter out visited schools
        deprived = deprived[
            ~deprived.get('__norm_name', pd.Series()).isin(visited_names)
        ]

        deprived['lead_type'] = 'Deprived Area'
        deprived['lead_category'] = 'deprived'
        deprived['lead_source'] = f'FSM >= {fsm_threshold}% (high deprivation)'

        print(f"    [OK] Found {len(deprived)} schools with FSM% >= {fsm_threshold}%")

        return deprived

    def _add_champion_connections(self, leads, champions):
        """Add information about nearest champion school to each lead."""
        print("\n  > Connecting leads to nearest champions...")

        # Filter champions with coordinates
        champions_geo = champions.dropna(subset=['latitude', 'longitude'])
        if len(champions_geo) == 0:
            leads['nearest_champion'] = None
            leads['distance_to_champion'] = None
            return leads

        # Filter leads with coordinates
        leads_geo = leads.dropna(subset=['gias_lat', 'gias_lon'])

        nearest_champions = []
        distances = []

        for idx, lead in leads.iterrows():
            if pd.isna(lead.get('gias_lat')) or pd.isna(lead.get('gias_lon')):
                nearest_champions.append(None)
                distances.append(None)
                continue

            lead_coord = (lead['gias_lat'], lead['gias_lon'])

            min_dist = float('inf')
            nearest = None

            for _, champ in champions_geo.iterrows():
                try:
                    champ_coord = (champ['latitude'], champ['longitude'])
                    dist = great_circle(lead_coord, champ_coord).kilometers

                    if dist < min_dist:
                        min_dist = dist
                        nearest = champ['school_name']
                except Exception:
                    continue

            nearest_champions.append(nearest)
            distances.append(min_dist if min_dist != float('inf') else None)

        leads['nearest_champion'] = nearest_champions
        leads['distance_to_champion_km'] = distances

        connected = sum(1 for n in nearest_champions if n is not None)
        print(f"    [OK] Connected {connected}/{len(leads)} leads to champions")

        return leads

    def _score_normal_leads(self, leads):
        """Score normal leads based on priority factors."""
        leads['priority_score'] = 0

        # Trust leads get highest priority
        leads.loc[leads['lead_type'] == 'Trust Lead', 'priority_score'] += 10

        # Closer to champions = higher priority
        if 'distance_to_champion_km' in leads.columns:
            # Normalize distance score (0-5 points)
            max_dist = leads['distance_to_champion_km'].max()
            if max_dist > 0:
                leads['priority_score'] += 5 * (1 - leads['distance_to_champion_km'] / max_dist)

        # Sort by score
        leads = leads.sort_values('priority_score', ascending=False)

        return leads

    def _score_deprived_leads(self, leads):
        """Score deprived area leads based on impact factors."""
        leads['impact_score'] = 0

        # Higher FSM% = higher impact score
        if 'fsm_percentage' in leads.columns:
            max_fsm = leads['fsm_percentage'].max()
            if max_fsm > 0:
                leads['impact_score'] += 10 * (leads['fsm_percentage'] / max_fsm)

        # Sort by score
        leads = leads.sort_values('impact_score', ascending=False)

        return leads

    def _save_leads(self, normal_leads, deprived_leads):
        """Save both lead lists to CSV files."""
        print("\n" + "=" * 80)
        print("SAVING LEAD LISTS")
        print("=" * 80)

        # Normal leads
        normal_file = self.config.get('normal_leads_file', 'outputs/normal_leads.csv')
        Path(normal_file).parent.mkdir(parents=True, exist_ok=True)

        if len(normal_leads) > 0:
            normal_leads.to_csv(normal_file, index=False, encoding='utf-8-sig')
            print(f"\n[OK] Normal leads saved: {normal_file}")
            print(f"  > {len(normal_leads)} leads for fundraising outreach")

        # Deprived area leads
        deprived_file = self.config.get('deprived_leads_file', 'outputs/deprived_area_leads.csv')

        if len(deprived_leads) > 0:
            deprived_leads.to_csv(deprived_file, index=False, encoding='utf-8-sig')
            print(f"\n[OK] Deprived area leads saved: {deprived_file}")
            print(f"  > {len(deprived_leads)} leads for social impact")

        print("\n" + "=" * 80)


# Helper function for backward compatibility
def run_enhanced_lead_generation(breck_schools, gias_path, config):
    """
    Run enhanced lead generation.

    Returns:
        tuple: (normal_leads_df, deprived_leads_df)
    """
    generator = EnhancedLeadGenerator(breck_schools, gias_path, config)
    return generator.run_complete_lead_generation()
