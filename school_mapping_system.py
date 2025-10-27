"""
BRECK FOUNDATION - SCHOOL MAPPING SYSTEM
=========================================
Complete system for mapping schools, identifying repeat visits,
and grouping schools by trusts/organizations.

Author: Hackathon Team
Date: October 2025
"""

import pandas as pd
import numpy as np
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import folium
from folium.plugins import MarkerCluster, HeatMap
import time
import json
from datetime import datetime
from collections import Counter, defaultdict
import re

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    'data_file': r'C:\Users\Paul\OneDrive - telecom-paristech.fr\Documents\Pro\Hackathon\JPMorgan Data for Good\V2\D4G Data - Lou working doc.xlsx',
    'output_dir': r'C:\Users\Paul\OneDrive - telecom-paristech.fr\Documents\Pro\Hackathon\JPMorgan Data for Good\V2\outputs',
    'geocode_cache_file': r'C:\Users\Paul\OneDrive - telecom-paristech.fr\Documents\Pro\Hackathon\JPMorgan Data for Good\V2\outputs\geocode_cache.json'
}


# ============================================================================
# SCHOOL DATA EXTRACTOR
# ============================================================================

class SchoolDataExtractor:
    """Extract and clean school data from Excel file."""

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.schools_df = None

    def load_and_clean(self):
        """Load and clean the school booking data."""
        print("=" * 80)
        print("LOADING SCHOOL BOOKING DATA")
        print("=" * 80)

        # Read first sheet
        df = pd.read_excel(self.file_path, sheet_name='Booking Form Data Direct Data')

        print(f"\nRaw data: {df.shape}")

        # Clean column names (row 1 contains headers)
        df.columns = df.iloc[1]
        df = df.iloc[2:].reset_index(drop=True)

        # Keep only relevant columns
        df = df[['Submission Date', 'Your Name', 'School Name',
                 'How did you hear about our talks?', 'Answer']].copy()

        # Convert submission date
        df['Submission Date'] = pd.to_datetime(df['Submission Date'], errors='coerce')

        # Clean school names
        df['School Name'] = df['School Name'].str.strip()
        df = df[df['School Name'].notna()].copy()

        # Add year column
        df['Year'] = df['Submission Date'].dt.year

        print(f"Cleaned data: {len(df)} bookings")
        print(f"Unique schools: {df['School Name'].nunique()}")
        print(f"Date range: {df['Submission Date'].min()} to {df['Submission Date'].max()}")

        self.schools_df = df
        return self

    def analyze_school_visits(self):
        """Analyze which schools have multiple visits."""
        print("\n" + "=" * 80)
        print("SCHOOL VISIT ANALYSIS")
        print("=" * 80)

        # Count visits per school
        visit_counts = self.schools_df['School Name'].value_counts()

        print(f"\nTotal unique schools: {len(visit_counts)}")
        print(f"Schools with 1 visit: {(visit_counts == 1).sum()}")
        print(f"Schools with 2+ visits: {(visit_counts >= 2).sum()}")
        print(f"Schools with 3+ visits: {(visit_counts >= 3).sum()}")

        # Show top repeat schools
        print("\nTop 10 schools by number of visits:")
        for school, count in visit_counts.head(10).items():
            print(f"  {school}: {count} visits")

        # Create summary dataframe
        school_summary = pd.DataFrame({
            'school_name': visit_counts.index,
            'visit_count': visit_counts.values
        })

        # Add first and last visit dates
        first_visits = self.schools_df.groupby('School Name')['Submission Date'].min()
        last_visits = self.schools_df.groupby('School Name')['Submission Date'].max()

        school_summary['first_visit'] = school_summary['school_name'].map(first_visits)
        school_summary['last_visit'] = school_summary['school_name'].map(last_visits)

        # Add how they heard about us (most recent)
        most_recent = self.schools_df.sort_values('Submission Date').groupby('School Name').last()
        school_summary['how_heard'] = school_summary['school_name'].map(most_recent['Answer'])
        school_summary['contact_person'] = school_summary['school_name'].map(most_recent['Your Name'])

        return school_summary

    def identify_repeat_bookings(self):
        """Identify which bookings were repeats."""
        repeat_mask = self.schools_df['Answer'] == 'Repeat Booking'

        print("\n" + "=" * 80)
        print("REPEAT BOOKING ANALYSIS")
        print("=" * 80)

        print(f"\nTotal bookings: {len(self.schools_df)}")
        print(f"Repeat bookings: {repeat_mask.sum()} ({repeat_mask.sum()/len(self.schools_df)*100:.1f}%)")
        print(f"New bookings: {(~repeat_mask).sum()} ({(~repeat_mask).sum()/len(self.schools_df)*100:.1f}%)")

        # Breakdown by source
        print("\nBooking sources:")
        source_counts = self.schools_df['Answer'].value_counts()
        for source, count in source_counts.items():
            pct = count / len(self.schools_df) * 100
            print(f"  {source}: {count} ({pct:.1f}%)")

        return self.schools_df


# ============================================================================
# SCHOOL GEOCODER
# ============================================================================

class SchoolGeocoder:
    """Geocode schools using Nominatim (OpenStreetMap)."""

    def __init__(self, cache_file: str = None):
        self.cache_file = cache_file
        self.cache = self._load_cache()
        self.geolocator = Nominatim(user_agent="breck_foundation_mapping", timeout=10)
        # Add rate limiting: 1 request per second for Nominatim
        self.geocode = RateLimiter(self.geolocator.geocode, min_delay_seconds=1.5)

    def _load_cache(self):
        """Load geocoding cache from file."""
        if self.cache_file:
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return {}
        return {}

    def _save_cache(self):
        """Save geocoding cache to file."""
        if self.cache_file:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, indent=2)

    def geocode_school(self, school_name: str):
        """
        Geocode a single school.

        Args:
            school_name: Name of the school

        Returns:
            dict with lat, lon, full_address, or None if not found
        """
        # Check cache first
        if school_name in self.cache:
            return self.cache[school_name]

        # Try different search queries
        queries = [
            f"{school_name}, UK",
            f"{school_name}, England",
            f"{school_name} School, UK"
        ]

        for query in queries:
            try:
                location = self.geocode(query)
                if location:
                    result = {
                        'lat': location.latitude,
                        'lon': location.longitude,
                        'address': location.address,
                        'query_used': query
                    }
                    self.cache[school_name] = result
                    self._save_cache()
                    return result
            except Exception as e:
                print(f"  Error geocoding {school_name}: {e}")
                continue

        # Not found
        self.cache[school_name] = None
        self._save_cache()
        return None

    def geocode_schools(self, school_names: list):
        """
        Geocode multiple schools.

        Args:
            school_names: List of school names

        Returns:
            DataFrame with school names and coordinates
        """
        print("\n" + "=" * 80)
        print("GEOCODING SCHOOLS")
        print("=" * 80)

        results = []
        total = len(school_names)

        for i, school in enumerate(school_names, 1):
            if i % 10 == 0 or i == total:
                print(f"  Progress: {i}/{total} ({i/total*100:.1f}%)")

            geo_data = self.geocode_school(school)

            if geo_data:
                results.append({
                    'school_name': school,
                    'latitude': geo_data['lat'],
                    'longitude': geo_data['lon'],
                    'address': geo_data['address'],
                    'geocoded': True
                })
            else:
                results.append({
                    'school_name': school,
                    'latitude': None,
                    'longitude': None,
                    'address': None,
                    'geocoded': False
                })

        df = pd.DataFrame(results)

        success_rate = df['geocoded'].sum() / len(df) * 100
        print(f"\nGeocoding complete:")
        print(f"  Successfully geocoded: {df['geocoded'].sum()}/{len(df)} ({success_rate:.1f}%)")
        print(f"  Failed: {(~df['geocoded']).sum()}")

        return df


# ============================================================================
# SCHOOL TRUST IDENTIFIER
# ============================================================================

class SchoolTrustIdentifier:
    """Identify schools belonging to same trusts/groups."""

    # Common patterns in UK school trusts
    TRUST_PATTERNS = [
        r'(.*?)\s+(Multi[- ]Academy Trust|MAT|Trust|Education Trust|Academy Trust)',
        r'(.*?)\s+(Schools?)',
        r'(.*?)\s+(Partnership|Federation|Collaboration)',
    ]

    def __init__(self):
        self.trust_groups = defaultdict(list)

    def identify_trusts(self, school_names: list):
        """
        Identify potential school trusts/groups.

        Args:
            school_names: List of school names

        Returns:
            Dictionary mapping trust names to school lists
        """
        print("\n" + "=" * 80)
        print("IDENTIFYING SCHOOL TRUSTS/GROUPS")
        print("=" * 80)

        # Group schools by common keywords
        keywords = defaultdict(list)

        for school in school_names:
            # Extract potential trust names
            words = school.lower().split()

            # Look for trust-related keywords
            if 'academy' in school.lower():
                # Extract academy name without "Primary", "Secondary", etc.
                clean_name = school.lower()
                for suffix in ['primary', 'secondary', 'junior', 'infant', 'school']:
                    clean_name = clean_name.replace(suffix, '').strip()
                keywords['academy_' + clean_name].append(school)

            # Look for common geographic or organizational names
            for word in words:
                if len(word) > 4 and word not in ['school', 'primary', 'secondary', 'academy',
                                                    'junior', 'infant', 'church']:
                    keywords[word].append(school)

        # Filter to groups with 2+ schools
        potential_groups = {k: v for k, v in keywords.items() if len(v) >= 2}

        print(f"\nFound {len(potential_groups)} potential school groups:")
        for group_name, schools in sorted(potential_groups.items(), key=lambda x: len(x[1]), reverse=True)[:10]:
            print(f"  {group_name}: {len(schools)} schools")
            for school in schools[:3]:
                print(f"    - {school}")
            if len(schools) > 3:
                print(f"    ... and {len(schools)-3} more")

        return potential_groups


# ============================================================================
# INTERACTIVE MAP CREATOR
# ============================================================================

class SchoolMapCreator:
    """Create interactive maps of schools."""

    def __init__(self):
        self.map = None

    def create_map(self, schools_data: pd.DataFrame, output_file: str):
        """
        Create interactive map of schools with visit counts.

        Args:
            schools_data: DataFrame with school info including lat/lon and visit counts
            output_file: Path to save HTML map
        """
        print("\n" + "=" * 80)
        print("CREATING INTERACTIVE MAP")
        print("=" * 80)

        # Filter to geocoded schools only
        geocoded = schools_data[schools_data['geocoded'] == True].copy()

        if len(geocoded) == 0:
            print("No geocoded schools to map!")
            return None

        # Calculate center of map (UK center)
        center_lat = geocoded['latitude'].mean()
        center_lon = geocoded['longitude'].mean()

        # Create base map
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=7,
            tiles='OpenStreetMap'
        )

        # Add title
        title_html = '''
        <div style="position: fixed;
                    top: 10px; left: 50px; width: 400px; height: 90px;
                    background-color: white; border:2px solid grey; z-index:9999;
                    font-size:14px; padding: 10px">
        <h4>Breck Foundation - School Interventions Map</h4>
        <p><span style="color: green;">●</span> Single visit |
           <span style="color: orange;">●</span> 2-3 visits |
           <span style="color: red;">●</span> 4+ visits</p>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(title_html))

        # Add markers for each school
        for idx, row in geocoded.iterrows():
            # Determine color based on visit count
            visit_count = row.get('visit_count', 1)
            if visit_count == 1:
                color = 'green'
                icon = 'info-sign'
            elif visit_count <= 3:
                color = 'orange'
                icon = 'star'
            else:
                color = 'red'
                icon = 'star'

            # Create popup text
            popup_text = f"""
            <b>{row['school_name']}</b><br>
            Visits: {visit_count}<br>
            First visit: {row.get('first_visit', 'Unknown')}<br>
            Last visit: {row.get('last_visit', 'Unknown')}<br>
            How heard: {row.get('how_heard', 'Unknown')}<br>
            Contact: {row.get('contact_person', 'Unknown')}
            """

            folium.Marker(
                location=[row['latitude'], row['longitude']],
                popup=folium.Popup(popup_text, max_width=300),
                tooltip=f"{row['school_name']} ({visit_count} visits)",
                icon=folium.Icon(color=color, icon=icon)
            ).add_to(m)

        # Save map
        m.save(output_file)

        print(f"\nInteractive map created: {output_file}")
        print(f"Mapped {len(geocoded)} schools")
        print(f"  Single visit: {(geocoded['visit_count'] == 1).sum()}")
        print(f"  2-3 visits: {((geocoded['visit_count'] >= 2) & (geocoded['visit_count'] <= 3)).sum()}")
        print(f"  4+ visits: {(geocoded['visit_count'] >= 4).sum()}")

        return output_file


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    print("=" * 80)
    print("BRECK FOUNDATION - SCHOOL MAPPING SYSTEM")
    print("=" * 80)
    print(f"\nTimestamp: {datetime.now().isoformat()}")
    print()

    # 1. Extract and analyze school data
    print("\n[STEP 1] EXTRACTING SCHOOL DATA")
    print("-" * 80)
    extractor = SchoolDataExtractor(CONFIG['data_file'])
    extractor.load_and_clean()
    extractor.identify_repeat_bookings()
    school_summary = extractor.analyze_school_visits()

    # Save school summary
    school_summary.to_csv(f"{CONFIG['output_dir']}/school_summary.csv", index=False)
    print(f"\nSchool summary saved to: {CONFIG['output_dir']}/school_summary.csv")

    # 2. Geocode schools
    print("\n[STEP 2] GEOCODING SCHOOLS")
    print("-" * 80)
    geocoder = SchoolGeocoder(cache_file=CONFIG['geocode_cache_file'])
    geocoded_schools = geocoder.geocode_schools(school_summary['school_name'].tolist())

    # Merge geocoded data with school summary
    full_data = school_summary.merge(geocoded_schools, on='school_name', how='left')

    # Save full dataset
    full_data.to_csv(f"{CONFIG['output_dir']}/schools_with_geocoding.csv", index=False)
    print(f"\nFull school data saved to: {CONFIG['output_dir']}/schools_with_geocoding.csv")

    # 3. Identify school trusts/groups
    print("\n[STEP 3] IDENTIFYING SCHOOL GROUPS")
    print("-" * 80)
    trust_identifier = SchoolTrustIdentifier()
    trust_groups = trust_identifier.identify_trusts(school_summary['school_name'].tolist())

    # Save trust groups
    with open(f"{CONFIG['output_dir']}/school_groups.json", 'w', encoding='utf-8') as f:
        json.dump(trust_groups, f, indent=2, ensure_ascii=False)
    print(f"\nSchool groups saved to: {CONFIG['output_dir']}/school_groups.json")

    # 4. Create interactive map
    print("\n[STEP 4] CREATING INTERACTIVE MAP")
    print("-" * 80)
    map_creator = SchoolMapCreator()
    map_file = f"{CONFIG['output_dir']}/breck_schools_map.html"
    map_creator.create_map(full_data, map_file)

    # Final summary
    print("\n" + "=" * 80)
    print("MAPPING COMPLETE!")
    print("=" * 80)
    print("\nOutput files:")
    print(f"  1. School summary: {CONFIG['output_dir']}/school_summary.csv")
    print(f"  2. Geocoded data: {CONFIG['output_dir']}/schools_with_geocoding.csv")
    print(f"  3. School groups: {CONFIG['output_dir']}/school_groups.json")
    print(f"  4. Interactive map: {CONFIG['output_dir']}/breck_schools_map.html")
    print(f"  5. Geocoding cache: {CONFIG['geocode_cache_file']}")

    print("\nNext steps:")
    print("  1. Open breck_schools_map.html in a web browser")
    print("  2. Review schools_with_geocoding.csv for manual corrections")
    print("  3. Use school_groups.json to identify contact opportunities")

    return full_data, trust_groups


if __name__ == "__main__":
    results = main()
