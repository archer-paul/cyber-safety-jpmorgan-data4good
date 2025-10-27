"""
ENHANCED SCHOOL GEOCODER
=========================
Uses multiple geocoding services for better accuracy:
1. Nominatim (OpenStreetMap) - Free
2. UK Government School Database lookup
3. Google-style search queries

This provides better geocoding for UK schools.
"""

import pandas as pd
import requests
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import time
import json
from pathlib import Path


class EnhancedSchoolGeocoder:
    """Enhanced geocoder specifically for UK schools."""

    def __init__(self, cache_file: str = None):
        self.cache_file = cache_file
        self.cache = self._load_cache()
        self.geolocator = Nominatim(
            user_agent="breck_foundation_school_mapper_v2",
            timeout=10
        )
        self.geocode = RateLimiter(self.geolocator.geocode, min_delay_seconds=1.5)
        self.session = requests.Session()

    def _load_cache(self):
        """Load geocoding cache."""
        if self.cache_file and Path(self.cache_file).exists():
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return {}
        return {}

    def _save_cache(self):
        """Save geocoding cache."""
        if self.cache_file:
            Path(self.cache_file).parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, indent=2, ensure_ascii=False)

    def _clean_school_name(self, name: str) -> str:
        """Clean school name for better geocoding."""
        # Remove special characters
        clean = name.strip()
        # Remove common suffixes that might confuse geocoder
        suffixes = ['Primary School', 'Secondary School', 'Junior School',
                    'Infant School', 'Academy', 'School']
        for suffix in suffixes:
            if clean.endswith(suffix):
                # Keep the suffix but also try without
                return clean
        return clean

    def geocode_with_nominatim(self, school_name: str) -> dict:
        """Geocode using Nominatim with multiple search strategies."""

        search_queries = [
            # Most specific first
            f"{school_name}, United Kingdom",
            f"{school_name}, England",
            f"{school_name} School, UK",
            f"{school_name}, School, United Kingdom",
        ]

        for query in search_queries:
            try:
                location = self.geocode(query, country_codes='gb')
                if location:
                    # Check if it's actually a school or educational institution
                    if any(word in location.address.lower() for word in
                           ['school', 'academy', 'college', 'education']):
                        return {
                            'lat': location.latitude,
                            'lon': location.longitude,
                            'address': location.address,
                            'source': 'nominatim',
                            'query': query,
                            'confidence': 'high'
                        }
                    else:
                        return {
                            'lat': location.latitude,
                            'lon': location.longitude,
                            'address': location.address,
                            'source': 'nominatim',
                            'query': query,
                            'confidence': 'medium'
                        }
            except Exception as e:
                continue

        return None

    def search_uk_school_database(self, school_name: str) -> dict:
        """
        Search for school in UK government database.
        Note: This is a placeholder - you would need to implement
        actual API calls to Get Information About Schools (GIAS) database.
        """
        # TODO: Implement GIAS API search if available
        # For now, return None
        return None

    def geocode_school(self, school_name: str, verbose: bool = False) -> dict:
        """
        Geocode a school using multiple methods.

        Returns:
            dict with geocoding result or None
        """
        # Check cache
        if school_name in self.cache:
            if verbose:
                print(f"  [CACHE] {school_name}")
            return self.cache[school_name]

        if verbose:
            print(f"  [GEOCODING] {school_name}...")

        # Try Nominatim first
        result = self.geocode_with_nominatim(school_name)

        if result:
            if verbose:
                print(f"    ✓ Found: {result['address'][:60]}...")
        else:
            if verbose:
                print(f"    ✗ Not found")

        # Cache result (even if None)
        self.cache[school_name] = result
        self._save_cache()

        return result

    def geocode_dataframe(self, df: pd.DataFrame, school_col: str = 'school_name') -> pd.DataFrame:
        """
        Geocode all schools in a dataframe.

        Args:
            df: DataFrame with school names
            school_col: Name of column containing school names

        Returns:
            DataFrame with added geocoding columns
        """
        results = []

        print(f"\nGeocoding {len(df)} schools...")
        print("-" * 60)

        for idx, row in df.iterrows():
            school_name = row[school_col]

            if idx > 0 and idx % 10 == 0:
                print(f"Progress: {idx}/{len(df)} ({idx/len(df)*100:.1f}%)")

            geo_result = self.geocode_school(school_name, verbose=False)

            row_result = row.to_dict()

            if geo_result:
                row_result['latitude'] = geo_result['lat']
                row_result['longitude'] = geo_result['lon']
                row_result['geocoded_address'] = geo_result['address']
                row_result['geocode_source'] = geo_result['source']
                row_result['geocode_confidence'] = geo_result.get('confidence', 'medium')
                row_result['geocoded'] = True
            else:
                row_result['latitude'] = None
                row_result['longitude'] = None
                row_result['geocoded_address'] = None
                row_result['geocode_source'] = None
                row_result['geocode_confidence'] = None
                row_result['geocoded'] = False

            results.append(row_result)

        result_df = pd.DataFrame(results)

        # Summary
        success = result_df['geocoded'].sum()
        total = len(result_df)
        print(f"\nGeocoding complete:")
        print(f"  Success: {success}/{total} ({success/total*100:.1f}%)")
        print(f"  Failed: {total-success}")

        if 'geocode_confidence' in result_df.columns:
            print(f"\nConfidence levels:")
            conf_counts = result_df['geocode_confidence'].value_counts()
            for conf, count in conf_counts.items():
                if conf:
                    print(f"  {conf}: {count}")

        return result_df


# Test function
def test_geocoder():
    """Test the geocoder with sample schools."""
    print("=" * 80)
    print("TESTING ENHANCED GEOCODER")
    print("=" * 80)

    test_schools = [
        "Springfield Primary School",
        "Dover Grammar School for Boys",
        "Baring Primary School",
        "Langshott Primary School"
    ]

    geocoder = EnhancedSchoolGeocoder(
        cache_file=r'C:\Users\Paul\OneDrive - telecom-paristech.fr\Documents\Pro\Hackathon\JPMorgan Data for Good\V2\outputs\geocode_test_cache.json'
    )

    print("\nTesting with sample schools:")
    print("-" * 80)

    for school in test_schools:
        result = geocoder.geocode_school(school, verbose=True)
        if result:
            print(f"    Coords: ({result['lat']:.4f}, {result['lon']:.4f})")
        print()


if __name__ == "__main__":
    test_geocoder()
