"""
GEOGRAPHIC CLUSTERING & VISUALIZATION MODULE - Breck Foundation
================================================================
Geographic analysis and clustering of workshop locations to:
1. Visualize geographic reach
2. Identify underserved areas
3. Optimize resource allocation
4. Create interactive maps

Author: Hackathon Team
Date: October 2025
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# GEOCODING UTILITIES
# ============================================================================

class SchoolGeocoder:
    """
    Geocode school names and postcodes to lat/long coordinates.
    """
    
    def __init__(self):
        """Initialize geocoder with UK-specific settings."""
        self.geocoded_schools = {}
        self.failed_geocodes = []
    
    def geocode_school(self, school_name: str, postcode: Optional[str] = None) -> Optional[Dict]:
        """
        Geocode a single school.
        
        Args:
            school_name: Name of the school
            postcode: UK postcode (if available)
            
        Returns:
            Dictionary with lat, lon, and metadata
        """
        # Check if already geocoded
        cache_key = f"{school_name}_{postcode}"
        if cache_key in self.geocoded_schools:
            return self.geocoded_schools[cache_key]
        
        try:
            # Try geocoding with geopy (if available)
            try:
                from geopy.geocoders import Nominatim
                from geopy.exc import GeocoderTimedOut, GeocoderServiceError
                
                geolocator = Nominatim(user_agent="breck_foundation_analysis", timeout=10)
                
                # Build query
                if postcode:
                    query = f"{school_name}, {postcode}, United Kingdom"
                else:
                    query = f"{school_name}, United Kingdom"
                
                location = geolocator.geocode(query)
                
                if location:
                    result = {
                        'school_name': school_name,
                        'postcode': postcode,
                        'latitude': location.latitude,
                        'longitude': location.longitude,
                        'address': location.address,
                        'source': 'nominatim'
                    }
                    self.geocoded_schools[cache_key] = result
                    return result
                else:
                    self.failed_geocodes.append(cache_key)
                    return None
                    
            except (ImportError, GeocoderTimedOut, GeocoderServiceError) as e:
                print(f"Geocoding service error: {e}")
                self.failed_geocodes.append(cache_key)
                return None
                
        except Exception as e:
            print(f"Error geocoding {school_name}: {e}")
            self.failed_geocodes.append(cache_key)
            return None
    
    def geocode_schools_batch(self, schools: List[str], 
                             postcodes: Optional[List[str]] = None,
                             delay: float = 1.0) -> pd.DataFrame:
        """
        Geocode multiple schools with rate limiting.
        
        Args:
            schools: List of school names
            postcodes: Optional list of postcodes
            delay: Delay between requests (seconds)
            
        Returns:
            DataFrame with geocoded results
        """
        import time
        
        print(f"Geocoding {len(schools)} schools...")
        
        if postcodes is None:
            postcodes = [None] * len(schools)
        
        results = []
        for i, (school, postcode) in enumerate(zip(schools, postcodes)):
            if i > 0 and i % 10 == 0:
                print(f"   Progress: {i}/{len(schools)}")
            
            result = self.geocode_school(school, postcode)
            if result:
                results.append(result)
            
            # Rate limiting
            time.sleep(delay)
        
        print(f"Successfully geocoded {len(results)}/{len(schools)} schools")
        print(f"Failed: {len(self.failed_geocodes)}")
        
        return pd.DataFrame(results) if results else pd.DataFrame()
    
    def use_uk_postcode_api(self, postcodes: List[str]) -> pd.DataFrame:
        """
        Use UK Postcode API for more accurate UK geocoding.
        
        Args:
            postcodes: List of UK postcodes
            
        Returns:
            DataFrame with geocoded postcodes
        """
        import requests
        
        print("🇬🇧 Using UK Postcode API...")
        
        results = []
        for postcode in postcodes:
            if pd.isna(postcode) or not postcode:
                continue
            
            try:
                # Clean postcode
                postcode = postcode.strip().replace(' ', '')
                
                # Call API
                url = f"https://api.postcodes.io/postcodes/{postcode}"
                response = requests.get(url, timeout=5)
                
                if response.status_code == 200:
                    data = response.json()
                    if data['status'] == 200:
                        result = data['result']
                        results.append({
                            'postcode': postcode,
                            'latitude': result['latitude'],
                            'longitude': result['longitude'],
                            'region': result.get('region'),
                            'parliamentary_constituency': result.get('parliamentary_constituency'),
                            'source': 'postcodes.io'
                        })
            except Exception as e:
                print(f"Error with postcode {postcode}: {e}")
        
        print(f"Geocoded {len(results)} postcodes")
        
        return pd.DataFrame(results)


# ============================================================================
# GEOGRAPHIC CLUSTERING
# ============================================================================

class GeographicCluster:
    """
    Perform clustering analysis on school locations.
    """
    
    def __init__(self, locations_df: pd.DataFrame):
        """
        Initialize with geocoded locations.
        
        Args:
            locations_df: DataFrame with latitude and longitude columns
        """
        self.locations_df = locations_df
        self.clusters = None
    
    def perform_kmeans_clustering(self, n_clusters: int = 5) -> pd.DataFrame:
        """
        Perform K-means clustering on locations.
        
        Args:
            n_clusters: Number of clusters
            
        Returns:
            DataFrame with cluster assignments
        """
        from sklearn.cluster import KMeans
        
        print(f"Performing K-means clustering (k={n_clusters})...")
        
        # Extract coordinates
        coords = self.locations_df[['latitude', 'longitude']].values
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.locations_df['cluster'] = kmeans.fit_predict(coords)
        
        # Add cluster centers
        centers = kmeans.cluster_centers_
        
        print(f"Clustered into {n_clusters} groups")
        
        # Calculate cluster statistics
        for i in range(n_clusters):
            cluster_size = (self.locations_df['cluster'] == i).sum()
            print(f"   • Cluster {i}: {cluster_size} schools")
        
        self.cluster_centers = centers
        self.clusters = self.locations_df.copy()
        
        return self.clusters
    
    def perform_dbscan_clustering(self, eps: float = 0.5, min_samples: int = 3) -> pd.DataFrame:
        """
        Perform DBSCAN clustering (density-based).
        
        Args:
            eps: Maximum distance between samples
            min_samples: Minimum samples in neighborhood
            
        Returns:
            DataFrame with cluster assignments
        """
        from sklearn.cluster import DBSCAN
        
        print(f"Performing DBSCAN clustering...")
        
        coords = self.locations_df[['latitude', 'longitude']].values
        
        # Convert eps from degrees to radians for haversine
        eps_rad = eps / 6371.0  # Earth radius in km
        
        dbscan = DBSCAN(eps=eps_rad, min_samples=min_samples, metric='haversine')
        self.locations_df['cluster'] = dbscan.fit_predict(np.radians(coords))
        
        n_clusters = len(set(self.locations_df['cluster'])) - (1 if -1 in self.locations_df['cluster'] else 0)
        n_noise = (self.locations_df['cluster'] == -1).sum()
        
        print(f"Found {n_clusters} clusters")
        print(f"Noise points: {n_noise}")
        
        self.clusters = self.locations_df.copy()
        
        return self.clusters
    
    def identify_coverage_gaps(self, target_radius_km: float = 50) -> Dict:
        """
        Identify geographic areas with low coverage.
        
        Args:
            target_radius_km: Target coverage radius in kilometers
            
        Returns:
            Dictionary with gap analysis
        """
        from sklearn.neighbors import NearestNeighbors
        
        print(f"Identifying coverage gaps (target radius: {target_radius_km}km)...")
        
        coords = self.locations_df[['latitude', 'longitude']].values
        
        # Find nearest neighbors
        nbrs = NearestNeighbors(n_neighbors=2, metric='haversine')
        nbrs.fit(np.radians(coords))
        
        distances, indices = nbrs.kneighbors(np.radians(coords))
        
        # Convert distances from radians to km
        distances_km = distances[:, 1] * 6371.0
        
        # Identify isolated schools
        isolated_schools = self.locations_df[distances_km > target_radius_km].copy()
        isolated_schools['distance_to_nearest'] = distances_km[distances_km > target_radius_km]
        
        gaps = {
            'isolated_schools': isolated_schools.to_dict('records'),
            'avg_nearest_distance_km': float(distances_km.mean()),
            'max_nearest_distance_km': float(distances_km.max()),
            'schools_beyond_target': len(isolated_schools)
        }
        
        print(f"Average distance to nearest school: {gaps['avg_nearest_distance_km']:.1f}km")
        print(f"{gaps['schools_beyond_target']} schools beyond {target_radius_km}km target")
        
        return gaps
    
    def calculate_geographic_metrics(self) -> Dict:
        """
        Calculate various geographic distribution metrics.
        
        Returns:
            Dictionary with metrics
        """
        print("Calculating geographic metrics...")
        
        metrics = {
            'total_schools': len(self.locations_df),
            'latitude_range': {
                'min': float(self.locations_df['latitude'].min()),
                'max': float(self.locations_df['latitude'].max()),
                'span': float(self.locations_df['latitude'].max() - self.locations_df['latitude'].min())
            },
            'longitude_range': {
                'min': float(self.locations_df['longitude'].min()),
                'max': float(self.locations_df['longitude'].max()),
                'span': float(self.locations_df['longitude'].max() - self.locations_df['longitude'].min())
            },
            'centroid': {
                'latitude': float(self.locations_df['latitude'].mean()),
                'longitude': float(self.locations_df['longitude'].mean())
            }
        }
        
        # Calculate coverage area (convex hull approximation)
        coords = self.locations_df[['latitude', 'longitude']].values
        if len(coords) >= 3:
            from scipy.spatial import ConvexHull
            try:
                hull = ConvexHull(coords)
                # Rough approximation of area in km²
                area_deg2 = hull.volume
                area_km2 = area_deg2 * (111 * 111)  # Very rough approximation
                metrics['coverage_area_km2'] = float(area_km2)
            except:
                metrics['coverage_area_km2'] = None
        
        print(f"Geographic centroid: ({metrics['centroid']['latitude']:.4f}, {metrics['centroid']['longitude']:.4f})")
        if 'coverage_area_km2' in metrics and metrics['coverage_area_km2']:
            print(f"Approximate coverage area: {metrics['coverage_area_km2']:.0f} km²")
        
        return metrics


# ============================================================================
# VISUALIZATION GENERATORS
# ============================================================================

class GeographicVisualizer:
    """
    Generate geographic visualizations and maps.
    """
    
    def __init__(self, locations_df: pd.DataFrame):
        """
        Initialize visualizer.
        
        Args:
            locations_df: DataFrame with geocoded locations
        """
        self.locations_df = locations_df
    
    def create_folium_map(self, output_path: str = '/home/claude/breck_schools_map.html',
                         cluster_column: Optional[str] = None,
                         max_connections_per_lead: int = 5) -> str:
        """
        Create an interactive Folium map.
        
        Args:
            output_path: Path to save HTML map
            cluster_column: Column name for cluster coloring
            
        Returns:
            Path to saved map
        """
        try:
            import folium
            from folium.plugins import MarkerCluster
            from pathlib import Path

            print("Creating interactive map with approach & leads...")

            # Calculate center
            center_lat = float(self.locations_df['latitude'].mean())
            center_lon = float(self.locations_df['longitude'].mean())

            # Create base map
            m = folium.Map(
                location=[center_lat, center_lon],
                zoom_start=7,
                tiles='OpenStreetMap'
            )

            # Color palette for clusters
            colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 
                     'lightred', 'beige', 'darkblue', 'darkgreen']

            # Attempt to enrich locations with school_summary (how the school was approached)
            try:
                base_dir = Path(__file__).resolve().parent
                possible_summary = possible_summary = base_dir.joinpath('outputs', 'school_summary.csv')
                if possible_summary.exists():
                    summary_df = pd.read_csv(possible_summary)
                    # Normalize names for matching
                    def _norm(s):
                        return str(s).lower().strip() if pd.notna(s) else ''
                    summary_map = { _norm(r['school_name']): r for _, r in summary_df.iterrows() }

                    def _lookup_how_heard(name):
                        return summary_map.get(_norm(name), {}).get('how_heard')

                    def _lookup_visit_count(name):
                        return summary_map.get(_norm(name), {}).get('visit_count')

                    self.locations_df['how_heard'] = self.locations_df['school_name'].apply(_lookup_how_heard)
                    self.locations_df['visit_count'] = self.locations_df['school_name'].apply(_lookup_visit_count)
                    print(f"   - Enriched locations with school_summary ({len(summary_df)} records)")
                else:
                    self.locations_df['how_heard'] = None
                    self.locations_df['visit_count'] = None
            except Exception as e:
                print(f"   [WARNING] Could not enrich with school_summary: {e}")
                self.locations_df['how_heard'] = None
                self.locations_df['visit_count'] = None

            # Attempt to load actionable leads (GIAS) and use them to highlight top leads
            leads_df = None
            try:
                possible_leads = base_dir.joinpath('outputs', 'actionable_school_leads.csv')
                if possible_leads.exists():
                    leads_df = pd.read_csv(possible_leads)
                    # Normalize gias_name for matching
                    leads_df['__norm_name'] = leads_df.get('gias_name', leads_df.get('gias_name', pd.Series([]))).fillna('').astype(str).str.lower().str.strip()
                    print(f"   - Loaded {len(leads_df)} actionable leads from {possible_leads}")
                else:
                    leads_df = None
            except Exception as e:
                print(f"   [WARNING] Could not load actionable leads: {e}")
                leads_df = None

            # Prepare layers
            cluster_layer = folium.FeatureGroup(name='Schools (by cluster)')
            leads_layer = folium.FeatureGroup(name='Actionable Leads', show=True)
            connections_layer = folium.FeatureGroup(name='Lead Connections', show=False)

            # Add markers for schools (colored by cluster if available)
            if cluster_column and cluster_column in self.locations_df.columns:
                for idx, row in self.locations_df.iterrows():
                    try:
                        cluster = int(row[cluster_column]) if pd.notna(row[cluster_column]) else 0
                    except Exception:
                        cluster = 0
                    color = colors[cluster % len(colors)]

                    popup_lines = [f"<b>{row.get('school_name', 'School')}</b>"]
                    if pd.notna(row.get('how_heard')):
                        popup_lines.append(f"Approach: {row.get('how_heard')}")
                    if pd.notna(row.get('visit_count')):
                        popup_lines.append(f"Visits: {row.get('visit_count')}")
                    popup_lines.append(f"Cluster: {cluster}")
                    popup_html = '<br>'.join(popup_lines)

                    folium.CircleMarker(
                        location=[row['latitude'], row['longitude']],
                        radius=7,
                        popup=popup_html,
                        color=color,
                        fill=True,
                        fillColor=color,
                        fillOpacity=0.7
                    ).add_to(cluster_layer)
            else:
                marker_cluster = MarkerCluster().add_to(cluster_layer)
                for idx, row in self.locations_df.iterrows():
                    popup_lines = [f"<b>{row.get('school_name', 'School')}</b>"]
                    if pd.notna(row.get('how_heard')):
                        popup_lines.append(f"Approach: {row.get('how_heard')}")
                    if pd.notna(row.get('visit_count')):
                        popup_lines.append(f"Visits: {row.get('visit_count')}")
                    popup_html = '<br>'.join(popup_lines)

                    folium.Marker(
                        location=[row['latitude'], row['longitude']],
                        popup=popup_html,
                        icon=folium.Icon(color='blue', icon='school', prefix='fa')
                    ).add_to(marker_cluster)

            # Add lead markers (distinctive color and icon) and connections to schools in same trust or cluster
            if leads_df is not None and not leads_df.empty:
                # Normalize helper
                def _norm(s):
                    return str(s).lower().strip() if pd.notna(s) else ''

                # Try to load a full GIAS/edubase file (to map visited schools to trust names)
                gias_df = None
                try:
                    possible_gias = base_dir.joinpath('edubasealldata20251024.csv')
                    if not possible_gias.exists():
                        possible_gias = base_dir.parent.joinpath('edubasealldata20251024.csv')
                    if possible_gias.exists():
                        try:
                            gias_df = pd.read_csv(possible_gias, low_memory=False)
                        except UnicodeDecodeError:
                            gias_df = pd.read_csv(possible_gias, encoding='latin-1', low_memory=False)
                        # Normalize and select expected columns
                        gias_cols_map = {
                            'EstablishmentName': 'gias_name',
                            'EstablishmentName (name)': 'gias_name',
                            'Trusts (name)': 'trust_name',
                            'Latitude': 'gias_lat',
                            'Longitude': 'gias_lon',
                            'URN': 'gias_urn'
                        }
                        existing = {c: gias_cols_map[c] for c in gias_cols_map.keys() if c in gias_df.columns}
                        if existing:
                            gias_df = gias_df[list(existing.keys())].rename(columns=existing)
                            # Add normalized name
                            gias_df['__norm_name'] = gias_df['gias_name'].astype(str).str.lower().str.strip()
                            print(f"   - Loaded full GIAS/edubase ({len(gias_df)} rows) for trust mapping")
                        else:
                            gias_df = None
                except Exception as e:
                    gias_df = None

                # Merge trust name into visited schools if possible (by normalized name)
                if gias_df is not None:
                    try:
                        # Simple exact merge first
                        tmp = self.locations_df.copy()
                        tmp['__norm_name'] = tmp['school_name'].astype(str).str.lower().str.strip()
                        merged = tmp.merge(gias_df[['__norm_name', 'trust_name']], on='__norm_name', how='left')
                        self.locations_df['trust_name'] = merged['trust_name']

                        # Fill remaining missing trust names using fuzzy matching if available
                        missing = self.locations_df[self.locations_df['trust_name'].isna()]
                        if len(missing) > 0:
                            choices = gias_df['gias_name'].astype(str).tolist()
                            try:
                                from rapidfuzz import process as rf_process, fuzz as rf_fuzz
                                for idx, row in missing.iterrows():
                                    name = row.get('school_name')
                                    if not name:
                                        continue
                                    match = rf_process.extractOne(str(name), choices, scorer=rf_fuzz.token_sort_ratio)
                                    if match and match[1] >= 85:
                                        matched_row = gias_df[gias_df['gias_name'].astype(str) == match[0]]
                                        if not matched_row.empty:
                                            self.locations_df.at[idx, 'trust_name'] = matched_row.iloc[0].get('trust_name')
                            except Exception:
                                import difflib
                                from difflib import get_close_matches, SequenceMatcher
                                for idx, row in missing.iterrows():
                                    name = row.get('school_name')
                                    if not name:
                                        continue
                                    matches = get_close_matches(str(name), choices, n=1, cutoff=0.7)
                                    if matches:
                                        score = int(SequenceMatcher(None, str(name), matches[0]).ratio() * 100)
                                        if score >= 85:
                                            matched_row = gias_df[gias_df['gias_name'].astype(str) == matches[0]]
                                            if not matched_row.empty:
                                                self.locations_df.at[idx, 'trust_name'] = matched_row.iloc[0].get('trust_name')

                        print("   - Merged trust names into visited schools where available (with fuzzy fallback)")
                    except Exception:
                        self.locations_df['trust_name'] = None
                else:
                    if 'trust_name' not in self.locations_df.columns:
                        self.locations_df['trust_name'] = None

                # Compute cluster centroids if cluster column exists
                cluster_centroids = {}
                if cluster_column and cluster_column in self.locations_df.columns:
                    try:
                        centroids = self.locations_df.groupby(cluster_column)[['latitude', 'longitude']].mean()
                        for cidx, rowc in centroids.iterrows():
                            cluster_centroids[cidx] = (float(rowc['latitude']), float(rowc['longitude']))
                    except Exception:
                        cluster_centroids = {}

                # For each lead, connect to visited schools that share trust OR belong to nearest cluster
                for _, lead in leads_df.iterrows():
                    try:
                        lead_lat = lead.get('gias_lat') if 'gias_lat' in lead else lead.get('Latitude')
                        lead_lon = lead.get('gias_lon') if 'gias_lon' in lead else lead.get('Longitude')
                        if pd.isna(lead_lat) or pd.isna(lead_lon):
                            continue
                        lead_name = lead.get('gias_name', lead.get('gias_name', 'Lead School'))
                        lead_reason = lead.get('lead_reason', '')
                        lead_source = lead.get('lead_source', '')
                        lead_trust = lead.get('trust_name', None)

                        popup_lines = [f"<b>{lead_name}</b>", f"Lead reason: {lead_reason}", f"Source: {lead_source}"]
                        if lead_trust and pd.notna(lead_trust):
                            popup_lines.append(f"Trust: {lead_trust}")
                        popup_html = '<br>'.join(popup_lines)

                        folium.Marker(
                            location=[float(lead_lat), float(lead_lon)],
                            popup=popup_html,
                            icon=folium.Icon(color='darkred', icon='flag', prefix='fa')
                        ).add_to(leads_layer)

                        # Track which visited schools we've already connected to (avoid duplicates)
                        connected = set()

                        # 1) Connect by trust
                        if lead_trust and pd.notna(lead_trust):
                            matches = self.locations_df[self.locations_df['trust_name'] == lead_trust]
                            for _, v in matches.iterrows():
                                try:
                                    if pd.isna(v['latitude']) or pd.isna(v['longitude']):
                                        continue
                                    key = (float(v['latitude']), float(v['longitude']))
                                    if key in connected:
                                        continue
                                    folium.PolyLine(locations=[[float(lead_lat), float(lead_lon)], [float(v['latitude']), float(v['longitude'])]], color='green', weight=2, opacity=0.8).add_to(connections_layer)
                                    connected.add(key)
                                    if len(connected) >= max_connections_per_lead:
                                        break
                                except Exception:
                                    continue

                        # 2) Connect by nearest cluster (if clusters known)
                        if cluster_centroids:
                            # find nearest cluster centroid to the lead
                            def haversine(lat1, lon1, lat2, lon2):
                                from math import radians, sin, cos, asin, sqrt
                                dlat = radians(lat2 - lat1)
                                dlon = radians(lon2 - lon1)
                                a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
                                return 2 * 6371 * asin(sqrt(a))

                            nearest_cluster = None
                            nearest_dist = float('inf')
                            for cidx, (clat, clon) in cluster_centroids.items():
                                try:
                                    d = haversine(float(lead_lat), float(lead_lon), clat, clon)
                                    if d < nearest_dist:
                                        nearest_dist = d
                                        nearest_cluster = cidx
                                except Exception:
                                    continue

                            if nearest_cluster is not None:
                                cluster_members = self.locations_df[self.locations_df[cluster_column] == nearest_cluster]
                                for _, v in cluster_members.iterrows():
                                    try:
                                        if pd.isna(v['latitude']) or pd.isna(v['longitude']):
                                            continue
                                        key = (float(v['latitude']), float(v['longitude']))
                                        if key in connected:
                                            continue
                                        folium.PolyLine(locations=[[float(lead_lat), float(lead_lon)], [float(v['latitude']), float(v['longitude'])]], color='orange', weight=1.5, opacity=0.6).add_to(connections_layer)
                                        connected.add(key)
                                        if len(connected) >= max_connections_per_lead:
                                            break
                                    except Exception:
                                        continue

                    except Exception:
                        continue

            # Add layers to map
            cluster_layer.add_to(m)
            leads_layer.add_to(m)
            connections_layer.add_to(m)

            folium.LayerControl(collapsed=False).add_to(m)

            # Save map
            m.save(output_path)
            print(f"Map saved to: {output_path}")
            return output_path

        except ImportError:
            print("Folium not installed. Install with: pip install folium")
            return None
    
    def create_heatmap(self, output_path: str = '/home/claude/breck_heatmap.html') -> str:
        """
        Create a heat map showing workshop concentration.
        
        Args:
            output_path: Path to save HTML map
            
        Returns:
            Path to saved map
        """
        try:
            import folium
            from folium.plugins import HeatMap
            
            print("Creating heat map...")
            
            # Calculate center
            center_lat = self.locations_df['latitude'].mean()
            center_lon = self.locations_df['longitude'].mean()
            
            # Create base map
            m = folium.Map(
                location=[center_lat, center_lon],
                zoom_start=7,
                tiles='OpenStreetMap'
            )
            
            # Prepare heat map data
            heat_data = [[row['latitude'], row['longitude']] 
                        for idx, row in self.locations_df.iterrows()]
            
            # Add heat map layer
            HeatMap(heat_data, radius=15, blur=25).add_to(m)
            
            # Save map
            m.save(output_path)
            
            print(f"Heat map saved to: {output_path}")
            
            return output_path
            
        except ImportError:
            print("Folium not installed. Install with: pip install folium")
            return None
    
    def create_static_plot(self, output_path: str = '/home/claude/breck_schools_plot.png',
                          cluster_column: Optional[str] = None):
        """
        Create static matplotlib visualization.
        
        Args:
            output_path: Path to save plot
            cluster_column: Column for cluster coloring
        """
        import matplotlib.pyplot as plt
        
        print("Creating static plot...")
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        if cluster_column and cluster_column in self.locations_df.columns:
            # Color by cluster
            scatter = ax.scatter(
                self.locations_df['longitude'],
                self.locations_df['latitude'],
                c=self.locations_df[cluster_column],
                cmap='tab10',
                s=100,
                alpha=0.6,
                edgecolors='black',
                linewidth=0.5
            )
            plt.colorbar(scatter, ax=ax, label='Cluster')
        else:
            ax.scatter(
                self.locations_df['longitude'],
                self.locations_df['latitude'],
                s=100,
                alpha=0.6,
                color='blue',
                edgecolors='black',
                linewidth=0.5
            )
        
        ax.set_xlabel('Longitude', fontsize=12)
        ax.set_ylabel('Latitude', fontsize=12)
        ax.set_title('Breck Foundation Workshop Locations', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Plot saved to: {output_path}")
        
        return output_path


# ============================================================================
# MAIN ORCHESTRATION
# ============================================================================

def run_geographic_analysis(schools: List[str], 
                           postcodes: Optional[List[str]] = None,
                           n_clusters: int = 5) -> Dict:
    """
    Run complete geographic analysis pipeline.
    
    Args:
        schools: List of school names
        postcodes: Optional list of postcodes
        n_clusters: Number of clusters for analysis
        
    Returns:
        Dictionary with all analysis results
    """
    print("=" * 80)
    print("GEOGRAPHIC CLUSTERING ANALYSIS")
    print("=" * 80)
    print()
    
    results = {
        'timestamp': pd.Timestamp.now().isoformat(),
        'input_schools': len(schools)
    }
    
    # 1. Geocode schools
    geocoder = SchoolGeocoder()
    geocoded_df = geocoder.geocode_schools_batch(schools, postcodes)
    
    if len(geocoded_df) == 0:
        print("No schools could be geocoded")
        return results
    
    results['geocoded_schools'] = len(geocoded_df)
    
    # 2. Perform clustering
    cluster_analyzer = GeographicCluster(geocoded_df)
    clustered_df = cluster_analyzer.perform_kmeans_clustering(n_clusters)
    
    results['clusters'] = cluster_analyzer.clusters.to_dict('records')
    
    # 3. Identify gaps
    gaps = cluster_analyzer.identify_coverage_gaps()
    results['coverage_gaps'] = gaps
    
    # 4. Calculate metrics
    metrics = cluster_analyzer.calculate_geographic_metrics()
    results['geographic_metrics'] = metrics
    
    # 5. Create visualizations
    visualizer = GeographicVisualizer(clustered_df)
    
    try:
        map_path = visualizer.create_folium_map(cluster_column='cluster')
        results['interactive_map'] = map_path
    except:
        print("Could not create interactive map")
    
    try:
        heatmap_path = visualizer.create_heatmap()
        results['heatmap'] = heatmap_path
    except:
        print("Could not create heatmap")
    
    try:
        plot_path = visualizer.create_static_plot(cluster_column='cluster')
        results['static_plot'] = plot_path
    except:
        print("Could not create static plot")
    
    print("\n" + "=" * 80)
    print("GEOGRAPHIC ANALYSIS COMPLETE")
    print("=" * 80)
    
    return results


if __name__ == "__main__":
    print("Geographic Clustering Module for Breck Foundation")
    print("This module should be imported and used with the main framework")
