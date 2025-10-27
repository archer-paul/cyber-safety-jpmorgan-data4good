"""
BRECK FOUNDATION - INTERACTIVE MAP GENERATOR
=============================================

Creates a stunning, interactive map for hackathon presentations showing:
- Visited schools (color-coded by visit frequency)
- Normal leads (fundraising opportunities)
- Deprived area leads (social impact opportunities)
- Connection lines between leads and champion schools

Author: JP Morgan Data for Good Hackathon Team
Date: October 2025
Version: 2.0
"""

import pandas as pd
import numpy as np
import folium
from folium import plugins
from pathlib import Path
from geopy.distance import great_circle


class BreckInteractiveMapGenerator:
    """
    Creates stunning interactive maps for the Breck Foundation.
    """

    # Color scheme for the map
    COLORS = {
        'visited_once': '#90EE90',      # Light green
        'visited_multiple': '#228B22',   # Forest green
        'champion': '#FFD700',           # Gold/Yellow for champions (4+ visits)
        'normal_lead': '#4169E1',        # Royal blue
        'deprived_lead': '#DC143C',      # Crimson red
        'connection': '#FFA500'          # Orange for connections
    }

    def __init__(self, config):
        """
        Initialize the map generator.

        Args:
            config (dict): Configuration with paths to data files
        """
        self.config = config
        self.visited_schools = None
        self.normal_leads = None
        self.deprived_leads = None

    def load_data(self):
        """Load all required data files."""
        print("\n" + "=" * 80)
        print("LOADING DATA FOR INTERACTIVE MAP")
        print("=" * 80)

        # Load visited schools
        visited_path = self.config.get('visited_schools_file')
        if visited_path and Path(visited_path).exists():
            self.visited_schools = pd.read_csv(visited_path)
            print(f"[OK] Loaded {len(self.visited_schools)} visited schools")
        else:
            print("[WARNING] Warning: No visited schools file found")
            self.visited_schools = pd.DataFrame()

        # Load normal leads
        normal_path = self.config.get('normal_leads_file')
        if normal_path and Path(normal_path).exists():
            self.normal_leads = pd.read_csv(normal_path)
            print(f"[OK] Loaded {len(self.normal_leads)} normal leads")
        else:
            print("[WARNING] Warning: No normal leads file found")
            self.normal_leads = pd.DataFrame()

        # Load deprived area leads
        deprived_path = self.config.get('deprived_leads_file')
        if deprived_path and Path(deprived_path).exists():
            self.deprived_leads = pd.read_csv(deprived_path)
            print(f"[OK] Loaded {len(self.deprived_leads)} deprived area leads")
        else:
            print("[WARNING] Warning: No deprived area leads file found")
            self.deprived_leads = pd.DataFrame()

    def create_interactive_map(self, output_path='outputs/breck_interactive_map.html', max_connections=3):
        """
        Create the main interactive map.

        Args:
            output_path (str): Path to save the HTML map
            max_connections (int): Max connection lines per lead

        Returns:
            str: Path to saved map
        """
        print("\n" + "=" * 80)
        print("CREATING INTERACTIVE MAP")
        print("=" * 80)

        # Calculate map center (UK center as fallback)
        center_lat, center_lon = self._calculate_center()

        # Create base map with detailed tiles showing roads and borders
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=7,
            tiles='OpenStreetMap',
            prefer_canvas=True
        )

        # Add custom legend
        self._add_legend(m)

        # Add title
        self._add_title(m)

        # Create layer groups
        visited_layer = folium.FeatureGroup(name=' Visited Schools', show=True)
        normal_leads_layer = folium.FeatureGroup(name=' Normal Leads (Fundraising)', show=True)
        deprived_leads_layer = folium.FeatureGroup(name=' Deprived Area Leads (Impact)', show=True)
        connections_layer = folium.FeatureGroup(name=' Connections to Champions', show=False)

        # Add visited schools
        print("\n> Adding visited schools to map...")
        self._add_visited_schools(m, visited_layer)

        # Add normal leads
        print("> Adding normal leads to map...")
        self._add_normal_leads(m, normal_leads_layer, connections_layer, max_connections)

        # Add deprived area leads
        print("> Adding deprived area leads to map...")
        self._add_deprived_leads(m, deprived_leads_layer, connections_layer, max_connections)

        # Add layers to map
        visited_layer.add_to(m)
        normal_leads_layer.add_to(m)
        deprived_leads_layer.add_to(m)
        connections_layer.add_to(m)

        # Add layer control
        folium.LayerControl(collapsed=False).add_to(m)

        # Add fullscreen button
        plugins.Fullscreen(position='topright').add_to(m)

        # Add measure tool
        plugins.MeasureControl(position='bottomleft').add_to(m)

        # Save map
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        m.save(output_path)

        print(f"\n[OK] Interactive map saved: {output_path}")
        print("=" * 80)

        return output_path

    def _calculate_center(self):
        """Calculate the center point for the map."""
        all_lats = []
        all_lons = []

        # Collect all coordinates
        if len(self.visited_schools) > 0:
            all_lats.extend(self.visited_schools['latitude'].dropna().tolist())
            all_lons.extend(self.visited_schools['longitude'].dropna().tolist())

        if len(self.normal_leads) > 0:
            all_lats.extend(self.normal_leads['gias_lat'].dropna().tolist())
            all_lons.extend(self.normal_leads['gias_lon'].dropna().tolist())

        if len(self.deprived_leads) > 0:
            all_lats.extend(self.deprived_leads['gias_lat'].dropna().tolist())
            all_lons.extend(self.deprived_leads['gias_lon'].dropna().tolist())

        if all_lats and all_lons:
            return np.mean(all_lats), np.mean(all_lons)
        else:
            # Default to UK center
            return 52.5, -1.5

    def _add_visited_schools(self, map_obj, layer):
        """Add visited schools to the map with color coding by visit frequency."""
        if len(self.visited_schools) == 0:
            return

        # Filter schools with coordinates
        schools = self.visited_schools.dropna(subset=['latitude', 'longitude'])

        for _, school in schools.iterrows():
            visit_count = school.get('visit_count', 1)

            # Determine color and icon based on visit count
            # Note: Max visits in current data is 2, so champions are 2+ visits
            if visit_count == 1:
                color = self.COLORS['visited_once']
                marker_color = 'lightgreen'
                icon = 'school'
                icon_color = 'white'
                prefix = 'fa'
                label = '1 visit'
            else:  # 2+ visits = Champion
                color = self.COLORS['champion']
                marker_color = 'orange'  # Use orange for yellow/gold effect
                icon = 'trophy'
                icon_color = 'white'
                prefix = 'fa'
                label = f'{visit_count} visits (CHAMPION!)'

            # Create rich popup
            popup_html = f"""
            <div style="width: 250px; font-family: Arial, sans-serif;">
                <h4 style="margin: 0 0 10px 0; color: {color};">
                    <i class="fa fa-school"></i> {school.get('school_name', 'Unknown')}
                </h4>
                <hr style="margin: 5px 0;">
                <p style="margin: 5px 0;">
                    <b>Status:</b> <span style="color: {color}; font-weight: bold;">{label}</span><br>
                    <b>First visit:</b> {school.get('first_visit', 'Unknown')}<br>
                    <b>Last visit:</b> {school.get('last_visit', 'Unknown')}<br>
                    <b>How heard:</b> {school.get('how_heard', 'Unknown')}<br>
                    <b>Contact:</b> {school.get('contact_person', 'Unknown')}
                </p>
            </div>
            """

            # Add marker with custom icon
            folium.Marker(
                location=[school['latitude'], school['longitude']],
                popup=folium.Popup(popup_html, max_width=300),
                tooltip=f"{school.get('school_name', 'School')} - {label}",
                icon=folium.Icon(
                    color=marker_color,
                    icon=icon,
                    prefix=prefix,
                    icon_color=icon_color
                )
            ).add_to(layer)

        print(f"  [OK] Added {len(schools)} visited schools")

    def _add_normal_leads(self, map_obj, layer, connections_layer, max_connections):
        """Add normal leads (fundraising) to the map."""
        if len(self.normal_leads) == 0:
            return

        # Filter leads with coordinates
        leads = self.normal_leads.dropna(subset=['gias_lat', 'gias_lon'])

        # Get champion schools for connections
        champions = self.visited_schools[self.visited_schools['visit_count'] >= 2] if len(self.visited_schools) > 0 else pd.DataFrame()
        champions = champions.dropna(subset=['latitude', 'longitude'])

        for _, lead in leads.iterrows():
            # Create popup
            popup_html = f"""
            <div style="width: 280px; font-family: Arial, sans-serif;">
                <h4 style="margin: 0 0 10px 0; color: {self.COLORS['normal_lead']};">
                    <i class="fa fa-flag"></i> {lead.get('gias_name', 'School')}
                </h4>
                <hr style="margin: 5px 0;">
                <p style="margin: 5px 0;">
                    <b>Type:</b> <span style="color: {self.COLORS['normal_lead']}; font-weight: bold;">NORMAL LEAD</span><br>
                    <b>Reason:</b> {lead.get('lead_type', 'Unknown')}<br>
                    <b>Source:</b> {lead.get('lead_source', 'Unknown')}<br>
                    <b>Trust:</b> {lead.get('trust_name', 'N/A')}<br>
                    <b>Nearest champion:</b> {lead.get('nearest_champion', 'Unknown')}<br>
                    <b>Distance:</b> {lead.get('distance_to_champion_km', 0):.1f} km<br>
                    <b>Priority score:</b> {lead.get('priority_score', 0):.1f}/10
                </p>
                <p style="margin: 10px 0 0 0; padding: 5px; background: #e6f3ff; border-radius: 3px; font-size: 11px;">
                     <b>Fundraising opportunity</b> - School in same trust or near champion schools
                </p>
            </div>
            """

            # Add marker
            folium.Marker(
                location=[lead['gias_lat'], lead['gias_lon']],
                popup=folium.Popup(popup_html, max_width=320),
                tooltip=f"{lead.get('gias_name', 'Lead')} - Normal Lead",
                icon=folium.Icon(
                    color='blue',
                    icon='briefcase',
                    prefix='fa'
                )
            ).add_to(layer)

            # Add connection lines to nearest champions
            self._add_connection_line(
                lead,
                champions,
                connections_layer,
                self.COLORS['normal_lead'],
                max_connections
            )

        print(f"  [OK] Added {len(leads)} normal leads")

    def _add_deprived_leads(self, map_obj, layer, connections_layer, max_connections):
        """Add deprived area leads (social impact) to the map."""
        if len(self.deprived_leads) == 0:
            return

        # Filter leads with coordinates
        leads = self.deprived_leads.dropna(subset=['gias_lat', 'gias_lon'])

        # Get champion schools for connections
        champions = self.visited_schools[self.visited_schools['visit_count'] >= 2] if len(self.visited_schools) > 0 else pd.DataFrame()
        champions = champions.dropna(subset=['latitude', 'longitude'])

        for _, lead in leads.iterrows():
            fsm = lead.get('fsm_percentage', 0)

            # Create popup
            popup_html = f"""
            <div style="width: 280px; font-family: Arial, sans-serif;">
                <h4 style="margin: 0 0 10px 0; color: {self.COLORS['deprived_lead']};">
                    <i class="fa fa-heart"></i> {lead.get('gias_name', 'School')}
                </h4>
                <hr style="margin: 5px 0;">
                <p style="margin: 5px 0;">
                    <b>Type:</b> <span style="color: {self.COLORS['deprived_lead']}; font-weight: bold;">DEPRIVED AREA LEAD</span><br>
                    <b>FSM%:</b> <span style="color: {self.COLORS['deprived_lead']}; font-weight: bold;">{fsm:.1f}%</span><br>
                    <b>Source:</b> {lead.get('lead_source', 'High deprivation')}<br>
                    <b>Trust:</b> {lead.get('trust_name', 'N/A')}<br>
                    <b>Nearest champion:</b> {lead.get('nearest_champion', 'Unknown')}<br>
                    <b>Distance:</b> {lead.get('distance_to_champion_km', 0):.1f} km<br>
                    <b>Impact score:</b> {lead.get('impact_score', 0):.1f}/10
                </p>
                <p style="margin: 10px 0 0 0; padding: 5px; background: #ffe6e6; border-radius: 3px; font-size: 11px;">
                     <b>High social impact opportunity</b> - School in deprived area (FSM {fsm:.1f}%)
                </p>
            </div>
            """

            # Add marker
            folium.Marker(
                location=[lead['gias_lat'], lead['gias_lon']],
                popup=folium.Popup(popup_html, max_width=320),
                tooltip=f"{lead.get('gias_name', 'Lead')} - Deprived Area (FSM {fsm:.1f}%)",
                icon=folium.Icon(
                    color='red',
                    icon='heart',
                    prefix='fa'
                )
            ).add_to(layer)

            # Add connection lines to nearest champions
            self._add_connection_line(
                lead,
                champions,
                connections_layer,
                self.COLORS['deprived_lead'],
                max_connections
            )

        print(f"  [OK] Added {len(leads)} deprived area leads")

    def _add_connection_line(self, lead, champions, layer, color, max_connections):
        """Add connection line from lead to nearest champion(s)."""
        if len(champions) == 0:
            return

        lead_coord = (lead['gias_lat'], lead['gias_lon'])
        connections = []

        # Find nearest champions
        for _, champ in champions.iterrows():
            try:
                champ_coord = (champ['latitude'], champ['longitude'])
                distance = great_circle(lead_coord, champ_coord).kilometers
                connections.append((distance, champ_coord, champ.get('school_name', 'Champion')))
            except Exception:
                continue

        # Sort by distance and take top N
        connections.sort(key=lambda x: x[0])
        connections = connections[:max_connections]

        # Draw connection lines
        for dist, champ_coord, champ_name in connections:
            folium.PolyLine(
                locations=[lead_coord, champ_coord],
                color=color,
                weight=2,
                opacity=0.4,
                popup=f"Connection to {champ_name} ({dist:.1f} km)",
                dash_array='5, 10'
            ).add_to(layer)

    def _add_legend(self, map_obj):
        """Add a beautiful legend to the map."""
        legend_html = f"""
        <div style="position: fixed;
                    bottom: 50px; left: 50px; width: 320px;
                    background-color: white;
                    border: 2px solid grey;
                    border-radius: 5px;
                    z-index: 9999;
                    font-size: 14px;
                    padding: 15px;
                    box-shadow: 0 0 15px rgba(0,0,0,0.2);">
            <h4 style="margin: 0 0 10px 0; text-align: center; color: #333;">
                 Map Legend
            </h4>
            <hr style="margin: 10px 0;">

            <div style="margin-bottom: 8px;">
                <span style="display: inline-block; width: 12px; height: 12px; background: {self.COLORS['visited_once']}; border-radius: 50%; margin-right: 8px;"></span>
                <b>Visited Once</b> (1 visit)
            </div>

            <div style="margin-bottom: 8px;">
                <span style="display: inline-block; width: 12px; height: 12px; background: {self.COLORS['champion']}; border-radius: 50%; margin-right: 8px;"></span>
                <b>Champion School</b> (2+ visits)
            </div>

            <hr style="margin: 10px 0;">

            <div style="margin-bottom: 8px;">
                <span style="display: inline-block; width: 12px; height: 12px; background: {self.COLORS['normal_lead']}; border-radius: 50%; margin-right: 8px;"></span>
                <b>Normal Lead</b> (Fundraising)
            </div>

            <div style="margin-bottom: 8px;">
                <span style="display: inline-block; width: 12px; height: 12px; background: {self.COLORS['deprived_lead']}; border-radius: 50%; margin-right: 8px;"></span>
                <b>Deprived Area Lead</b> (High Impact)
            </div>

            <hr style="margin: 10px 0;">

            <div style="font-size: 11px; color: #666; text-align: center; margin-top: 10px;">
                Click markers for details<br>
                Toggle layers in top-right corner
            </div>
        </div>
        """

        map_obj.get_root().html.add_child(folium.Element(legend_html))

    def _add_title(self, map_obj):
        """Title removed - professional map without header banner."""
        pass


def generate_interactive_map(config):
    """
    Generate the interactive map.

    Args:
        config (dict): Configuration dictionary with file paths

    Returns:
        str: Path to generated map
    """
    generator = BreckInteractiveMapGenerator(config)
    generator.load_data()
    return generator.create_interactive_map(
        output_path=config.get('map_output_path', 'outputs/breck_interactive_map.html'),
        max_connections=config.get('max_connections_per_lead', 2)
    )
