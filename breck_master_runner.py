"""
BRECK FOUNDATION - MASTER ANALYSIS RUNNER
==========================================
Complete analysis pipeline integrating all modules:
- Data loading and cleaning
- Effectiveness analysis
- NLP text analysis
- Geographic clustering
- Cyber safety trends
- Recommendations generation

Run this script to perform the complete analysis for the hackathon.

Author: Hackathon Team
Date: October 2025
"""

import sys
import json
from datetime import datetime
from pathlib import Path

# Import all modules
from breck_comprehensive_framework import BreckDataLoader, BreckComprehensiveAnalyzer
from breck_nlp_advanced import FeedbackTextAnalyzer, run_complete_nlp_analysis, export_nlp_results
from breck_geographic_clustering import run_geographic_analysis
from breck_lead_generator_v2 import EnhancedLeadGenerator
from breck_interactive_map_generator import generate_interactive_map

import pandas as pd
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    'data_file': r'C:\Users\Paul\OneDrive - telecom-paristech.fr\Documents\Pro\Hackathon\JPMorgan Data for Good\V2\Breck_Internal_Data.xlsx',
    # Path to the national schools database (used by LeadGenerator). Update to the provided edubase file.
    'gias_database': r'C:\Users\Paul\OneDrive - telecom-paristech.fr\Documents\Pro\Hackathon\JPMorgan Data for Good\V2\edubasealldata20251024.csv',
    # Optional path to a precomputed Breck school summary (visit counts). Defaults to outputs/school_summary.csv
    'breck_summary_file': r'C:\Users\Paul\OneDrive - telecom-paristech.fr\Documents\Pro\Hackathon\JPMorgan Data for Good\V2\outputs\school_summary.csv',
    'output_dir': r'C:\Users\Paul\OneDrive - telecom-paristech.fr\Documents\Pro\Hackathon\JPMorgan Data for Good\V2\outputs',
    'n_clusters': 5,  # For geographic clustering
    'create_visualizations': True,
    'analyze_news': False,  # Set to True if news data is available
    'news_file': None,  # Path to news data CSV if available

    # Enhanced Lead Generation Configuration
    # champion_min_visits set to 2 as max observed visits is 2
    'champion_min_visits': 2,
    'hotspot_radius_km': 5.0,
    'fsm_threshold': 30.0,  # % FSM for deprived areas
    'normal_leads_file': r'C:\Users\Paul\OneDrive - telecom-paristech.fr\Documents\Pro\Hackathon\JPMorgan Data for Good\V2\outputs\normal_leads.csv',
    'deprived_leads_file': r'C:\Users\Paul\OneDrive - telecom-paristech.fr\Documents\Pro\Hackathon\JPMorgan Data for Good\V2\outputs\deprived_area_leads.csv',

    # Interactive Map Configuration
    'map_output_path': r'C:\Users\Paul\OneDrive - telecom-paristech.fr\Documents\Pro\Hackathon\JPMorgan Data for Good\V2\outputs\breck_interactive_map.html',
    'max_connections_per_lead': 2,
    'visited_schools_file': r'C:\Users\Paul\OneDrive - telecom-paristech.fr\Documents\Pro\Hackathon\JPMorgan Data for Good\V2\outputs\schools_with_geocoding.csv'
}


# ============================================================================
# MAIN ANALYSIS PIPELINE
# ============================================================================

class BreckMasterAnalyzer:
    """
    Master orchestrator for complete Breck Foundation analysis.
    """
    
    def __init__(self, config: dict):
        """
        Initialize master analyzer with configuration.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'config': config
        }
        self.loader = None
        self.analyzer = None
        
        # Create output directory
        Path(config['output_dir']).mkdir(parents=True, exist_ok=True)
    
    def run_complete_analysis(self):
        """
        Execute complete analysis pipeline.
        """
        print("=" * 80)
        print(">> BRECK FOUNDATION - COMPLETE DATA ANALYSIS")
        print("   JP Morgan Data for Good Hackathon 2025")
        print("=" * 80)
        print(f"\nTimestamp: {self.results['timestamp']}")
        print(f"Data file: {self.config['data_file']}")
        print(f"Output directory: {self.config['output_dir']}")
        print("\n" + "=" * 80)
        
        # Phase 1: Data Loading
        print("\n[1] PHASE 1: DATA LOADING & CLEANING")
        print("-" * 80)
        self._load_and_clean_data()
        
        # Phase 2: Core Analysis
        print("\n[2] PHASE 2: CORE EFFECTIVENESS ANALYSIS")
        print("-" * 80)
        self._analyze_effectiveness()
        
        # Phase 3: Data Gap Analysis
        print("\n[3] PHASE 3: DATA COLLECTION GAPS")
        print("-" * 80)
        self._analyze_data_gaps()
        
        # Phase 4: Session Analysis
        print("\n[4] PHASE 4: SESSION DELIVERY ANALYSIS")
        print("-" * 80)
        self._analyze_sessions()
        
        # Phase 5: NLP Analysis
        print("\n[5] PHASE 5: NLP TEXT ANALYSIS")
        print("-" * 80)
        self._run_nlp_analysis()
        
        # Phase 6: Geographic Analysis
        print("\n[6] PHASE 6: GEOGRAPHIC CLUSTERING")
        print("-" * 80)
        self._run_geographic_analysis()
        
        # Phase 7: Generate Recommendations
        print("\n[7] PHASE 7: RECOMMENDATIONS GENERATION")
        print("-" * 80)
        self._generate_recommendations()
        
        # Phase 8: Export All Results
        print("\n[8] PHASE 8: EXPORT RESULTS")
        print("-" * 80)
        self._export_all_results()
        
        # Final Summary
        self._print_final_summary()
        
        return self.results
    
    def _load_and_clean_data(self):
        """Phase 1: Load and clean all data."""
        try:
            self.loader = BreckDataLoader(self.config['data_file'])
            self.loader.load_all_sheets()
            self.loader.clean_young_person_feedback()
            self.loader.clean_adult_feedback()
            self.loader.clean_class_feedback()
            self.loader.parse_all_session_data()
            self.loader.clean_qualitative_feedback()
            
            # Store summary
            self.results['data_summary'] = self.loader.get_summary()

            print("\n[OK] Data loading complete!")
            print(f"   - Youth responses: {len(self.loader.clean_data.get('youth_feedback', []))}")
            print(f"   - Adult responses: {len(self.loader.clean_data.get('adult_feedback', []))}")
            print(f"   - Session records: {len(self.loader.clean_data.get('all_sessions', []))}")
            
        except Exception as e:
            print(f"\n[ERROR] Error in data loading: {e}")
            raise
    
    def _analyze_effectiveness(self):
        """Phase 2: Analyze program effectiveness."""
        try:
            self.analyzer = BreckComprehensiveAnalyzer(self.loader)
            effectiveness = self.analyzer.analyze_program_effectiveness()
            self.results['effectiveness_analysis'] = effectiveness
            
            print("\n[OK] Effectiveness analysis complete!")

        except Exception as e:
            print(f"\n[ERROR] Error in effectiveness analysis: {e}")
    
    def _analyze_data_gaps(self):
        """Phase 3: Identify data collection gaps."""
        try:
            gaps = self.analyzer.analyze_data_collection_gaps()
            self.results['data_gaps'] = gaps
            
            print("\n[OK] Data gap analysis complete!")
            print(f"   - Missing data points: {len(gaps.get('missing_data_points', []))}")
            print(f"   - Suggested additions: {len(gaps.get('suggested_additions', []))}")

        except Exception as e:
            print(f"\n[ERROR] Error in gap analysis: {e}")
    
    def _analyze_sessions(self):
        """Phase 4: Analyze session delivery."""
        try:
            sessions = self.analyzer.analyze_session_delivery()
            self.results['session_analysis'] = sessions
            
            print("\n[OK] Session analysis complete!")
            if 'totals' in sessions:
                print(f"   - Total sessions: {sessions['totals'].get('total_sessions', 0):.0f}")

        except Exception as e:
            print(f"\n[ERROR] Error in session analysis: {e}")
    
    def _run_nlp_analysis(self):
        """Phase 5: Run NLP analysis on feedback text."""
        try:
            if 'youth_feedback' in self.loader.clean_data:
                feedback_df = self.loader.clean_data['youth_feedback']
                
                # Load news data if available
                news_df = None
                if self.config['analyze_news'] and self.config['news_file']:
                    try:
                        news_df = pd.read_csv(self.config['news_file'])
                        print(f"   - Loaded news data: {len(news_df)} articles")
                    except:
                        print("   [WARNING] Could not load news data")
                
                # Run NLP analysis
                nlp_results = run_complete_nlp_analysis(feedback_df, news_df)
                self.results['nlp_analysis'] = nlp_results
                
                # Export NLP results separately
                nlp_output_path = f"{self.config['output_dir']}/breck_nlp_results.json"
                export_nlp_results(nlp_results, nlp_output_path)
                
                print("\n[OK] NLP analysis complete!")
            else:
                print("\n[WARNING] No youth feedback data for NLP analysis")

        except Exception as e:
            print(f"\n[ERROR] Error in NLP analysis: {e}")
            import traceback
            traceback.print_exc()
    
    def _run_geographic_analysis(self):
        """Phase 6: Run geographic clustering analysis."""
        try:
            if 'youth_feedback' in self.loader.clean_data:
                # Extract school list
                schools = self.loader.extract_school_list()
                
                if len(schools) > 0:
                    print(f"   - Found {len(schools)} unique schools")
                    
                    # Run geographic analysis
                    geo_results = run_geographic_analysis(
                        schools,
                        n_clusters=self.config['n_clusters']
                    )
                    self.results['geographic_analysis'] = geo_results
                    
                    print("\n[OK] Geographic analysis complete!")
                    if 'geocoded_schools' in geo_results:
                        print(f"   - Geocoded: {geo_results['geocoded_schools']}/{len(schools)} schools")
                else:
                    print("\n[WARNING] No schools identified for geographic analysis")
            else:
                print("\n[WARNING] No feedback data for geographic analysis")

        except Exception as e:
            print(f"\n[ERROR] Error in geographic analysis: {e}")
            import traceback
            traceback.print_exc()

        # Phase 6.5: Enhanced Lead Generation
        print("\n[6.5] PHASE 6.5: ENHANCED LEAD GENERATION")
        print("-" * 80)
        self._run_enhanced_lead_generation()

        # Phase 6.6: Interactive Map Generation
        print("\n[6.6] PHASE 6.6: INTERACTIVE MAP GENERATION")
        print("-" * 80)
        self._generate_interactive_map()
    
    def _generate_recommendations(self):
        """Phase 7: Generate actionable recommendations."""
        recommendations = {
            'data_collection': [],
            'workshop_content': [],
            'delivery_optimization': [],
            'geographic_expansion': []
        }
        
        # Data collection recommendations
        if 'data_gaps' in self.results:
            recommendations['data_collection'] = self.results['data_gaps'].get('suggested_additions', [])
        
        # Workshop content recommendations from NLP
        if 'nlp_analysis' in self.results:
            nlp = self.results['nlp_analysis']
            
            # From improvement suggestions
            if 'feedback_analysis' in nlp and 'improvements' in nlp['feedback_analysis']:
                improvements = nlp['feedback_analysis']['improvements'].get('improvement_themes', {})
                for theme, count in sorted(improvements.items(), key=lambda x: x[1], reverse=True)[:3]:
                    recommendations['workshop_content'].append(
                        f"Address '{theme}' feedback - mentioned by {count} students"
                    )
            
            # From positive feedback
            if 'feedback_analysis' in nlp and 'positive' in nlp['feedback_analysis']:
                positive = nlp['feedback_analysis']['positive'].get('themes', {})
                top_theme = max(positive.items(), key=lambda x: x[1])[0] if positive else None
                if top_theme:
                    recommendations['workshop_content'].append(
                        f"Continue emphasizing '{top_theme}' - most appreciated aspect"
                    )
        
        # Delivery optimization from effectiveness analysis
        if 'effectiveness_analysis' in self.results:
            eff = self.results['effectiveness_analysis']
            
            # Age-specific recommendations
            if 'by_age_group' in eff:
                recommendations['delivery_optimization'].append(
                    "Consider age-specific workshop adaptations based on effectiveness variations"
                )
            
            # External speaker impact
            if 'engagement_factors' in eff and 'external_speaker' in eff['engagement_factors']:
                recommendations['delivery_optimization'].append(
                    "Prioritize external speaker delivery - shows positive impact on engagement"
                )
        
        # Geographic recommendations
        if 'geographic_analysis' in self.results and 'coverage_gaps' in self.results['geographic_analysis']:
            gaps = self.results['geographic_analysis']['coverage_gaps']
            if gaps.get('schools_beyond_target', 0) > 0:
                recommendations['geographic_expansion'].append(
                    f"Identify {gaps['schools_beyond_target']} isolated schools for targeted outreach"
                )
        
        # Add general best practices
        recommendations['workshop_content'].extend([
            "Increase interactive elements (quiz, role-play scenarios)",
            "Add more visual content and real case examples",
            "Extend Q&A time for deeper discussion"
        ])
        
        self.results['recommendations'] = recommendations

        print("\n[OK] Recommendations generated!")
        print(f"   - Data collection: {len(recommendations['data_collection'])} items")
        print(f"   - Workshop content: {len(recommendations['workshop_content'])} items")
        print(f"   - Delivery optimization: {len(recommendations['delivery_optimization'])} items")
        print(f"   - Geographic expansion: {len(recommendations['geographic_expansion'])} items")

    def _run_enhanced_lead_generation(self):
        """
        Run enhanced lead generation with dual-track system:
        - Normal leads (trust + hotspot) for fundraising
        - Deprived area leads (high FSM%) for social impact
        """
        try:
            # Load Breck school summary with geocoding
            breck_summary_path = Path(self.config.get('breck_summary_file', 'outputs/school_summary.csv'))

            if breck_summary_path.exists():
                print(f"Loading Breck school summary from: {breck_summary_path}")
                breck_df = pd.read_csv(breck_summary_path)
            else:
                print("[WARNING] No school_summary.csv found, attempting to aggregate from loaded data...")
                breck_df = pd.DataFrame()

                if self.loader and hasattr(self.loader, 'clean_data') and 'youth_feedback' in self.loader.clean_data:
                    yf = self.loader.clean_data['youth_feedback']
                    agg = yf.groupby('school_name').size().reset_index(name='visit_count')
                    breck_df = agg
                    breck_df['latitude'] = None
                    breck_df['longitude'] = None
                    print(f"[OK] Aggregated {len(breck_df)} schools from youth feedback")
                else:
                    print("[WARNING] Cannot find Breck school list. Skipping lead generation.")
                    return

            # Try to load geocoded schools if available
            geocoded_path = Path(self.config['output_dir']) / 'schools_with_geocoding.csv'
            if geocoded_path.exists():
                geocoded_df = pd.read_csv(geocoded_path)
                # Merge geocoded coordinates into breck_df
                breck_df = breck_df.merge(
                    geocoded_df[['school_name', 'latitude', 'longitude']],
                    on='school_name',
                    how='left',
                    suffixes=('', '_geo')
                )
                # Use geocoded coordinates if available
                if 'latitude_geo' in breck_df.columns:
                    breck_df['latitude'] = breck_df['latitude_geo'].fillna(breck_df.get('latitude'))
                    breck_df['longitude'] = breck_df['longitude_geo'].fillna(breck_df.get('longitude'))
                    breck_df.drop(columns=['latitude_geo', 'longitude_geo'], inplace=True)

            gias_path = self.config.get('gias_database')
            if not gias_path:
                print("[WARNING] No 'gias_database' configured. Skipping lead generation.")
                return

            # Create enhanced lead generator
            lead_config = {
                'champion_min_visits': self.config.get('champion_min_visits', 2),
                'hotspot_radius_km': self.config.get('hotspot_radius_km', 5.0),
                'fsm_threshold': self.config.get('fsm_threshold', 30.0),
                'normal_leads_file': self.config.get('normal_leads_file', 'outputs/normal_leads.csv'),
                'deprived_leads_file': self.config.get('deprived_leads_file', 'outputs/deprived_area_leads.csv')
            }

            generator = EnhancedLeadGenerator(breck_df, gias_path, lead_config)
            normal_leads, deprived_leads = generator.run_complete_lead_generation()

            # Store results
            self.results['enhanced_lead_generation'] = {
                'normal_leads_count': len(normal_leads),
                'deprived_leads_count': len(deprived_leads),
                'normal_leads_file': lead_config['normal_leads_file'],
                'deprived_leads_file': lead_config['deprived_leads_file']
            }

            print(f"\n[OK] Enhanced lead generation complete!")
            print(f"  > Normal leads: {len(normal_leads)}")
            print(f"  > Deprived area leads: {len(deprived_leads)}")

        except Exception as e:
            print(f"\n[ERROR] Enhanced lead generation failed: {e}")
            import traceback
            traceback.print_exc()

    def _generate_interactive_map(self):
        """Generate the interactive map with all schools and leads."""
        try:
            map_config = {
                'visited_schools_file': self.config.get('visited_schools_file', 'outputs/schools_with_geocoding.csv'),
                'normal_leads_file': self.config.get('normal_leads_file', 'outputs/normal_leads.csv'),
                'deprived_leads_file': self.config.get('deprived_leads_file', 'outputs/deprived_area_leads.csv'),
                'map_output_path': self.config.get('map_output_path', 'outputs/breck_interactive_map.html'),
                'max_connections_per_lead': self.config.get('max_connections_per_lead', 2)
            }

            map_path = generate_interactive_map(map_config)

            self.results['interactive_map'] = {
                'map_file': map_path
            }

            print(f"\n[OK] Interactive map generated: {map_path}")

        except Exception as e:
            print(f"\n[ERROR] Interactive map generation failed: {e}")
            import traceback
            traceback.print_exc()
    
    def _export_all_results(self):
        """Phase 8: Export all results to files."""
        try:
            # Convert results to JSON-serializable format
            def convert_to_serializable(obj):
                """Convert pandas/numpy objects to JSON-serializable types."""
                if isinstance(obj, dict):
                    return {str(k): convert_to_serializable(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_to_serializable(item) for item in obj]
                elif hasattr(obj, 'to_dict'):
                    return convert_to_serializable(obj.to_dict())
                else:
                    return str(obj) if not isinstance(obj, (str, int, float, bool, type(None))) else obj

            serializable_results = convert_to_serializable(self.results)

            # Main results JSON
            main_output = f"{self.config['output_dir']}/breck_complete_analysis.json"
            with open(main_output, 'w') as f:
                json.dump(serializable_results, f, indent=2, default=str)

            print(f"\n[OK] Main results exported to: {main_output}")
            
            # Create summary report
            self._create_summary_report()
            
            # Create recommendations document
            self._create_recommendations_doc()

        except Exception as e:
            print(f"\n[ERROR] Error exporting results: {e}")
    
    def _create_summary_report(self):
        """Create a human-readable summary report."""
        report_path = f"{self.config['output_dir']}/breck_executive_summary.txt"
        
        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("BRECK FOUNDATION - EXECUTIVE SUMMARY\n")
            f.write("JP Morgan Data for Good Hackathon 2025\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Analysis Date: {self.results['timestamp']}\n\n")
            
            # Key findings
            f.write("KEY FINDINGS\n")
            f.write("-" * 80 + "\n\n")
            
            if 'data_summary' in self.results and 'key_metrics' in self.results['data_summary']:
                metrics = self.results['data_summary']['key_metrics']
                
                if 'youth_feedback' in metrics:
                    yf = metrics['youth_feedback']
                    f.write(f"Youth Feedback:\n")
                    f.write(f"  - Total responses: {yf.get('total_responses', 0)}\n")
                    f.write(f"  - Average effectiveness: {yf.get('avg_effectiveness_score', 0):.2f}/5\n")
                    f.write(f"  - Satisfaction rate: {yf.get('satisfaction_rate', 0):.1f}%\n")
                    f.write(f"  - Unique schools: {yf.get('unique_schools', 0)}\n\n")
                
                if 'sessions' in metrics:
                    sess = metrics['sessions']
                    f.write(f"Session Delivery:\n")
                    f.write(f"  - Total sessions: {sess.get('total_sessions', 0):.0f}\n")
                    f.write(f"  - Session types: {sess.get('session_types', 0)}\n\n")
            
            # Top recommendations
            f.write("\nTOP RECOMMENDATIONS\n")
            f.write("-" * 80 + "\n\n")
            
            if 'recommendations' in self.results:
                recs = self.results['recommendations']
                
                for category, items in recs.items():
                    if items:
                        f.write(f"{category.replace('_', ' ').title()}:\n")
                        for item in items[:3]:  # Top 3 per category
                            f.write(f"  - {item}\n")
                        f.write("\n")
            
            f.write("\n" + "=" * 80 + "\n")
        
        print(f"   - Summary report: {report_path}")
    
    def _create_recommendations_doc(self):
        """Create detailed recommendations document."""
        rec_path = f"{self.config['output_dir']}/breck_recommendations.txt"
        
        with open(rec_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("DETAILED RECOMMENDATIONS FOR BRECK FOUNDATION\n")
            f.write("=" * 80 + "\n\n")
            
            if 'recommendations' in self.results:
                for category, items in self.results['recommendations'].items():
                    f.write(f"\n{category.replace('_', ' ').upper()}\n")
                    f.write("-" * 80 + "\n\n")
                    
                    for i, item in enumerate(items, 1):
                        f.write(f"{i}. {item}\n")
                    
                    f.write("\n")
        
        print(f"   - Recommendations: {rec_path}")
    
    def _print_final_summary(self):
        """Print final summary to console."""
        print("\n" + "=" * 80)
        print("[SUCCESS] COMPLETE ANALYSIS FINISHED SUCCESSFULLY!")
        print("=" * 80)

        print("\n>> OUTPUT FILES:")
        print(f"   - Main results: {self.config['output_dir']}/breck_complete_analysis.json")
        print(f"   - NLP analysis: {self.config['output_dir']}/breck_nlp_results.json")
        print(f"   - Summary report: {self.config['output_dir']}/breck_executive_summary.txt")
        print(f"   - Recommendations: {self.config['output_dir']}/breck_recommendations.txt")

        if 'geographic_analysis' in self.results:
            if 'interactive_map' in self.results['geographic_analysis']:
                print(f"   - Geographic map: {self.results['geographic_analysis']['interactive_map']}")

        # Enhanced outputs
        print("\n>> ENHANCED LEAD GENERATION:")
        if 'enhanced_lead_generation' in self.results:
            elg = self.results['enhanced_lead_generation']
            print(f"   - Normal leads (fundraising): {elg.get('normal_leads_file', 'N/A')}")
            print(f"     > {elg.get('normal_leads_count', 0)} opportunities")
            print(f"   - Deprived area leads (impact): {elg.get('deprived_leads_file', 'N/A')}")
            print(f"     > {elg.get('deprived_leads_count', 0)} opportunities")

        print("\n>> INTERACTIVE MAP:")
        if 'interactive_map' in self.results:
            print(f"   -   {self.results['interactive_map'].get('map_file', 'N/A')}")
            print("   - Features:")
            print("     • Color-coded visited schools (1 visit, multiple visits, champions)")
            print("     • Normal leads (fundraising - blue markers)")
            print("     • Deprived area leads (high impact - red markers)")
            print("     • Connection lines to champion schools")
            print("     • Interactive popups with detailed information")

        print("\n>> NEXT STEPS:")
        print("   1. * Open the interactive map in your browser - IT'S STUNNING!")
        print("   2. Review normal leads CSV for fundraising outreach")
        print("   3. Review deprived area leads CSV for impact-focused outreach")
        print("   4. Examine detailed recommendations document")
        print("   5. Present findings to Breck Foundation team")
        print("   6. Use lead connections to warm up your outreach!")

        print("\n" + "=" * 80)
        print(" READY FOR HACKATHON PRESENTATION! ")
        print("=" * 80 + "\n")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main execution function.
    """
    try:
        # Create master analyzer
        master = BreckMasterAnalyzer(CONFIG)
        
        # Run complete analysis
        results = master.run_complete_analysis()
        
        return master
        
    except Exception as e:
        print(f"\n[CRITICAL ERROR] {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    master_analyzer = main()
