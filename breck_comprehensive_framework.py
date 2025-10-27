"""
BRECK FOUNDATION COMPREHENSIVE DATA ANALYSIS FRAMEWORK
JP Morgan Data for Good Hackathon 2025
======================================================
Complete framework to analyze Breck Foundation data and generate
actionable insights to improve workshop impact and cyber safety education.

Author: Hackathon Team
Date: October 2025
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime
import re
import json
from typing import Dict, List, Tuple, Optional

# ============================================================================
# DATA LOADER - Handles all Excel sheets comprehensively
# ============================================================================

class BreckDataLoader:
    """
    Comprehensive data loader for all Breck Foundation data sources.
    Handles messy Excel structures and extracts all available information.
    """
    
    def __init__(self, main_file_path: str):
        """
        Initialize the data loader.
        
        Args:
            main_file_path: Path to Breck_Internal_Data.xlsx
        """
        self.main_file_path = main_file_path
        self.raw_data = {}
        self.clean_data = {}
        self.metadata = {}
        
    def load_all_sheets(self) -> 'BreckDataLoader':
        """
        Load all sheets from the main Excel file.
        """
        print("Loading all data sheets...")
        
        try:
            excel_file = pd.ExcelFile(self.main_file_path)
            
            for sheet_name in excel_file.sheet_names:
                try:
                    self.raw_data[sheet_name] = pd.read_excel(
                        self.main_file_path, 
                        sheet_name=sheet_name,
                        header=None  # We'll handle headers manually
                    )
                    print(f"{sheet_name}: {self.raw_data[sheet_name].shape}")
                except Exception as e:
                    print(f"Error loading {sheet_name}: {e}")
            
            print(f"Successfully loaded {len(self.raw_data)} sheets!\n")
            
        except Exception as e:
            print(f"Error loading Excel file: {e}")
            raise
        
        return self
    
    def parse_session_data(self, sheet_name: str) -> pd.DataFrame:
        """
        Parse session tracking sheets (monthly data with complex headers).
        
        Args:
            sheet_name: Name of the session tracking sheet
            
        Returns:
            Cleaned DataFrame with session data
        """
        df = self.raw_data[sheet_name].copy()
        
        # Extract month headers from row 1 (dates)
        months = []
        session_types = []
        
        # Find date columns (every 3 columns typically)
        for i in range(0, len(df.columns), 3):
            if i < len(df.columns):
                date_val = df.iloc[1, i]
                if pd.notna(date_val) and 'Unnamed' not in str(date_val):
                    try:
                        date_obj = pd.to_datetime(date_val)
                        months.append(date_obj)
                    except:
                        pass
        
        # Extract actual session data (skip first 3 header rows)
        session_data = df.iloc[3:].copy()
        
        # Restructure into long format
        records = []
        for idx, row in session_data.iterrows():
            session_type = row.iloc[0]
            if pd.isna(session_type) or session_type == '':
                continue
                
            col_idx = 1
            for month_date in months:
                if col_idx < len(row):
                    count = row.iloc[col_idx]
                    if pd.notna(count):
                        try:
                            count = float(count)
                            records.append({
                                'month': month_date,
                                'session_type': session_type,
                                'count': count,
                                'data_source': sheet_name
                            })
                        except:
                            pass
                col_idx += 3  # Skip to next month (3 columns per month typically)
        
        return pd.DataFrame(records)
    
    def clean_young_person_feedback(self) -> 'BreckDataLoader':
        """
        Clean and structure youth feedback data (main survey responses).
        """
        print("Cleaning youth feedback data...")
        
        df = self.raw_data['Feedback Young Person'].copy()
        
        # Set proper column names (first row)
        df.columns = df.iloc[0]
        df = df.iloc[1:].reset_index(drop=True)
        
        # Convert dates
        df['Submission Date'] = pd.to_datetime(df['Submission Date'], errors='coerce')
        
        # Clean age groups
        def clean_age(age_str):
            if pd.isna(age_str):
                return None
            age_str = str(age_str)
            if '11 - 13' in age_str or '11-13' in age_str:
                return '11-13'
            elif '14 - 16' in age_str or '14-16' in age_str:
                return '14-16'
            elif '16+' in age_str:
                return '16+'
            elif '10 or under' in age_str:
                return '10 or under'
            return age_str
        
        df['age_group'] = df['How old are you?'].apply(clean_age)
        
        # Clean gender
        def clean_gender(gender_str):
            if pd.isna(gender_str):
                return None
            gender_str = str(gender_str).lower()
            if 'male' in gender_str and 'female' not in gender_str:
                return 'Male'
            elif 'female' in gender_str:
                return 'Female'
            elif 'prefer not' in gender_str:
                return 'Prefer not to say'
            elif 'other' in gender_str:
                return 'Other'
            return gender_str.title()
        
        df['gender'] = df['What gender are you?'].apply(clean_gender)
        
        # Clean effectiveness rating
        rating_col = 'How would you rate the overall effectiveness of the presentation in raising awareness about online safety and grooming?'
        def clean_rating(rating_str):
            if pd.isna(rating_str):
                return None
            rating_str = str(rating_str)
            if 'Excellent' in rating_str:
                return 'Excellent'
            elif 'Good' in rating_str:
                return 'Good'
            elif 'Fair' in rating_str:
                return 'Fair'
            elif 'Poor' in rating_str:
                return 'Poor'
            return None
        
        df['effectiveness_rating'] = df[rating_col].apply(clean_rating)
        
        # Convert to numeric score
        rating_map = {'Excellent': 5, 'Good': 4, 'Fair': 3, 'Poor': 2, 'Very Poor': 1}
        df['effectiveness_score'] = df['effectiveness_rating'].map(rating_map)
        
        # Clean binary responses
        def clean_binary(response):
            if pd.isna(response):
                return None
            response = str(response).lower()
            return 'Yes' if 'yes' in response else ('No' if 'no' in response else None)
        
        speaker_col = 'Did you find the presentation more engaging because it was delivered by an external speaker?'
        df['external_speaker_engaging'] = df[speaker_col].apply(clean_binary)
        
        # Extract confidence metrics (assuming 1-5 Likert scale)
        df['confident_identifying_dangers'] = pd.to_numeric(
            df['I feel more confident in identifying dangers online'], 
            errors='coerce'
        )
        df['understanding_grooming'] = pd.to_numeric(
            df['I have a better understanding of grooming and how to recognise it'], 
            errors='coerce'
        )
        df['understanding_online_vs_real'] = pd.to_numeric(
            df['I understand the difference between online friends and friends in real life'], 
            errors='coerce'
        )
        
        # Would recommend
        df['would_recommend'] = df['Would you recommend this session to other students your age?'].apply(clean_binary)
        
        # Extract school names for geographic analysis
        df['school_name'] = df['What is the name of your school?']
        
        # Text feedback columns
        df['liked_most'] = df['What parts did you like the most or find the most interesting or useful?']
        df['improvements'] = df['Is there anything you feel was missing or could be improved in the presentation? If so, please provide details.']
        df['additions'] = df['What could we add to the presentation to make it better?']
        
        self.clean_data['youth_feedback'] = df

        print(f"   [OK] Cleaned {len(df)} youth responses")
        print(f"   [OK] Date range: {df['Submission Date'].min()} to {df['Submission Date'].max()}")
        print(f"   [OK] Average effectiveness: {df['effectiveness_score'].mean():.2f}/5")
        print()
        
        return self
    
    def clean_adult_feedback(self) -> 'BreckDataLoader':
        """
        Clean adult (teacher/staff) feedback data.
        """
        print("Cleaning adult feedback data...")
        
        df = self.raw_data['Feedback Adult'].copy()
        
        # Set proper headers (skip first 2 rows, use row 3)
        df.columns = df.iloc[2]
        df = df.iloc[3:].reset_index(drop=True)
        
        # Remove empty rows
        df = df.dropna(how='all')
        
        self.clean_data['adult_feedback'] = df
        
        print(f"Cleaned {len(df)} adult responses")
        print()
        
        return self
    
    def clean_class_feedback(self) -> 'BreckDataLoader':
        """
        Clean class-level feedback data.
        """
        print("Cleaning class feedback data...")
        
        df = self.raw_data['Feedback Class'].copy()
        
        # Set proper headers
        df.columns = df.iloc[0]
        df = df.iloc[1:].reset_index(drop=True)
        
        # Convert date if column exists
        if 'Submission Date' in df.columns:
            df['Submission Date'] = pd.to_datetime(df['Submission Date'], errors='coerce')
        
        self.clean_data['class_feedback'] = df
        
        print(f"Cleaned {len(df)} class responses")
        print()
        
        return self
    
    def parse_all_session_data(self) -> 'BreckDataLoader':
        """
        Parse all session tracking sheets into unified format.
        """
        print("Parsing session data...")
        
        session_sheets = [
            'BF Staff Core Funded 24-25',
            'BF Staff Core Paid 24-25',
            'Freelance Staff Session 24-25',
            'RISE e2e Sessions 24-25',
            'Game Over Sessions 24-25'
        ]
        
        all_sessions = []
        
        for sheet_name in session_sheets:
            if sheet_name in self.raw_data:
                try:
                    session_df = self.parse_session_data(sheet_name)
                    if len(session_df) > 0:
                        all_sessions.append(session_df)
                    print(f"{sheet_name}: {len(session_df)} records")
                except Exception as e:
                    print(f"Error parsing {sheet_name}: {e}")
        
        if all_sessions:
            combined = pd.concat(all_sessions, ignore_index=True)
            if 'month' in combined.columns:
                combined = combined.sort_values('month')
            self.clean_data['all_sessions'] = combined
            
            print(f"\n  Total session records: {len(combined)}")
            if len(combined) > 0:
                print(f" Date range: {combined['month'].min()} to {combined['month'].max()}")
                print(f"  Total sessions delivered: {combined['count'].sum():.0f}")
        else:
            # Create empty dataframe with proper structure
            self.clean_data['all_sessions'] = pd.DataFrame(columns=['month', 'session_type', 'count', 'data_source'])
            print(f"\n No session data could be parsed (complex Excel structure)")
        
        print()
        return self
    
    def clean_qualitative_feedback(self) -> 'BreckDataLoader':
        """
        Clean non-numerical (qualitative) feedback from various stakeholders.
        """
        print("Cleaning qualitative feedback...")
        
        df = self.raw_data['Non-Numerical Feedback'].copy()
        
        # Set proper headers
        df.columns = df.iloc[0]
        df = df.iloc[1:].reset_index(drop=True)
        
        # Convert date if column exists
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        
        # Remove empty rows (check for Feedback column if it exists)
        feedback_cols = [col for col in df.columns if 'feedback' in str(col).lower()]
        if feedback_cols:
            df = df.dropna(subset=[feedback_cols[0]])
        
        self.clean_data['qualitative_feedback'] = df
        
        print(f"Cleaned {len(df)} qualitative responses")
        print()
        
        return self
    
    def extract_school_list(self) -> List[str]:
        """
        Extract unique list of schools from feedback data.
        
        Returns:
            List of unique school names
        """
        schools = set()
        
        if 'youth_feedback' in self.clean_data:
            youth_schools = self.clean_data['youth_feedback']['school_name'].dropna().unique()
            schools.update(youth_schools)
        
        if 'adult_feedback' in self.clean_data:
            # Adult feedback might have school name in first column
            adult_df = self.clean_data['adult_feedback']
            if len(adult_df.columns) > 0:
                first_col = adult_df.iloc[:, 0].dropna().unique()
                schools.update([s for s in first_col if isinstance(s, str) and len(s) > 3])
        
        return sorted([s for s in schools if s])
    
    def get_summary(self) -> Dict:
        """
        Generate comprehensive summary of all loaded data.
        
        Returns:
            Dictionary with summary statistics
        """
        summary = {
            'timestamp': datetime.now().isoformat(),
            'data_sources': {},
            'key_metrics': {}
        }
        
        # Summarize each data source
        for key, df in self.clean_data.items():
            summary['data_sources'][key] = {
                'rows': len(df),
                'columns': len(df.columns),
                'date_range': None
            }
            
            # Try to extract date range
            date_cols = [col for col in df.columns if isinstance(col, str) and 'date' in col.lower()]
            if date_cols:
                try:
                    dates = pd.to_datetime(df[date_cols[0]], errors='coerce')
                    summary['data_sources'][key]['date_range'] = {
                        'start': str(dates.min()),
                        'end': str(dates.max())
                    }
                except:
                    pass
        
        # Key metrics from youth feedback
        if 'youth_feedback' in self.clean_data:
            yf = self.clean_data['youth_feedback']
            summary['key_metrics']['youth_feedback'] = {
                'total_responses': len(yf),
                'avg_effectiveness_score': float(yf['effectiveness_score'].mean()),
                'satisfaction_rate': float((yf['effectiveness_rating'].isin(['Excellent', 'Good']).sum() / len(yf) * 100)),
                'recommendation_rate': float((yf['would_recommend'] == 'Yes').sum() / yf['would_recommend'].notna().sum() * 100) if yf['would_recommend'].notna().sum() > 0 else None,
                'unique_schools': len(yf['school_name'].dropna().unique())
            }
        
        # Session metrics
        if 'all_sessions' in self.clean_data and len(self.clean_data['all_sessions']) > 0:
            sessions = self.clean_data['all_sessions']
            summary['key_metrics']['sessions'] = {
                'total_sessions': float(sessions['count'].sum()),
                'session_types': len(sessions['session_type'].unique()),
                'months_active': len(sessions['month'].unique())
            }
        
        return summary


# ============================================================================
# COMPREHENSIVE ANALYZER - Main analysis engine
# ============================================================================

class BreckComprehensiveAnalyzer:
    """
    Comprehensive analyzer for all Breck Foundation data.
    """
    
    def __init__(self, data_loader: BreckDataLoader):
        """
        Initialize analyzer with loaded data.
        
        Args:
            data_loader: BreckDataLoader instance with cleaned data
        """
        self.loader = data_loader
        self.data = data_loader.clean_data
        self.results = {}
    
    def analyze_program_effectiveness(self) -> Dict:
        """
        Analyze overall program effectiveness across all metrics.
        
        Returns:
            Dictionary with effectiveness analysis
        """
        print("PROGRAM EFFECTIVENESS ANALYSIS")
        print("=" * 80)
        
        if 'youth_feedback' not in self.data:
            print("Youth feedback data not available")
            return {}
        
        df = self.data['youth_feedback']
        
        results = {
            'overall_metrics': {},
            'by_age_group': {},
            'by_gender': {},
            'temporal_trends': {},
            'engagement_factors': {}
        }
        
        # Overall metrics
        print("\n1. OVERALL EFFECTIVENESS:")
        results['overall_metrics'] = {
            'sample_size': len(df),
            'avg_effectiveness_score': df['effectiveness_score'].mean(),
            'satisfaction_rate': (df['effectiveness_rating'].isin(['Excellent', 'Good']).sum() / len(df) * 100),
            'recommendation_rate': (df['would_recommend'] == 'Yes').sum() / df['would_recommend'].notna().sum() * 100 if df['would_recommend'].notna().sum() > 0 else None
        }
        
        print(f"   - Sample size: {results['overall_metrics']['sample_size']}")
        print(f"   - Average effectiveness: {results['overall_metrics']['avg_effectiveness_score']:.2f}/5")
        print(f"   - Satisfaction rate: {results['overall_metrics']['satisfaction_rate']:.1f}%")
        if results['overall_metrics']['recommendation_rate']:
            print(f"   - Recommendation rate: {results['overall_metrics']['recommendation_rate']:.1f}%")
        
        # By age group
        print("\n2. EFFECTIVENESS BY AGE GROUP:")
        age_analysis = df.groupby('age_group')['effectiveness_score'].agg([
            ('mean', 'mean'),
            ('count', 'count'),
            ('std', 'std')
        ]).round(2)
        print(age_analysis)
        results['by_age_group'] = age_analysis.to_dict()
        
        # By gender
        print("\n3. EFFECTIVENESS BY GENDER:")
        gender_analysis = df.groupby('gender')['effectiveness_score'].agg([
            ('mean', 'mean'),
            ('count', 'count'),
            ('std', 'std')
        ]).round(2)
        print(gender_analysis)
        results['by_gender'] = gender_analysis.to_dict()
        
        # Temporal trends
        print("\n4. TEMPORAL TRENDS:")
        df['month'] = df['Submission Date'].dt.to_period('M')
        temporal = df.groupby('month')['effectiveness_score'].agg([
            ('mean', 'mean'),
            ('count', 'count')
        ]).round(2)
        print(temporal.tail(6))  # Show last 6 months
        results['temporal_trends'] = temporal.to_dict()
        
        # Impact of external speaker
        print("\n5. EXTERNAL SPEAKER IMPACT:")
        if df['external_speaker_engaging'].notna().any():
            speaker_impact = df.groupby('external_speaker_engaging')['effectiveness_score'].agg([
                ('mean', 'mean'),
                ('count', 'count')
            ]).round(2)
            print(speaker_impact)
            results['engagement_factors']['external_speaker'] = speaker_impact.to_dict()
        
        # Confidence metrics
        print("\n6. LEARNING OUTCOMES (Average Scores):")
        confidence_metrics = {
            'identifying_dangers': df['confident_identifying_dangers'].mean(),
            'understanding_grooming': df['understanding_grooming'].mean(),
            'online_vs_real_friends': df['understanding_online_vs_real'].mean()
        }
        for metric, value in confidence_metrics.items():
            print(f"   - {metric.replace('_', ' ').title()}: {value:.2f}")
        results['overall_metrics']['learning_outcomes'] = confidence_metrics
        
        print("\n" + "=" * 80 + "\n")
        
        self.results['effectiveness'] = results
        return results
    
    def analyze_data_collection_gaps(self) -> Dict:
        """
        Identify gaps in current data collection methods.
        
        Returns:
            Dictionary with identified gaps and recommendations
        """
        print("DATA COLLECTION GAPS ANALYSIS")
        print("=" * 80)
        
        gaps = {
            'missing_data_points': [],
            'low_response_fields': [],
            'suggested_additions': [],
            'data_quality_issues': []
        }
        
        # Analyze youth feedback completeness
        if 'youth_feedback' in self.data:
            df = self.data['youth_feedback']
            
            print("\n1. RESPONSE COMPLETENESS:")
            for col in df.columns:
                missing_pct = (df[col].isna().sum() / len(df) * 100)
                if missing_pct > 20:
                    gaps['low_response_fields'].append({
                        'field': col,
                        'missing_percentage': missing_pct
                    })
                    print(f"   - {col}: {missing_pct:.1f}% missing")
        
        # Check for pre/post comparison capability
        print("\n2. PRE/POST ASSESSMENT:")
        if 'YA Pre Assembly Feedback' in self.loader.raw_data:
            print("   - Pre-assembly data available")
        else:
            gaps['missing_data_points'].append({
                'category': 'Pre-assessment',
                'description': 'No baseline knowledge assessment before workshops',
                'impact': 'Cannot measure knowledge gain'
            })
            print("   - No pre-assembly baseline data")
        
        # Check for follow-up data
        print("\n3. LONG-TERM IMPACT:")
        gaps['missing_data_points'].append({
            'category': 'Follow-up',
            'description': 'No 3-month or 6-month follow-up surveys',
            'impact': 'Cannot measure behavior change or knowledge retention'
        })
        print("   - No long-term follow-up data")
        
        # Geographic data
        print("\n4. GEOGRAPHIC DATA:")
        if 'youth_feedback' in self.data:
            schools_with_location = 0  # We'd need geocoded data
            print(f"   - Schools identified: {len(self.loader.extract_school_list())}")
            print("   - No geographic coordinates for mapping")
            gaps['missing_data_points'].append({
                'category': 'Geographic',
                'description': 'School locations not geocoded',
                'impact': 'Cannot create heat maps or analyze geographic patterns'
            })
        
        # Behavioral indicators
        print("\n5. BEHAVIORAL INDICATORS:")
        gaps['missing_data_points'].append({
            'category': 'Behavioral',
            'description': 'No data on actual behavior change (reporting incidents, changing privacy settings)',
            'impact': 'Cannot measure real-world application of learnings'
        })
        print("   - No behavioral outcome tracking")
        
        # Recommendations
        print("\n6. RECOMMENDED ADDITIONS:")
        recommendations = [
            "Implement pre/post matched surveys with unique IDs",
            "Add 3-month follow-up survey to measure behavior change",
            "Collect school postcodes for geographic analysis",
            "Add questions about specific actions taken (e.g., 'Have you changed your privacy settings?')",
            "Include parental feedback surveys",
            "Track incident reporting rates in participating schools",
            "Add more demographic variables (ethnicity, SEN status) for equity analysis",
            "Include questions about specific online platforms used"
        ]
        
        gaps['suggested_additions'] = recommendations
        for rec in recommendations:
            print(f"   - {rec}")
        
        print("\n" + "=" * 80 + "\n")
        
        self.results['data_gaps'] = gaps
        return gaps
    
    def analyze_session_delivery(self) -> Dict:
        """
        Analyze session delivery patterns and capacity.
        
        Returns:
            Dictionary with session analysis
        """
        print("SESSION DELIVERY ANALYSIS")
        print("=" * 80)
        
        if 'all_sessions' not in self.data:
            print("Session data not available")
            return {}
        
        sessions = self.data['all_sessions']
        
        results = {
            'totals': {},
            'by_type': {},
            'by_source': {},
            'trends': {}
        }
        
        # Overall totals
        print("\n1. OVERALL SESSION DELIVERY:")
        results['totals'] = {
            'total_sessions': sessions['count'].sum(),
            'unique_session_types': len(sessions['session_type'].unique()),
            'months_active': len(sessions['month'].unique()),
            'data_sources': len(sessions['data_source'].unique())
        }
        print(f"   - Total sessions: {results['totals']['total_sessions']:.0f}")
        print(f"   - Session types: {results['totals']['unique_session_types']}")
        print(f"   - Active months: {results['totals']['months_active']}")
        
        # By session type
        print("\n2. BY SESSION TYPE:")
        by_type = sessions.groupby('session_type')['count'].sum().sort_values(ascending=False)
        print(by_type)
        results['by_type'] = by_type.to_dict()
        
        # By funding source
        print("\n3. BY DATA SOURCE:")
        by_source = sessions.groupby('data_source')['count'].sum().sort_values(ascending=False)
        print(by_source)
        results['by_source'] = by_source.to_dict()
        
        # Monthly trends
        print("\n4. MONTHLY DELIVERY TRENDS:")
        monthly = sessions.groupby('month')['count'].sum().sort_index()
        print(monthly.tail(6))
        results['trends']['monthly'] = monthly.to_dict()
        
        print("\n" + "=" * 80 + "\n")
        
        self.results['sessions'] = results
        return results
    
    def export_results(self, output_path: str = '/home/claude/breck_comprehensive_results.json'):
        """
        Export all analysis results to JSON.
        
        Args:
            output_path: Path for output JSON file
        """
        # Convert results to JSON-serializable format
        output = {
            'timestamp': datetime.now().isoformat(),
            'summary': self.loader.get_summary(),
            'analysis_results': self.results
        }
        
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2, default=str)
        
        print(f"Results exported to: {output_path}")
        return output_path


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main execution function - runs complete analysis pipeline.
    """
    print("=" * 80)
    print("BRECK FOUNDATION COMPREHENSIVE DATA ANALYSIS")
    print("   JP Morgan Data for Good Hackathon 2025")
    print("=" * 80)
    print()
    
    # File path
    data_file = '/mnt/user-data/uploads/Breck_Internal_Data.xlsx'
    
    # 1. LOAD ALL DATA
    print("PHASE 1: DATA LOADING")
    print("-" * 80)
    loader = BreckDataLoader(data_file)
    loader.load_all_sheets()
    loader.clean_young_person_feedback()
    loader.clean_adult_feedback()
    loader.clean_class_feedback()
    loader.parse_all_session_data()
    loader.clean_qualitative_feedback()
    
    print("\nDATA SUMMARY:")
    print("-" * 80)
    summary = loader.get_summary()
    print(json.dumps(summary, indent=2, default=str))
    print()
    
    # 2. COMPREHENSIVE ANALYSIS
    print("\nPHASE 2: COMPREHENSIVE ANALYSIS")
    print("-" * 80)
    analyzer = BreckComprehensiveAnalyzer(loader)
    analyzer.analyze_program_effectiveness()
    analyzer.analyze_data_collection_gaps()
    analyzer.analyze_session_delivery()
    
    # 3. EXPORT RESULTS
    print("\nPHASE 3: EXPORT RESULTS")
    print("-" * 80)
    output_path = analyzer.export_results()
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE!")
    print("=" * 80)
    print(f"\nResults saved to: {output_path}")
    print("\nNext steps:")
    print("  1. Run NLP analysis on feedback text")
    print("  2. Create geographic visualization of schools")
    print("  3. Analyze external cyber safety trends")
    print("  4. Generate recommendations report")
    
    return loader, analyzer


if __name__ == "__main__":
    loader, analyzer = main()
