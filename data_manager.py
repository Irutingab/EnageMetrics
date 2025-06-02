import pandas as pd
import streamlit as st

class DataManager:
    """Handles data loading, caching, and preprocessing operations"""
    
    def __init__(self, filename="student_performance_cleaned.csv"):
        self.filename = filename
        self.df = None
    
    @st.cache_data
    def load_data(_self):
        """Load and cache the student performance data"""
        try:
            df = pd.read_csv(_self.filename)
            return df
        except FileNotFoundError:
            st.error(f"Dataset '{_self.filename}' not found. Please ensure the file is in the correct directory.")
            return None
    
    def categorize_data(self, df):
        """Create categories for better visualization"""
        # Create performance categories based on Exam_Score
        df['Performance_Category'] = pd.cut(df['Exam_Score'], 
                                        bins=[0, 60, 70, 80, 90, 100], 
                                        labels=['F (0-59)', 'D (60-69)', 'C (70-79)', 'B (80-89)', 'A (90-100)'])
        
        # Create attendance categories
        df['Attendance_Category'] = pd.cut(df['Attendance'], 
                                        bins=[0, 70, 85, 100], 
                                        labels=['Poor (≤70%)', 'Good (71-85%)', 'Excellent (>85%)'])
        
        # Create study hours categories
        df['Study_Hours_Category'] = pd.cut(df['Hours_Studied'], 
                                        bins=[0, 10, 20, 50], 
                                        labels=['Low (≤10h)', 'Medium (11-20h)', 'High (>20h)'])
        
        return df
    
    def get_processed_data(self):
        """Get fully processed data with categories"""
        if self.df is None:
            self.df = self.load_data()
            if self.df is not None:
                self.df = self.categorize_data(self.df)
        return self.df
    
    def apply_filters(self, df, selected_involvement, selected_schools):
        """Apply sidebar filters to the dataset"""
        return df[
            (df['Parental_Involvement'].isin(selected_involvement)) &
            (df['School_Type'].isin(selected_schools))
        ]
