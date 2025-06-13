import pandas as pd
import numpy as np

class Analytics:
    """Handles statistical analysis and calculations"""
    
    @staticmethod
    def analyze_parental_involvement_correlation(df):
        """Analyze correlation between parental involvement and student performance"""
        # Convert parental involvement to numeric for correlation
        involvement_mapping = {'Low': 1, 'Medium': 2, 'High': 3}
        df_analysis = df.copy()
        df_analysis['Parental_Involvement_Numeric'] = df_analysis['Parental_Involvement'].map(involvement_mapping)
        
        # Calculate correlation coefficient using pandas
        correlation = df_analysis['Parental_Involvement_Numeric'].corr(df_analysis['Exam_Score'])
        
        # Simple significance estimation based on correlation strength and sample size
        n = len(df_analysis)
        if n > 30:  # Large sample
            p_value = 0.001 if abs(correlation) > 0.3 else 0.05 if abs(correlation) > 0.2 else 0.1
        else:  # Small sample
            p_value = 0.01 if abs(correlation) > 0.5 else 0.1
        
        return correlation, p_value
    
    @staticmethod
    def get_performance_insights(df):
        """Get key performance insights and statistics"""
        insights = {}
        
        # Basic statistics
        insights['total_students'] = len(df)
        insights['avg_score'] = df['Exam_Score'].mean()
        insights['avg_attendance'] = df['Attendance'].mean()
        insights['high_performers_pct'] = (df['Exam_Score'] >= 70).mean() * 100
        insights['high_involvement_pct'] = (df['Parental_Involvement'] == 'High').mean() * 100
        
        # Parental involvement impact
        high_involvement = df[df['Parental_Involvement'] == 'High']['Exam_Score'].mean()
        low_involvement = df[df['Parental_Involvement'] == 'Low']['Exam_Score'].mean()
        
        if not pd.isna(high_involvement) and not pd.isna(low_involvement):
            insights['involvement_impact'] = high_involvement - low_involvement
            insights['high_involvement_mean'] = high_involvement
            insights['low_involvement_mean'] = low_involvement
        
        # High performer traits
        high_performers = df[df['Exam_Score'] >= 70]
        if len(high_performers) > 0:
            insights['high_perf_traits'] = {
                'High Parental Involvement': (high_performers['Parental_Involvement'] == 'High').mean() * 100,
                'Excellent Attendance': (high_performers['Attendance'] > 85).mean() * 100,
                'High Study Hours': (high_performers['Hours_Studied'] > 20).mean() * 100
            }
        return insights
    
    @staticmethod
    def get_correlation_insights(df):
        """Get correlation insights for numeric features"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if 'Exam_Score' not in numeric_cols:
            return None
            
        correlations = df[numeric_cols].corr()['Exam_Score'].abs().sort_values(ascending=False)
        
        # Get top factors (excluding Exam_Score itself)
        top_factors = correlations.drop('Exam_Score').head(5)
        
        correlation_data = []
        for factor, corr in top_factors.items():
            strength = "Strong" if corr > 0.3 else "Moderate" if corr > 0.1 else "Weak"
            correlation_data.append({
                'factor': factor,
                'correlation': corr,
                'strength': strength
            })
        
        return correlation_data
