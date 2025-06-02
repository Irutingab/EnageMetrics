import streamlit as st
import pandas as pd
from data_manager import DataManager
from visualizations import Visualizations
from analytics import Analytics

class StudentDashboard:
    """Main dashboard class that orchestrates the entire application"""
    
    def __init__(self):
        self.data_manager = DataManager()
        self.visualizations = Visualizations()
        self.analytics = Analytics()
        self.setup_page_config()
        self.load_custom_css()
    
    def setup_page_config(self):
        """Configure Streamlit page settings"""
        st.set_page_config(
            page_title="Student Performance & Parental Engagement Dashboard",
            layout="wide",
            initial_sidebar_state="expanded"
        )
    
    def load_custom_css(self):
        """Load custom CSS styling"""
        st.markdown("""
        <style>
            .main > div {
                padding: 2rem 1rem;
            }
            .metric-card {
                background-color: #f8f9fa;
                padding: 1rem;
                border-radius: 0.5rem;
                border-left: 4px solid #1f77b4;
            }
            .correlation-strong { color: #28a745; font-weight: bold; }
            .correlation-moderate { color: #ffc107; font-weight: bold; }
            .correlation-weak { color: #dc3545; font-weight: bold; }
        </style>
        """, unsafe_allow_html=True)
    
    def render_sidebar(self, df):
        """Render sidebar filters and return filtered data"""
        st.sidebar.header("Dashboard Filters")
        
        # Parental involvement filter
        involvement_options = df['Parental_Involvement'].unique()
        selected_involvement = st.sidebar.multiselect(
            "Select Parental Involvement Levels",
            options=involvement_options,
            default=involvement_options
        )
        
        # School type filter
        school_options = df['School_Type'].unique()
        selected_schools = st.sidebar.multiselect(
            "Select School Types",
            options=school_options,
            default=school_options
        )
        
        return self.data_manager.apply_filters(df, selected_involvement, selected_schools)
    
    def render_executive_summary(self, filtered_df):
        """Render executive summary metrics"""
        st.header("Executive Summary")
        insights = self.analytics.get_performance_insights(filtered_df)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Students", f"{insights['total_students']:,}")
        with col2:
            st.metric("Average Exam Score", f"{insights['avg_score']:.1f}")
        with col3:
            st.metric("Average Attendance", f"{insights['avg_attendance']:.1f}%")
        with col4:
            st.metric("High Performers (≥80)", f"{insights['high_performers_pct']:.1f}%")
    
    def render_distribution_charts(self, filtered_df):
        """Render engagement and performance distribution charts"""
        st.header("Engagement & Performance Distribution")
        col1, col2 = st.columns(2)
        
        with col1:
            donut_involvement = self.visualizations.create_donut_chart(
                filtered_df, 'Parental_Involvement', 
                'Parental Involvement Distribution',
                colors=['#FF6B6B', '#4ECDC4', '#45B7D1']
            )
            st.pyplot(donut_involvement)
        
        with col2:
            histogram_performance = self.visualizations.create_histogram_chart(
                filtered_df, 'Performance_Category', 
                'Academic Performance Distribution',
                colors=['#FF6B6B', '#FFA07A', '#FFD700', '#98FB98', '#90EE90']
            )
            st.pyplot(histogram_performance)
        
        # Second row
        col1, col2 = st.columns(2)
        
        with col1:
            donut_attendance = self.visualizations.create_donut_chart(
                filtered_df, 'Attendance_Category', 
                'Attendance Level Distribution',
                colors=['#FF6B6B', '#FFD700', '#90EE90']
            )
            st.pyplot(donut_attendance)
        
        with col2:
            donut_study = self.visualizations.create_donut_chart(
                filtered_df, 'Study_Hours_Category', 
                'Study Hours Distribution',
                colors=['#FF6B6B', '#FFA07A', '#90EE90']
            )
            st.pyplot(donut_study)
    
    def render_performance_analysis(self, filtered_df):
        """Render performance distribution and advanced analysis"""
        st.header("Performance Distribution Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            hist_scores = self.visualizations.create_histogram_with_kde(
                filtered_df, 'Exam_Score', 'Distribution of Exam Scores'
            )
            st.pyplot(hist_scores)
        
        with col2:
            hist_hours = self.visualizations.create_histogram_with_kde(
                filtered_df, 'Hours_Studied', 'Distribution of Study Hours'
            )
            st.pyplot(hist_hours)
        
        # Advanced Analysis
        st.header("Advanced Performance Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            bar_chart = self.visualizations.create_bar_chart_scores_by_involvement(filtered_df)
            st.pyplot(bar_chart)
        
        with col2:
            scatter_plot = self.visualizations.create_scatter_attendance_vs_scores(filtered_df)
            st.pyplot(scatter_plot)
        
        # Box plots
        st.subheader("Score Distribution by Demographics")
        box_plots = self.visualizations.create_box_plot_scores_by_education(filtered_df)
        st.pyplot(box_plots)
    
    def render_correlation_analysis(self, filtered_df):
        """Render correlation analysis section"""
        st.header("Correlation Analysis")
        corr_fig = self.visualizations.create_correlation_heatmap(filtered_df)
        
        if corr_fig:
            st.pyplot(corr_fig)
            
            # Correlation insights
            correlation_data = self.analytics.get_correlation_insights(filtered_df)
            if correlation_data:
                st.subheader("Key Correlations with Exam Performance")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Strongest Positive Factors:**")
                    for item in correlation_data:
                        st.write(f"• {item['factor']}: {item['correlation']:.3f} ({item['strength']})")
                
                with col2:
                    involvement_stats = filtered_df.groupby('Parental_Involvement')['Exam_Score'].agg(['mean', 'count']).round(2)
                    st.write("**Performance by Parental Involvement:**")
                    st.dataframe(involvement_stats)
    
    def render_insights_and_recommendations(self, filtered_df):
        """Render key insights and recommendations"""
        st.header("Key Insights & Recommendations")
        insights = self.analytics.get_performance_insights(filtered_df)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Performance Insights")
            if 'involvement_impact' in insights:
                st.markdown(f"""
                <div class='metric-card'>
                    <strong>Parental Involvement Impact:</strong><br>
                    High involvement students score {insights['involvement_impact']:.1f} points higher on average
                    ({insights['high_involvement_mean']:.1f} vs {insights['low_involvement_mean']:.1f})
                </div>
                """, unsafe_allow_html=True)
            
            if 'high_perf_traits' in insights:
                st.write("**Common traits of high performers:**")
                for trait, percentage in insights['high_perf_traits'].items():
                    st.write(f"• {trait}: {percentage:.1f}%")
        
        with col2:
            st.subheader("Actionable Recommendations")
            recommendations = [
                "**Enhance Parental Engagement Programs**: Focus on involving parents in academic planning",
                "**Attendance Improvement Initiatives**: Target students with <85% attendance",
                "**Study Habits Workshop**: Promote effective study time management",
                "**School-Family Communication**: Strengthen regular progress updates",
                "**Peer Support Groups**: Connect high and low-performing students"
            ]
            
            for rec in recommendations:
                st.markdown(f"• {rec}")
    
    def render_parental_involvement_analysis(self, filtered_df):
        """Render detailed parental involvement analysis"""
        st.header("Parental Involvement Impact Analysis")
        
        try:
            correlation, p_value = self.analytics.analyze_parental_involvement_correlation(filtered_df)
            
            # Display correlation results
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Correlation Coefficient", 
                    f"{correlation:.3f}",
                    help="Ranges from -1 to 1. Values closer to 1 indicate stronger positive correlation."
                )
            
            with col2:
                significance = "Significant" if p_value < 0.05 else "Not Significant"
                st.metric(
                    "Statistical Significance", 
                    significance,
                    help=f"P-value: {p_value:.3f}. Significant if p < 0.05"
                )
            
            with col3:
                if abs(correlation) > 0.5:
                    strength = "Strong"
                    color = "🟢"
                elif abs(correlation) > 0.3:
                    strength = "Moderate" 
                    color = "🟡"
                else:
                    strength = "Weak"
                    color = "🔴"
                
                st.metric(
                    "Correlation Strength",
                    f"{color} {strength}",
                    help="Strong: |r| > 0.5, Moderate: |r| > 0.3, Weak: |r| ≤ 0.3"
                )
            
            # Detailed analysis charts
            st.subheader("📈 Detailed Analysis")
            involvement_analysis_fig = self.visualizations.create_parental_involvement_analysis(filtered_df)
            st.pyplot(involvement_analysis_fig)
            
        except Exception as e:
            st.error(f"Error in correlation analysis: {str(e)}")
    
    def render_export_section(self, filtered_df):
        """Render data export and summary section"""
        st.header("Export Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Download Filtered Data"):
                csv = filtered_df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name="student_performance_analysis.csv",
                    mime="text/csv"
                )
        
        with col2:
            insights = self.analytics.get_performance_insights(filtered_df)
            summary_stats = {
                'Metric': ['Total Students', 'Avg Exam Score', 'Avg Attendance', 'High Performers %', 'High Involvement %'],
                'Value': [
                    insights['total_students'],
                    f"{insights['avg_score']:.1f}",
                    f"{insights['avg_attendance']:.1f}",
                    f"{insights['high_performers_pct']:.1f}",
                    f"{insights['high_involvement_pct']:.1f}"
                ]
            }
            summary_df = pd.DataFrame(summary_stats)
            st.dataframe(summary_df)
    
    def run(self):
        """Main method to run the dashboard"""
        # Load data
        df = self.data_manager.get_processed_data()
        if df is None:
            return
        
        # Title and description
        st.title("Student Performance & Parental Engagement Dashboard")
        st.markdown("""
        This dashboard analyzes the relationship between parental involvement and student performance using 
        professional visualizations including donut charts, pie charts, and histograms to provide clear insights
        for educational decision-making.
        """)
        
        # Apply filters
        filtered_df = self.render_sidebar(df)
        
        # Render all sections
        self.render_executive_summary(filtered_df)
        self.render_distribution_charts(filtered_df)
        self.render_performance_analysis(filtered_df)
        self.render_correlation_analysis(filtered_df)
        self.render_insights_and_recommendations(filtered_df)
        self.render_parental_involvement_analysis(filtered_df)
        self.render_export_section(filtered_df)
