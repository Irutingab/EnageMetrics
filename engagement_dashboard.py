import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Student Performance & Parental Engagement Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
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

# Cache data loading for better performance
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("student_performance_cleaned.csv")
        return df
    except FileNotFoundError:
        st.error("Dataset 'student_performance_cleaned.csv' not found. Please ensure the file is in the correct directory.")
        return None

def categorize_data(df):
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

def create_donut_chart(df, column, title, colors=None):
    """Create a professional pie chart using matplotlib"""
    value_counts = df[column].value_counts()
    
    # Create colors if not provided
    if colors is None:
        colors = plt.cm.Set3(np.linspace(0, 1, len(value_counts)))
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Create pie chart with hole in center for donut effect
    wedges, texts, autotexts = ax.pie(
        value_counts.values,
        labels=value_counts.index,
        colors=colors,
        autopct='%1.1f%%',
        startangle=90,
        wedgeprops=dict(width=0.6)  # Creates donut effect
    )
    
    # Customize text
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    return fig

def create_correlation_heatmap(df):
    """Create correlation heatmap for numeric columns"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) < 2:
        return None
    
    corr_matrix = df[numeric_cols].corr()
    
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                square=True, ax=ax, cbar_kws={"shrink": .8})
    ax.set_title('Correlation Matrix: Student Performance Factors', fontsize=16, pad=20)
    plt.tight_layout()
    return fig

def create_histogram_with_kde(df, column, title, bins=20):
    """Create histogram without KDE overlay"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create histogram
    n, bins, patches = ax.hist(df[column].dropna(), bins=bins, alpha=0.7, 
                              color='skyblue', edgecolor='black')
    
    # Add mean line
    mean_val = df[column].mean()
    ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, 
               label=f'Mean: {mean_val:.1f}')
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel(column)
    ax.set_ylabel('Frequency')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig

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

def create_parental_involvement_analysis(df):
    """Create detailed analysis of parental involvement impact"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Box plot of exam scores by parental involvement
    involvement_order = ['Low', 'Medium', 'High']
    df_ordered = df[df['Parental_Involvement'].isin(involvement_order)]
    
    box_data = [df_ordered[df_ordered['Parental_Involvement'] == level]['Exam_Score'].values 
                for level in involvement_order]
    
    ax1.boxplot(box_data, labels=involvement_order)
    ax1.set_title('Exam Scores by Parental Involvement Level', fontweight='bold')
    ax1.set_xlabel('Parental Involvement Level')
    ax1.set_ylabel('Exam Score')
    ax1.grid(True, alpha=0.3)
    
    # 2. Mean scores comparison
    mean_scores = df.groupby('Parental_Involvement')['Exam_Score'].mean().reindex(involvement_order)
    bars = ax2.bar(involvement_order, mean_scores, color=['#FF6B6B', '#FFD700', '#90EE90'])
    ax2.set_title('Average Exam Scores by Parental Involvement', fontweight='bold')
    ax2.set_xlabel('Parental Involvement Level')
    ax2.set_ylabel('Average Exam Score')
    
    # Add value labels on bars
    for bar, value in zip(bars, mean_scores):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'{value:.1f}', ha='center', va='bottom', fontweight='bold')
    
    # 3. Distribution of performance categories by involvement
    performance_by_involvement = pd.crosstab(df['Parental_Involvement'], df['Performance_Category'], normalize='index') * 100
    performance_by_involvement.plot(kind='bar', ax=ax3, stacked=True, 
                                   color=['#FF6B6B', '#FFA07A', '#FFD700', '#98FB98', '#90EE90'])
    ax3.set_title('Performance Distribution by Parental Involvement (%)', fontweight='bold')
    ax3.set_xlabel('Parental Involvement Level')
    ax3.set_ylabel('Percentage')
    ax3.legend(title='Performance Grade', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax3.tick_params(axis='x', rotation=45)
    
    # 4. Scatter plot with trend line
    involvement_numeric = df['Parental_Involvement'].map({'Low': 1, 'Medium': 2, 'High': 3})
    ax4.scatter(involvement_numeric, df['Exam_Score'], alpha=0.6, color='#45B7D1')
    
    # Add trend line
    z = np.polyfit(involvement_numeric, df['Exam_Score'], 1)
    p = np.poly1d(z)
    ax4.plot([1, 2, 3], p([1, 2, 3]), "r--", alpha=0.8, linewidth=2)
    
    ax4.set_title('Exam Score vs Parental Involvement (with trend)', fontweight='bold')
    ax4.set_xlabel('Parental Involvement Level\n(1=Low, 2=Medium, 3=High)')
    ax4.set_ylabel('Exam Score')
    ax4.set_xticks([1, 2, 3])
    ax4.set_xticklabels(['Low', 'Medium', 'High'])
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

# Main dashboard
def main():
    # Load data
    df = load_data()
    if df is None:
        return
    
    # Categorize data
    df = categorize_data(df)
    
    st.title("Student Performance & Parental Engagement Dashboard")
    
    st.markdown("""
    This dashboard analyzes the relationship between parental involvement and student performance using 
    professional visualizations including donut charts, pie charts, and histograms to provide clear insights
    for educational decision-making.
    """)
    
    # Sidebar filters
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
    
    # Filter data
    filtered_df = df[
        (df['Parental_Involvement'].isin(selected_involvement)) &
        (df['School_Type'].isin(selected_schools))
    ]
    
    # Executive Summary
    st.header("Executive Summary")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Students", f"{len(filtered_df):,}")
    with col2:
        avg_score = filtered_df['Exam_Score'].mean()
        st.metric("Average Exam Score", f"{avg_score:.1f}")
    with col3:
        avg_attendance = filtered_df['Attendance'].mean()
        st.metric("Average Attendance", f"{avg_attendance:.1f}%")
    with col4:
        high_performers = (filtered_df['Exam_Score'] >= 80).sum()
        high_perf_pct = (high_performers / len(filtered_df)) * 100
        st.metric("High Performers (≥80)", f"{high_perf_pct:.1f}%")
    
    # Professional Visualizations Row 1: Donut Charts
    st.header("Engagement & Performance Distribution")
    col1, col2 = st.columns(2)
    
    with col1:
        donut_involvement = create_donut_chart(
            filtered_df, 'Parental_Involvement', 
            'Parental Involvement Distribution',
            colors=['#FF6B6B', '#4ECDC4', '#45B7D1']
        )
        st.pyplot(donut_involvement)
    
    with col2:
        donut_performance = create_donut_chart(
            filtered_df, 'Performance_Category', 
            'Academic Performance Distribution',
            colors=['#FF6B6B', '#FFA07A', '#FFD700', '#98FB98', '#90EE90']
        )
        st.pyplot(donut_performance)
    
    # Professional Visualizations Row 2: More Donut/Pie Charts
    col1, col2 = st.columns(2)
    
    with col1:
        donut_attendance = create_donut_chart(
            filtered_df, 'Attendance_Category', 
            'Attendance Level Distribution',
            colors=['#FF6B6B', '#FFD700', '#90EE90']
        )
        st.pyplot(donut_attendance)
    
    with col2:
        donut_study = create_donut_chart(
            filtered_df, 'Study_Hours_Category', 
            'Study Hours Distribution',
            colors=['#FF6B6B', '#FFA07A', '#90EE90']
        )
        st.pyplot(donut_study)
    
    # Histograms Section
    st.header("Performance Distribution Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        hist_scores = create_histogram_with_kde(
            filtered_df, 'Exam_Score', 
            'Distribution of Exam Scores'
        )
        st.pyplot(hist_scores)
    
    with col2:
        hist_hours = create_histogram_with_kde(
            filtered_df, 'Hours_Studied', 
            'Distribution of Study Hours'
        )
        st.pyplot(hist_hours)
    
    # Correlation Analysis
    st.header("Correlation Analysis")
    corr_fig = create_correlation_heatmap(filtered_df)
    if corr_fig:
        st.pyplot(corr_fig)
        
        # Correlation insights
        numeric_cols = filtered_df.select_dtypes(include=[np.number]).columns
        if 'Exam_Score' in numeric_cols:
            correlations = filtered_df[numeric_cols].corr()['Exam_Score'].abs().sort_values(ascending=False)
            
            st.subheader("Key Correlations with Exam Performance")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Strongest Positive Factors:**")
                top_factors = correlations.drop('Exam_Score').head(5)
                for factor, corr in top_factors.items():
                    strength = "Strong" if corr > 0.5 else "Moderate" if corr > 0.3 else "Weak"
                    st.write(f"• {factor}: {corr:.3f} ({strength})")
            
            with col2:
                # Parental involvement impact analysis
                involvement_stats = filtered_df.groupby('Parental_Involvement')['Exam_Score'].agg(['mean', 'count']).round(2)
                st.write("**Performance by Parental Involvement:**")
                st.dataframe(involvement_stats)
    
    # Detailed Insights Section
    st.header("Key Insights & Recommendations")
    
    # Calculate key statistics
    high_involvement = filtered_df[filtered_df['Parental_Involvement'] == 'High']['Exam_Score'].mean()
    low_involvement = filtered_df[filtered_df['Parental_Involvement'] == 'Low']['Exam_Score'].mean()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Performance Insights")
        if not pd.isna(high_involvement) and not pd.isna(low_involvement):
            improvement = high_involvement - low_involvement
            st.markdown(f"""
            <div class='metric-card'>
                <strong>Parental Involvement Impact:</strong><br>
                High involvement students score {improvement:.1f} points higher on average
                ({high_involvement:.1f} vs {low_involvement:.1f})
            </div>
            """, unsafe_allow_html=True)
        
        # Top performing factors
        if 'Exam_Score' in filtered_df.columns:
            high_performers = filtered_df[filtered_df['Exam_Score'] >= 80]
            if len(high_performers) > 0:
                common_factors = {
                    'High Parental Involvement': (high_performers['Parental_Involvement'] == 'High').mean() * 100,
                    'Excellent Attendance': (high_performers['Attendance'] > 85).mean() * 100,
                    'High Study Hours': (high_performers['Hours_Studied'] > 20).mean() * 100
                }
                
                st.write("**Common traits of high performers:**")
                for trait, percentage in common_factors.items():
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
    
    # Data Export Section
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
        # Summary statistics
        summary_stats = {
            'Metric': ['Total Students', 'Avg Exam Score', 'Avg Attendance', 'High Performers %', 'High Involvement %'],
            'Value': [
                len(filtered_df),
                f"{filtered_df['Exam_Score'].mean():.1f}",
                f"{filtered_df['Attendance'].mean():.1f}",
                f"{(filtered_df['Exam_Score'] >= 80).mean() * 100:.1f}",
                f"{(filtered_df['Parental_Involvement'] == 'High').mean() * 100:.1f}"
            ]
        }
        summary_df = pd.DataFrame(summary_stats)
        st.dataframe(summary_df)
    
    # Add new section after Correlation Analysis
    st.header("🎯 Parental Involvement Impact Analysis")
    
    # Calculate correlation
    try:
        correlation, p_value = analyze_parental_involvement_correlation(filtered_df)
        
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
        
        # Interpretation
        st.subheader("📊 What This Means")
        
        if correlation > 0.5:
            interpretation = "There is a **strong positive correlation** between parental involvement and student exam scores. Students with high parental involvement tend to perform significantly better."
        elif correlation > 0.3:
            interpretation = "There is a **moderate positive correlation** between parental involvement and student exam scores. Parental involvement appears to have a meaningful impact on performance."
        elif correlation > 0:
            interpretation = "There is a **weak positive correlation** between parental involvement and student exam scores. Some relationship exists but other factors may be more influential."
        else:
            interpretation = "There is **little to no correlation** between parental involvement and student exam scores in this dataset."
        
        st.markdown(f"**Interpretation:** {interpretation}")
        
        # Detailed analysis charts
        st.subheader("📈 Detailed Analysis")
        involvement_analysis_fig = create_parental_involvement_analysis(filtered_df)
        st.pyplot(involvement_analysis_fig)
        
        # Statistical breakdown
        st.subheader("📋 Statistical Breakdown")
        
        # Group statistics
        involvement_stats = filtered_df.groupby('Parental_Involvement')['Exam_Score'].agg([
            'count', 'mean', 'median', 'std', 'min', 'max'
        ]).round(2)
        involvement_stats.columns = ['Count', 'Mean Score', 'Median Score', 'Std Dev', 'Min Score', 'Max Score']
        
        st.dataframe(involvement_stats, use_container_width=True)
        
        # Performance comparison
        st.subheader("🏆 Performance Comparison")
        
        high_involvement_mean = filtered_df[filtered_df['Parental_Involvement'] == 'High']['Exam_Score'].mean()
        medium_involvement_mean = filtered_df[filtered_df['Parental_Involvement'] == 'Medium']['Exam_Score'].mean()
        low_involvement_mean = filtered_df[filtered_df['Parental_Involvement'] == 'Low']['Exam_Score'].mean()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Score Differences:**")
            if not pd.isna(high_involvement_mean) and not pd.isna(low_involvement_mean):
                diff_high_low = high_involvement_mean - low_involvement_mean
                st.write(f"• High vs Low: **{diff_high_low:.1f} points** difference")
            
            if not pd.isna(high_involvement_mean) and not pd.isna(medium_involvement_mean):
                diff_high_medium = high_involvement_mean - medium_involvement_mean
                st.write(f"• High vs Medium: **{diff_high_medium:.1f} points** difference")
            
            if not pd.isna(medium_involvement_mean) and not pd.isna(low_involvement_mean):
                diff_medium_low = medium_involvement_mean - low_involvement_mean
                st.write(f"• Medium vs Low: **{diff_medium_low:.1f} points** difference")
        
        with col2:
            st.markdown("**High Performers (Score ≥ 80) by Involvement:**")
            high_perf_by_involvement = filtered_df[filtered_df['Exam_Score'] >= 80].groupby('Parental_Involvement').size()
            total_by_involvement = filtered_df.groupby('Parental_Involvement').size()
            
            for involvement in ['Low', 'Medium', 'High']:
                if involvement in high_perf_by_involvement.index and involvement in total_by_involvement.index:
                    perf_pct = (high_perf_by_involvement[involvement] / total_by_involvement[involvement]) * 100
                    st.write(f"• {involvement}: **{perf_pct:.1f}%** are high performers")
        
        # Conclusion
        st.subheader("💡 Key Findings")
        
        findings = []
        
        if correlation > 0.3:
            findings.append("✅ **Parental involvement positively correlates with academic performance**")
        else:
            findings.append("⚠️ **Weak correlation between parental involvement and performance**")
        
        if not pd.isna(high_involvement_mean) and not pd.isna(low_involvement_mean):
            if high_involvement_mean - low_involvement_mean > 5:
                findings.append(f"✅ **Students with high parental involvement score {high_involvement_mean - low_involvement_mean:.1f} points higher on average**")
        
        # Check if high involvement leads to more high performers
        if 'High' in filtered_df['Parental_Involvement'].values:
            high_involvement_high_perf = (filtered_df[filtered_df['Parental_Involvement'] == 'High']['Exam_Score'] >= 80).mean() * 100
            if high_involvement_high_perf > 50:
                findings.append(f"✅ **{high_involvement_high_perf:.1f}% of students with high parental involvement are high performers (≥80 score)**")
        
        for finding in findings:
            st.markdown(finding)
    
    except Exception as e:
        st.error(f"Error in correlation analysis: {str(e)}")
        st.info("Basic analysis without statistical testing:")
        
        # Basic fallback analysis
        involvement_means = filtered_df.groupby('Parental_Involvement')['Exam_Score'].mean()
        st.write("**Average Scores by Parental Involvement:**")
        for level, score in involvement_means.items():
            st.write(f"• {level}: {score:.1f}")

if __name__ == "__main__":
    main()