import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

class Visualizations:
    """Handles all visualization creation methods"""
    
    @staticmethod
    def create_donut_chart(df, column, title, colors=None):
        """Create a professional pie chart using matplotlib"""
        value_counts = df[column].value_counts()
        
        if colors is None:
            colors = plt.cm.Set3(np.linspace(0, 1, len(value_counts)))
        
        fig, ax = plt.subplots(figsize=(8, 8))
        
        wedges, texts, autotexts = ax.pie(
            value_counts.values,
            labels=value_counts.index,
            colors=colors,
            autopct='%1.1f%%',
            startangle=90,
            wedgeprops=dict(width=0.6)
        )
        
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        return fig
    
    @staticmethod
    def create_histogram_chart(df, column, title, colors=None, bins=None):
        """Create a histogram for categorical or numerical data"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if df[column].dtype == 'object' or pd.api.types.is_categorical_dtype(df[column]):
            # For categorical data, create a bar chart
            value_counts = df[column].value_counts()
            bars = ax.bar(range(len(value_counts)), value_counts.values, 
                         color=colors[:len(value_counts)] if colors else None)
            ax.set_xticks(range(len(value_counts)))
            ax.set_xticklabels(value_counts.index, rotation=45, ha='right')
            ax.set_ylabel('Count')
            
            # Add value labels on bars
            for bar, value in zip(bars, value_counts.values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                       str(value), ha='center', va='bottom', fontweight='bold')
        else:
            # For numerical data, create a histogram
            n, bins, patches = ax.hist(df[column].dropna(), bins=bins or 20, 
                                     color=colors[0] if colors else 'skyblue', 
                                     alpha=0.7, edgecolor='black')
            ax.set_xlabel(column)
            ax.set_ylabel('Frequency')
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig
    
    @staticmethod
    def create_histogram_with_kde(df, column, title, bins=20):
        """Create histogram without KDE overlay"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        n, bins, patches = ax.hist(df[column].dropna(), bins=bins, alpha=0.7, 
                                color='skyblue', edgecolor='black')
        
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
    
    @staticmethod
    def create_correlation_heatmap(df):
        """Create correlation heatmap for numeric columns using matplotlib only"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) < 2:
            return None
        
        corr_matrix = df[numeric_cols].corr()
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        im = ax.imshow(corr_matrix, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
        
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Correlation Coefficient')
        
        ax.set_xticks(range(len(corr_matrix.columns)))
        ax.set_yticks(range(len(corr_matrix.columns)))
        ax.set_xticklabels(corr_matrix.columns, rotation=45, ha='right')
        ax.set_yticklabels(corr_matrix.columns)
        
        for i in range(len(corr_matrix.columns)):
            for j in range(len(corr_matrix.columns)):
                text = ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                            ha="center", va="center", color="black", fontsize=8)
        
        ax.set_title('Correlation Matrix: Student Performance Factors', fontsize=16, pad=20)
        plt.tight_layout()
        return fig
    
    @staticmethod
    def create_bar_chart_scores_by_involvement(df):
        """Create bar chart showing average exam scores by parental involvement"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        involvement_order = ['Low', 'Medium', 'High']
        mean_scores = df.groupby('Parental_Involvement')['Exam_Score'].mean()
        mean_scores = mean_scores.reindex(involvement_order)
        
        bars = ax.bar(involvement_order, mean_scores, 
                    color=['#FF6B6B', '#FFD700', '#90EE90'],
                    edgecolor='black', linewidth=1.2)
        
        for bar, value in zip(bars, mean_scores):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                    f'{value:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=12)
        
        ax.set_title('Average Exam Scores by Parental Involvement Level', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Parental Involvement Level', fontsize=12)
        ax.set_ylabel('Average Exam Score', fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, max(mean_scores) * 1.1)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def create_scatter_attendance_vs_scores(df):
        """Create scatter plot of Attendance vs Exam_Score colored by Parental_Involvement"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        colors = {'Low': '#FF6B6B', 'Medium': '#FFD700', 'High': '#90EE90'}
        
        for involvement in ['Low', 'Medium', 'High']:
            subset = df[df['Parental_Involvement'] == involvement]
            ax.scatter(subset['Attendance'], subset['Exam_Score'], 
                    c=colors[involvement], label=f'{involvement} Involvement',
                    alpha=0.7, s=50, edgecolors='black', linewidth=0.5)
        
        z = np.polyfit(df['Attendance'], df['Exam_Score'], 1)
        p = np.poly1d(z)
        ax.plot(df['Attendance'], p(df['Attendance']), "r--", alpha=0.8, linewidth=2, label='Trend Line')
        
        ax.set_title('Attendance vs Exam Score (by Parental Involvement)', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Attendance (%)', fontsize=12)
        ax.set_ylabel('Exam Score', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def create_box_plot_scores_by_education(df):
        """Create box plot showing exam score distribution by parental education level"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        education_order = ['High School', 'College', 'Postgraduate']
        education_data = [df[df['Parental_Education_Level'] == level]['Exam_Score'].values 
                        for level in education_order if level in df['Parental_Education_Level'].values]
        education_labels = [level for level in education_order if level in df['Parental_Education_Level'].values]
        
        if education_data:
            box1 = ax1.boxplot(education_data, labels=education_labels, patch_artist=True)
            colors1 = ['#FF6B6B', '#FFD700', '#90EE90']
            for patch, color in zip(box1['boxes'], colors1[:len(box1['boxes'])]):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            ax1.set_title('Exam Score Distribution by Parental Education Level', fontweight='bold')
            ax1.set_xlabel('Parental Education Level')
            ax1.set_ylabel('Exam Score')
            ax1.grid(True, alpha=0.3)
        
        income_order = ['Low', 'Medium', 'High']
        income_data = [df[df['Family_Income'] == level]['Exam_Score'].values 
                    for level in income_order if level in df['Family_Income'].values]
        income_labels = [level for level in income_order if level in df['Family_Income'].values]
        
        if income_data:
            box2 = ax2.boxplot(income_data, labels=income_labels, patch_artist=True)
            colors2 = ['#FF6B6B', '#FFD700', '#90EE90']
            for patch, color in zip(box2['boxes'], colors2[:len(box2['boxes'])]):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            ax2.set_title('Exam Score Distribution by Family Income', fontweight='bold')
            ax2.set_xlabel('Family Income Level')
            ax2.set_ylabel('Exam Score')
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def create_parental_involvement_analysis(df):
        """Create detailed analysis of parental involvement impact"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        involvement_order = ['Low', 'Medium', 'High']
        df_ordered = df[df['Parental_Involvement'].isin(involvement_order)]
        
        box_data = [df_ordered[df_ordered['Parental_Involvement'] == level]['Exam_Score'].values 
                    for level in involvement_order]
        
        ax1.boxplot(box_data, labels=involvement_order)
        ax1.set_title('Exam Scores by Parental Involvement Level', fontweight='bold')
        ax1.set_xlabel('Parental Involvement Level')
        ax1.set_ylabel('Exam Score')
        ax1.grid(True, alpha=0.3)
        
        mean_scores = df.groupby('Parental_Involvement')['Exam_Score'].mean().reindex(involvement_order)
        bars = ax2.bar(involvement_order, mean_scores, color=['#FF6B6B', '#FFD700', '#90EE90'])
        ax2.set_title('Average Exam Scores by Parental Involvement', fontweight='bold')
        ax2.set_xlabel('Parental Involvement Level')
        ax2.set_ylabel('Average Exam Score')
        
        for bar, value in zip(bars, mean_scores):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                    f'{value:.1f}', ha='center', va='bottom', fontweight='bold')
        
        performance_by_involvement = pd.crosstab(df['Parental_Involvement'], df['Performance_Category'], normalize='index') * 100
        performance_by_involvement.plot(kind='bar', ax=ax3, stacked=True, 
                                       color=['#FF6B6B', '#FFA07A', '#FFD700', '#98FB98', '#90EE90'])
        ax3.set_title('Performance Distribution by Parental Involvement (%)', fontweight='bold')
        ax3.set_xlabel('Parental Involvement Level')
        ax3.set_ylabel('Percentage')
        ax3.legend(title='Performance Grade', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax3.tick_params(axis='x', rotation=45)
        
        involvement_numeric = df['Parental_Involvement'].map({'Low': 1, 'Medium': 2, 'High': 3})
        ax4.scatter(involvement_numeric, df['Exam_Score'], alpha=0.6, color='#45B7D1')
        
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
