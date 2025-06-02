# Student Performance & Parental Engagement Dashboard

A comprehensive Streamlit dashboard that analyzes the relationship between parental involvement and student academic performance using professional visualizations and statistical analysis.

## Project Overview

This dashboard provides insights into how parental engagement levels correlate with student outcomes including exam scores, attendance rates, and overall academic performance. The project uses matplotlib-based visualizations to ensure compatibility and professional presentation.

## Dataset Information

### Primary Dataset: student_performance_cleaned.csv
- **Student Records**: Comprehensive dataset with student performance metrics
- **Parental Involvement**: Three levels (Low, Medium, High) of parental engagement
- **Academic Metrics**: Exam scores, attendance rates, study hours, and performance categories
- **Demographics**: School type, family income, parental education, and other factors

## Dashboard Features

### Professional Visualizations
1. **Donut Charts**: Parental involvement distribution and attendance patterns using matplotlib
2. **Histograms**: Grade distributions, study hours, and performance categories
3. **Box Plots**: Score distributions by demographics and education levels
4. **Scatter Plots**: Attendance vs exam scores with parental involvement overlay
5. **Correlation Heatmaps**: Statistical relationships between all numeric variables
6. **Bar Charts**: Average performance by parental involvement levels

### Statistical Analysis
- **Correlation Analysis**: Pearson correlation coefficients with significance testing
- **Performance Insights**: Automated analysis of high-performer characteristics
- **Demographic Breakdowns**: Performance segmentation by various factors
- **Trend Analysis**: Parental involvement impact quantification

## Technical Implementation

### Architecture
The dashboard is built with a modular architecture:

```
dashboard.py          # Main dashboard orchestration
├── data_manager.py   # Data loading and preprocessing
├── visualizations.py # All chart creation methods
└── analytics.py      # Statistical analysis functions
```

### Key Components

#### DataManager Class
- Loads and caches student performance data
- Creates performance categories (A, B, C, D, F grades)
- Generates attendance and study hours categories
- Applies dynamic filtering based on user selections

#### Visualizations Class
- **create_histogram_chart()**: Handles both categorical and numerical data
- **create_donut_chart()**: Professional pie charts with donut styling
- **create_correlation_heatmap()**: Statistical relationship visualization
- **create_parental_involvement_analysis()**: Comprehensive 4-panel analysis
- All visualizations use matplotlib for consistency and compatibility

#### Analytics Class
- Correlation analysis with statistical significance testing
- Performance insights calculation
- Parental involvement impact quantification
- High performer trait identification

### Dashboard Sections

1. **Executive Summary**: Key metrics overview with student counts and averages
2. **Distribution Charts**: Engagement and performance patterns using donut charts and histograms
3. **Performance Analysis**: Grade distributions and advanced correlation analysis
4. **Correlation Matrix**: Statistical relationships with significance indicators
5. **Insights & Recommendations**: Data-driven actionable recommendations
6. **Detailed Analysis**: Comprehensive parental involvement impact assessment
7. **Export Capabilities**: CSV download and summary statistics

## Usage Instructions

1. **Install Dependencies**:
   ```bash
   pip install streamlit pandas matplotlib numpy
   ```

2. **Prepare Data**:
   - Ensure `student_performance_cleaned.csv` is in the project directory
   - Verify data includes required columns (Exam_Score, Parental_Involvement, Attendance, etc.)

3. **Run Dashboard**:
   ```bash
   streamlit run dashboard.py
   ```

4. **Navigate Features**:
   - Use sidebar filters to segment data by parental involvement and school type
   - Review executive summary for key insights
   - Explore distribution charts to understand patterns
   - Analyze correlation matrix for statistical relationships
   - Export filtered data and summary statistics

## Key Findings Framework

The dashboard automatically calculates and presents:
- **Involvement Impact**: Score difference between high and low parental involvement
- **High Performer Traits**: Common characteristics of top-performing students
- **Correlation Strength**: Statistical significance of relationships
- **Demographic Patterns**: Performance variations across different groups

## Data Quality Features

- **Categorical Data Handling**: Proper processing of both numerical and categorical variables
- **Missing Data Management**: Robust handling of incomplete records
- **Statistical Validation**: Significance testing for correlation analysis
- **Dynamic Categorization**: Automatic binning of continuous variables

## Visualization Specifications

### Histogram Implementation
The dashboard correctly implements histograms for:
- **Categorical Data**: Bar charts with value labels for performance categories
- **Numerical Data**: Traditional histograms with mean indicators
- **Professional Styling**: Consistent color schemes and formatting

### Chart Features
- **Interactive Elements**: Hover details and dynamic filtering
- **Export Quality**: High-resolution matplotlib figures
- **Color Consistency**: Professional color palette throughout
- **Responsive Design**: Adaptable to different screen sizes

## Professional Applications

This dashboard is designed for:
- **School Administrators**: Understanding engagement-performance relationships
- **Educational Researchers**: Statistical analysis of parental involvement
- **Policy Makers**: Evidence-based decision making for family engagement programs
- **Parent Coordinators**: Identifying effective engagement strategies

## Future Enhancements

- **Predictive Modeling**: Machine learning for performance prediction
- **Time Series Analysis**: Longitudinal engagement tracking
- **Comparative Benchmarking**: Multi-school performance comparison
- **Advanced Segmentation**: More granular demographic analysis
- **Real-time Updates**: API integration for live data feeds