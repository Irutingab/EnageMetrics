# Parental Engagement Dashboard

A comprehensive Streamlit dashboard designed to help school administrators and parents understand how parental engagement affects student performance over time.

## Project Overview

This dashboard provides insights into the relationship between parental engagement metrics (logins, messages, meeting attendance) and student outcomes (grades, attendance). The project emphasizes accessibility and data quality, ensuring all stakeholders can effectively use the insights.

## Dataset Information

### Primary Dataset: student_performance_cleaned.csv
- **Pre-cleaned Data**: High-quality dataset with minimal missing values
- **Comprehensive Metrics**: Parental involvement indicators and student performance data
- **Professional Analysis**: Advanced visualizations including donuts, pie charts, and histograms

### Legacy Dataset: parent_engagement_data.csv
- **Raw Data Processing**: Demonstrates data cleaning workflows
- **Quality Assessment**: Shows data preparation steps

## Dashboard Features

### Professional Visualizations
1. **Donut Charts**: Parental involvement distribution and engagement levels
2. **Pie Charts**: Performance categories and attendance patterns
3. **Histograms**: Grade distributions and engagement frequency
4. **Correlation Heatmaps**: Statistical relationships between variables
5. **Scatter Plots**: Multi-dimensional engagement vs performance analysis

### Key Analytics
- **Engagement Scoring**: Quantified parental involvement metrics
- **Performance Segmentation**: Student achievement categorization
- **Correlation Analysis**: Statistical significance testing
- **Trend Identification**: Pattern recognition in engagement-performance relationships

## Project Completion Steps

### 1. Enhanced Data Analysis
- **Dataset**: student_performance_cleaned.csv with 1,000,000+ records
- **Quality**: Pre-processed with minimal missing values
- **Scope**: Comprehensive parental involvement and student outcome metrics

### 2. Professional Dashboard Design
The new dashboard includes:
- **Executive Summary**: Key insights at a glance
- **Engagement Distribution**: Donut charts showing involvement patterns
- **Performance Categories**: Pie charts for achievement levels
- **Statistical Analysis**: Correlation matrices and significance testing
- **Interactive Filtering**: Dynamic data exploration

### 3. Advanced Visualizations

#### Donut Charts
- **Parental Involvement Levels**: High, Medium, Low engagement categories
- **Communication Patterns**: Message frequency distributions
- **Meeting Attendance**: Participation rate breakdowns

#### Pie Charts
- **Academic Performance**: Grade categories (A, B, C, D, F)
- **Attendance Levels**: Regular, Occasional, Poor attendance
- **Engagement Types**: Login frequency, message activity, meeting participation

#### Histograms
- **Grade Distributions**: Normal distribution analysis
- **Login Frequency**: Engagement pattern identification
- **Attendance Rates**: Performance correlation insights

### 4. Correlation Analysis Framework
```python
# Advanced correlation analysis
correlation_matrix = df_clean.corr()
significant_correlations = identify_significant_relationships()
engagement_performance_score = calculate_composite_metrics()
```

### 5. Professional UI/UX Design

#### Clean Interface
- **Minimalist Design**: Focus on data insights
- **Color Coding**: Consistent theme throughout
- **Professional Typography**: Clear, readable fonts
- **Logical Layout**: Intuitive information hierarchy

#### Interactive Elements
- **Dynamic Filtering**: Real-time chart updates
- **Hover Details**: Rich tooltips for data points
- **Export Options**: Professional report generation
- **Responsive Design**: Works across devices

## Technical Implementation

### Enhanced Dependencies
```python
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr
```

### Data Processing Pipeline
1. **Load**: Efficient loading of cleaned dataset
2. **Validate**: Quality checks and data integrity
3. **Transform**: Create composite engagement scores
4. **Analyze**: Statistical correlation testing
5. **Visualize**: Professional chart generation
6. **Report**: Insight summarization

## Usage Instructions

1. **Install Enhanced Dependencies**:
   ```bash
   pip install streamlit pandas plotly seaborn matplotlib numpy scipy
   ```

2. **Prepare Data**:
   - Ensure `student_performance_cleaned.csv` is in the project directory
   - Verify columns include engagement metrics and performance indicators

3. **Run Enhanced Dashboard**:
   ```bash
   streamlit run student_performance_dashboard.py
   ```

4. **Navigate Dashboard**:
   - Review executive summary
   - Explore donut and pie chart insights
   - Analyze correlation patterns
   - Export findings for reporting

## Data Quality Commitment

This enhanced dashboard provides:
- **Pre-validated Data**: Using cleaned, high-quality dataset
- **Statistical Rigor**: Correlation significance testing
- **Professional Presentation**: Executive-ready visualizations
- **Actionable Insights**: Clear recommendations based on data

The goal is to provide school administrators with professional-grade analytics to make informed decisions about parental engagement programs and student support strategies.

## Future Enhancements

- **Predictive Modeling**: Machine learning for engagement recommendations
- **Time Series Analysis**: Longitudinal engagement tracking
- **Comparative Analysis**: School-to-school benchmarking
- **Mobile Dashboard**: Responsive design for mobile access
- **API Integration**: Real-time data updates from school systems