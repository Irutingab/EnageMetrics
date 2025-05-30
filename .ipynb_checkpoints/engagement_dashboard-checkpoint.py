import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("parent_engagement_mock_data.csv")

st.title("Parental Engagement Dashboard")

st.markdown(
    """
    This dashboard was developed for school administrators and parents to understand how parental engagement affects student performance over time. While visualizations offer quick insights, I also considered accessibility by including a simplified text-based view for users with visual or physical impairments. The goal is to support all families — regardless of ability — in following their child’s learning journey, identifying struggles early, and actively participating in their success.
    """
)

# Sidebar filters
students = st.sidebar.multiselect(
    "Select Students",
    options=df["Student_Name"].unique(),
    default=df["Student_Name"].unique()
)
filtered_df = df[df["Student_Name"].isin(students)]

# Overview metrics
st.header("Overview")
st.write(f"Total students: **{filtered_df.shape[0]}**")
st.write(f"Average Grade: **{filtered_df['Avg_Grade'].mean():.2f}**")
st.write(f"Average Attendance: **{filtered_df['Avg_Attendance'].mean():.2f}%**")

# Create two columns side-by-side
col1, col2 = st.columns(2)

# Column 1: Scatterplot
with col1:
    st.subheader("Engagement vs Student Performance")
    fig1, ax1 = plt.subplots()
    sns.scatterplot(
        data=filtered_df,
        x="Total_Logins",
        y="Avg_Grade",
        hue="Total_Messages",
        size="Avg_Attendance",
        palette="viridis",
        ax=ax1
    )
    ax1.set_xlabel("Total Parent Logins")
    ax1.set_ylabel("Average Student Grade")
    ax1.set_title("Logins vs Grades")
    st.pyplot(fig1)

# Column 2: Correlation Heatmap
with col2:
    st.subheader("Correlation Heatmap")
    numeric_cols = filtered_df.select_dtypes(include='number')
    corr = numeric_cols.corr()
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax2)
    ax2.set_title("Correlations Between Metrics")
    st.pyplot(fig2)

# Full width: Grade Distribution
st.subheader("Grade Distribution")
fig3, ax3 = plt.subplots()
sns.histplot(filtered_df["Avg_Grade"], kde=True, ax=ax3)
ax3.set_title("Distribution of Average Grades")
st.pyplot(fig3)
