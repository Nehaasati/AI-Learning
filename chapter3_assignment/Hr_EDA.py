
# HR EMPLOYEE DATA - EXPLORATORY DATA ANALYSIS
# Management-Focused EDA | No Machine Learning


import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")


# 1. SETTINGS


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FILE_PATH = os.path.join(BASE_DIR, "hr_employee_data.xlsx")
OUTPUT_DIR = os.path.join(BASE_DIR, "eda_outputs")

os.makedirs(OUTPUT_DIR, exist_ok=True)

sns.set_theme(style="whitegrid")
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 140)


# 2. LOAD DATA


print("\n" + "=" * 70)
print("HR EMPLOYEE DATA - EXPLORATORY DATA ANALYSIS")
print("=" * 70)

df = pd.read_excel(FILE_PATH)

print("\nDataset loaded successfully.")
print(f"Rows: {df.shape[0]:,}")
print(f"Columns: {df.shape[1]}")

# 3. BASIC DATA QUALITY CHECK


print("\n" + "=" * 70)
print("1. DATA QUALITY CHECK")
print("=" * 70)

print("\nColumn names:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)

missing_summary = pd.DataFrame({
    "Missing_Count": missing,
    "Missing_Percentage": missing_pct
})

print(missing_summary)

print("\nDuplicate rows:")
print(df.duplicated().sum())

print("\nUnique employees:")
print(df["Emp_Id"].nunique())

# Check duplicate employee IDs
duplicate_ids = df["Emp_Id"].duplicated().sum()
print(f"Duplicate employee IDs: {duplicate_ids}")


# 4. STANDARDIZE COLUMN NAMES FOR ANALYSIS


# Keep original columns but create convenient aliases
df["Attrition"] = df["left"].map({0: "Stayed", 1: "Left"})
df["Work_Accident_Status"] = df["Work_accident"].map(
    {0: "No Accident", 1: "Accident"}
)
df["Promotion_Status"] = df["promotion_last_5years"].map(
    {0: "No Promotion", 1: "Promoted"}
)


# 5. EXECUTIVE KPI SUMMARY


print("\n" + "=" * 70)
print("2. EXECUTIVE MANAGEMENT KPIs")
print("=" * 70)

total_employees = len(df)

employees_left = df["left"].sum()
employees_stayed = (df["left"] == 0).sum()

attrition_rate = df["left"].mean() * 100

avg_satisfaction = df["satisfaction_level"].mean()
avg_evaluation = df["last_evaluation"].mean()
avg_hours = df["average_montly_hours"].mean()
avg_projects = df["number_project"].mean()
avg_tenure = df["time_spend_company"].mean()

promotion_rate = df["promotion_last_5years"].mean() * 100
accident_rate = df["Work_accident"].mean() * 100

print(f"\nTotal Employees:              {total_employees:,}")
print(f"Employees Who Left:           {employees_left:,}")
print(f"Employees Who Stayed:         {employees_stayed:,}")
print(f"Attrition Rate:               {attrition_rate:.2f}%")
print(f"Average Satisfaction:         {avg_satisfaction:.2f}")
print(f"Average Evaluation:           {avg_evaluation:.2f}")
print(f"Average Monthly Hours:        {avg_hours:.1f}")
print(f"Average Projects:             {avg_projects:.2f}")
print(f"Average Years at Company:     {avg_tenure:.2f}")
print(f"Promotion Rate (5 years):     {promotion_rate:.2f}%")
print(f"Work Accident Rate:           {accident_rate:.2f}%")

# 6. DESCRIPTIVE STATISTICS


print("\n" + "=" * 70)
print("3. DESCRIPTIVE STATISTICS")
print("=" * 70)

numeric_columns = [
    "satisfaction_level",
    "last_evaluation",
    "number_project",
    "average_montly_hours",
    "time_spend_company",
    "Work_accident",
    "left",
    "promotion_last_5years"
]

print(
    df[numeric_columns]
    .describe()
    .round(2)
    .to_string()
)


# 7. DEPARTMENT ANALYSIS


print("\n" + "=" * 70)
print("4. DEPARTMENT ANALYSIS")
print("=" * 70)

department_summary = (
    df.groupby("Department")
    .agg(
        Employees=("Emp_Id", "count"),
        Attrition_Rate=("left", "mean"),
        Avg_Satisfaction=("satisfaction_level", "mean"),
        Avg_Evaluation=("last_evaluation", "mean"),
        Avg_Monthly_Hours=("average_montly_hours", "mean"),
        Avg_Projects=("number_project", "mean"),
        Avg_Tenure=("time_spend_company", "mean"),
        Promotion_Rate=("promotion_last_5years", "mean")
    )
    .reset_index()
)

department_summary["Attrition_Rate"] *= 100
department_summary["Promotion_Rate"] *= 100

department_summary = department_summary.round(2)

print("\nDepartment summary:")
print(department_summary.to_string(index=False))

department_summary.to_csv(
    f"{OUTPUT_DIR}/department_summary.csv",
    index=False
)

# 8. SALARY ANALYSIS


print("\n" + "=" * 70)
print("5. SALARY ANALYSIS")
print("=" * 70)

salary_summary = (
    df.groupby("salary")
    .agg(
        Employees=("Emp_Id", "count"),
        Attrition_Rate=("left", "mean"),
        Avg_Satisfaction=("satisfaction_level", "mean"),
        Avg_Evaluation=("last_evaluation", "mean"),
        Avg_Monthly_Hours=("average_montly_hours", "mean"),
        Avg_Projects=("number_project", "mean"),
        Avg_Tenure=("time_spend_company", "mean"),
        Promotion_Rate=("promotion_last_5years", "mean")
    )
    .reset_index()
)

salary_summary["Attrition_Rate"] *= 100
salary_summary["Promotion_Rate"] *= 100

salary_summary = salary_summary.round(2)

print("\nSalary summary:")
print(salary_summary.to_string(index=False))

salary_summary.to_csv(
    f"{OUTPUT_DIR}/salary_summary.csv",
    index=False
)


# 9. ATTRITION ANALYSIS


print("\n" + "=" * 70)
print("6. ATTRITION ANALYSIS")
print("=" * 70)

attrition_summary = (
    df.groupby("Attrition")
    .agg(
        Employees=("Emp_Id", "count"),
        Avg_Satisfaction=("satisfaction_level", "mean"),
        Avg_Evaluation=("last_evaluation", "mean"),
        Avg_Projects=("number_project", "mean"),
        Avg_Monthly_Hours=("average_montly_hours", "mean"),
        Avg_Tenure=("time_spend_company", "mean"),
        Promotion_Rate=("promotion_last_5years", "mean"),
        Accident_Rate=("Work_accident", "mean")
    )
    .reset_index()
)

attrition_summary["Promotion_Rate"] *= 100
attrition_summary["Accident_Rate"] *= 100

print("\nEmployees who stayed vs left:")
print(attrition_summary.round(2).to_string(index=False))

attrition_summary.to_csv(
    f"{OUTPUT_DIR}/attrition_summary.csv",
    index=False
)

# ------------------------------------------------------------
# 10. ATTRITION BY DEPARTMENT
# ------------------------------------------------------------

attrition_department = pd.crosstab(
    df["Department"],
    df["Attrition"],
    normalize="index"
) * 100

print("\n" + "=" * 70)
print("7. ATTRITION BY DEPARTMENT")
print("=" * 70)

print(attrition_department.round(2).to_string())


# 11. ATTRITION BY SALARY


attrition_salary = pd.crosstab(
    df["salary"],
    df["Attrition"],
    normalize="index"
) * 100

print("\n" + "=" * 70)
print("8. ATTRITION BY SALARY")
print("=" * 70)

print(attrition_salary.round(2).to_string())

# 12. SATISFACTION ANALYSIS


print("\n" + "=" * 70)
print("9. SATISFACTION ANALYSIS")
print("=" * 70)

df["Satisfaction_Group"] = pd.cut(
    df["satisfaction_level"],
    bins=[0, 0.25, 0.50, 0.75, 1.00],
    labels=[
        "Very Low",
        "Low",
        "Moderate",
        "High"
    ],
    include_lowest=True
)

satisfaction_summary = (
    df.groupby("Satisfaction_Group", observed=True)
    .agg(
        Employees=("Emp_Id", "count"),
        Attrition_Rate=("left", "mean"),
        Avg_Hours=("average_montly_hours", "mean"),
        Avg_Projects=("number_project", "mean")
    )
    .reset_index()
)

satisfaction_summary["Attrition_Rate"] *= 100

print(satisfaction_summary.round(2).to_string(index=False))


# 13. WORKLOAD ANALYSIS


print("\n" + "=" * 70)
print("10. WORKLOAD ANALYSIS")
print("=" * 70)

df["Workload_Group"] = pd.cut(
    df["average_montly_hours"],
    bins=[0, 160, 200, 240, np.inf],
    labels=[
        "Low (<160h)",
        "Normal (160-200h)",
        "High (200-240h)",
        "Very High (>240h)"
    ]
)

workload_summary = (
    df.groupby("Workload_Group", observed=True)
    .agg(
        Employees=("Emp_Id", "count"),
        Attrition_Rate=("left", "mean"),
        Avg_Satisfaction=("satisfaction_level", "mean"),
        Avg_Projects=("number_project", "mean")
    )
    .reset_index()
)

workload_summary["Attrition_Rate"] *= 100

print(workload_summary.round(2).to_string(index=False))


# 14. TENURE ANALYSIS


print("\n" + "=" * 70)
print("11. TENURE ANALYSIS")
print("=" * 70)

tenure_summary = (
    df.groupby("time_spend_company")
    .agg(
        Employees=("Emp_Id", "count"),
        Attrition_Rate=("left", "mean"),
        Avg_Satisfaction=("satisfaction_level", "mean"),
        Avg_Hours=("average_montly_hours", "mean"),
        Avg_Projects=("number_project", "mean")
    )
    .reset_index()
)

tenure_summary["Attrition_Rate"] *= 100

print(tenure_summary.round(2).to_string(index=False))


# 15. PROJECT LOAD ANALYSIS


print("\n" + "=" * 70)
print("12. PROJECT LOAD ANALYSIS")
print("=" * 70)

project_summary = (
    df.groupby("number_project")
    .agg(
        Employees=("Emp_Id", "count"),
        Attrition_Rate=("left", "mean"),
        Avg_Satisfaction=("satisfaction_level", "mean"),
        Avg_Hours=("average_montly_hours", "mean")
    )
    .reset_index()
)

project_summary["Attrition_Rate"] *= 100

print(project_summary.round(2).to_string(index=False))

# 16. PROMOTION ANALYSIS


print("\n" + "=" * 70)
print("13. PROMOTION ANALYSIS")
print("=" * 70)

promotion_summary = (
    df.groupby("Promotion_Status")
    .agg(
        Employees=("Emp_Id", "count"),
        Attrition_Rate=("left", "mean"),
        Avg_Satisfaction=("satisfaction_level", "mean"),
        Avg_Evaluation=("last_evaluation", "mean"),
        Avg_Hours=("average_montly_hours", "mean"),
        Avg_Tenure=("time_spend_company", "mean")
    )
    .reset_index()
)

promotion_summary["Attrition_Rate"] *= 100

print(promotion_summary.round(2).to_string(index=False))


# 17. CORRELATION ANALYSIS


print("\n" + "=" * 70)
print("14. CORRELATION ANALYSIS")
print("=" * 70)

correlation_columns = [
    "satisfaction_level",
    "last_evaluation",
    "number_project",
    "average_montly_hours",
    "time_spend_company",
    "Work_accident",
    "left",
    "promotion_last_5years"
]

correlation_matrix = df[correlation_columns].corr()

print(correlation_matrix.round(2).to_string())

correlation_matrix.to_csv(
    f"{OUTPUT_DIR}/correlation_matrix.csv"
)

# 18. MANAGEMENT INSIGHTS


print("\n" + "=" * 70)
print("15. MANAGEMENT-ORIENTED INSIGHTS")
print("=" * 70)

# Highest attrition department
dept_attrition = (
    df.groupby("Department")["left"]
    .mean()
    .sort_values(ascending=False)
)

highest_attrition_dept = dept_attrition.index[0]
highest_attrition_dept_rate = dept_attrition.iloc[0] * 100

# Highest attrition salary
salary_attrition = (
    df.groupby("salary")["left"]
    .mean()
    .sort_values(ascending=False)
)

highest_attrition_salary = salary_attrition.index[0]
highest_attrition_salary_rate = salary_attrition.iloc[0] * 100

# Satisfaction comparison
stayed_satisfaction = df.loc[
    df["left"] == 0, "satisfaction_level"
].mean()

left_satisfaction = df.loc[
    df["left"] == 1, "satisfaction_level"
].mean()

# Hours comparison
stayed_hours = df.loc[
    df["left"] == 0, "average_montly_hours"
].mean()

left_hours = df.loc[
    df["left"] == 1, "average_montly_hours"
].mean()

# Project comparison
stayed_projects = df.loc[
    df["left"] == 0, "number_project"
].mean()

left_projects = df.loc[
    df["left"] == 1, "number_project"
].mean()

print(f"""
1. Overall attrition rate:
   {attrition_rate:.2f}%

2. Department with the highest observed attrition:
   {highest_attrition_dept}
   Attrition rate: {highest_attrition_dept_rate:.2f}%

3. Salary group with the highest observed attrition:
   {highest_attrition_salary}
   Attrition rate: {highest_attrition_salary_rate:.2f}%

4. Average satisfaction:
   Employees who stayed: {stayed_satisfaction:.2f}
   Employees who left:   {left_satisfaction:.2f}

5. Average monthly hours:
   Employees who stayed: {stayed_hours:.1f}
   Employees who left:   {left_hours:.1f}

6. Average number of projects:
   Employees who stayed: {stayed_projects:.2f}
   Employees who left:   {left_projects:.2f}

7. Promotion rate:
   {promotion_rate:.2f}% of employees received a promotion
   in the previous five years.

8. Work accident rate:
   {accident_rate:.2f}% of employees had a recorded
   work accident.
""")

# 19. VISUALIZATION 1 - ATTRITION


plt.figure(figsize=(8, 5))

attrition_counts = df["Attrition"].value_counts()

sns.barplot(
    x=attrition_counts.index,
    y=attrition_counts.values
)

plt.title("Employee Attrition")
plt.xlabel("Employee Status")
plt.ylabel("Number of Employees")

for i, value in enumerate(attrition_counts.values):
    plt.text(i, value + 100, f"{value:,}", ha="center")

plt.tight_layout()
plt.savefig(
    f"{OUTPUT_DIR}/01_attrition_overview.png",
    dpi=150
)
plt.show()

# 20. VISUALIZATION 2 - ATTRITION BY DEPARTMENT


plt.figure(figsize=(11, 6))

dept_plot = (
    df.groupby("Department")["left"]
    .mean()
    .sort_values(ascending=False) * 100
)

sns.barplot(
    x=dept_plot.values,
    y=dept_plot.index
)

plt.title("Attrition Rate by Department")
plt.xlabel("Attrition Rate (%)")
plt.ylabel("Department")

for i, value in enumerate(dept_plot.values):
    plt.text(value + 0.3, i, f"{value:.1f}%", va="center")

plt.tight_layout()
plt.savefig(
    f"{OUTPUT_DIR}/02_attrition_by_department.png",
    dpi=150
)
plt.show()


# 21. VISUALIZATION 3 - ATTRITION BY SALARY


plt.figure(figsize=(8, 5))

salary_order = ["low", "medium", "high"]

salary_plot = (
    df.groupby("salary")["left"]
    .mean()
    .reindex(salary_order) * 100
)

sns.barplot(
    x=salary_plot.index,
    y=salary_plot.values
)

plt.title("Attrition Rate by Salary Level")
plt.xlabel("Salary Level")
plt.ylabel("Attrition Rate (%)")

for i, value in enumerate(salary_plot.values):
    plt.text(i, value + 0.5, f"{value:.1f}%", ha="center")

plt.tight_layout()
plt.savefig(
    f"{OUTPUT_DIR}/03_attrition_by_salary.png",
    dpi=150
)
plt.show()

# ------------------------------------------------------------
# 22. VISUALIZATION 4 - SATISFACTION VS ATTRITION
# ------------------------------------------------------------

plt.figure(figsize=(9, 6))

sns.boxplot(
    data=df,
    x="Attrition",
    y="satisfaction_level"
)

plt.title("Employee Satisfaction vs Attrition")
plt.xlabel("Employee Status")
plt.ylabel("Satisfaction Level")

plt.tight_layout()
plt.savefig(
    f"{OUTPUT_DIR}/04_satisfaction_vs_attrition.png",
    dpi=150
)
plt.show()


# 23. VISUALIZATION 5 - MONTHLY HOURS VS ATTRITION


plt.figure(figsize=(9, 6))

sns.boxplot(
    data=df,
    x="Attrition",
    y="average_montly_hours"
)

plt.title("Monthly Working Hours vs Attrition")
plt.xlabel("Employee Status")
plt.ylabel("Average Monthly Hours")

plt.tight_layout()
plt.savefig(
    f"{OUTPUT_DIR}/05_hours_vs_attrition.png",
    dpi=150
)
plt.show()


# 24. VISUALIZATION 6 - PROJECTS VS ATTRITION


plt.figure(figsize=(10, 6))

project_attrition = (
    df.groupby("number_project")["left"]
    .mean() * 100
)

sns.barplot(
    x=project_attrition.index,
    y=project_attrition.values
)

plt.title("Attrition Rate by Number of Projects")
plt.xlabel("Number of Projects")
plt.ylabel("Attrition Rate (%)")

for i, value in enumerate(project_attrition.values):
    plt.text(i, value + 0.5, f"{value:.1f}%", ha="center")

plt.tight_layout()
plt.savefig(
    f"{OUTPUT_DIR}/06_projects_vs_attrition.png",
    dpi=150
)
plt.show()


# 25. VISUALIZATION 7 - TENURE VS ATTRITION


plt.figure(figsize=(10, 6))

tenure_attrition = (
    df.groupby("time_spend_company")["left"]
    .mean() * 100
)

sns.lineplot(
    x=tenure_attrition.index,
    y=tenure_attrition.values,
    marker="o"
)

plt.title("Attrition Rate by Years at Company")
plt.xlabel("Years at Company")
plt.ylabel("Attrition Rate (%)")

plt.tight_layout()
plt.savefig(
    f"{OUTPUT_DIR}/07_tenure_vs_attrition.png",
    dpi=150
)
plt.show()

# 26. VISUALIZATION 8 - PROMOTION VS ATTRITION


plt.figure(figsize=(8, 5))

promotion_attrition = (
    df.groupby("Promotion_Status")["left"]
    .mean() * 100
)

sns.barplot(
    x=promotion_attrition.index,
    y=promotion_attrition.values
)

plt.title("Attrition Rate by Promotion Status")
plt.xlabel("Promotion Status")
plt.ylabel("Attrition Rate (%)")

for i, value in enumerate(promotion_attrition.values):
    plt.text(i, value + 0.5, f"{value:.1f}%", ha="center")

plt.tight_layout()
plt.savefig(
    f"{OUTPUT_DIR}/08_promotion_vs_attrition.png",
    dpi=150
)
plt.show()


# 27. VISUALIZATION 9 - SATISFACTION DISTRIBUTION


plt.figure(figsize=(9, 6))

sns.histplot(
    data=df,
    x="satisfaction_level",
    bins=20,
    kde=True
)

plt.title("Distribution of Employee Satisfaction")
plt.xlabel("Satisfaction Level")
plt.ylabel("Number of Employees")

plt.tight_layout()
plt.savefig(
    f"{OUTPUT_DIR}/09_satisfaction_distribution.png",
    dpi=150
)
plt.show()


# 28. VISUALIZATION 10 - CORRELATION HEATMAP


plt.figure(figsize=(11, 8))

sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    linewidths=0.5
)

plt.title("HR Metrics Correlation Matrix")

plt.tight_layout()
plt.savefig(
    f"{OUTPUT_DIR}/10_correlation_heatmap.png",
    dpi=150
)
plt.show()


# 29. SAVE CLEANED ANALYSIS DATA


df.to_csv(
    f"{OUTPUT_DIR}/hr_analysis_data.csv",
    index=False
)

# 30. FINAL SUMMARY


print("\n" + "=" * 70)
print("EDA COMPLETE")
print("=" * 70)

print(f"""
Analysis completed successfully.

Output folder:
    {OUTPUT_DIR}/

Generated files include:
    - department_summary.csv
    - salary_summary.csv
    - attrition_summary.csv
    - correlation_matrix.csv
    - hr_analysis_data.csv
    - 10 management-focused charts

The analysis is descriptive only.
No machine learning models were used.
""")

print("=" * 70)