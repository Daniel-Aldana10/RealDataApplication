# Gallstone Data Analysis Project

This project performs a comprehensive exploratory data analysis of medical data related to gallstones, including demographic variables, body composition measurements, and various health indicators.

## Data Dictionary

### Variables Description

| Variable Name | Role | Type | Demographic | Description | Units | Missing Values |
|--------------|------|------|-------------|-------------|-------|----------------|
| Gallstone Status | Target | Binary | | Target variable, Gallstones present(1), and absent(0) | | no |
| Age | Feature | Integer | Age | Age of the person | | no |
| Gender | Feature | Categorical | Gender | Gender of the person | | no |
| Comorbidity | Feature | Categorical | | Concomitant diseases | | no |
| Coronary Artery Disease (CAD) | Feature | Binary | | Cardiovascular disease | | no |
| Hypothyroidism | Feature | Binary | | Underactive thyroid gland | | no |
| Hyperlipidemia | Feature | Binary | | High levels of fat in the blood | | no |
| Diabetes Mellitus (DM) | Feature | Binary | | High blood sugar | | no |
| Height | Feature | Integer | | Height is the length | | no |
| Weight | Feature | Continuous | | Body weight | | no |

### Class Labels

- **Gallstone Status**: 0 (No), 1 (Yes)
- **Gender**: 0 (Male), 1 (Female)
- **Comorbidity**: 
  - 0 (No comorbidities present)
  - 1 (One comorbid condition)
  - 2 (Two comorbid conditions)
  - 3 (Three or more comorbid conditions)
- **Coronary Artery Disease**: 0 (No), 1 (Yes)
- **Hypothyroidism**: 0 (No), 1 (Yes)
- **Hyperlipidemia**: 0 (No), 1 (Yes)
- **Diabetes Mellitus**: 0 (No), 1 (Yes)
- **Hepatic Fat Accumulation (HFA)**:
  - 0 (No fat accumulation)
  - 1 (Grade 1 - mild)
  - 2 (Grade 2 - moderate)
  - 3 (Grade 3 - severe)
  - 4 (Grade 4 - very severe)

## Sample Data (10 Examples)

```
| Gallstone | Age | Gender | Comorbidity | CAD | Hypothyroidism | Hyperlipidemia | DM | Height | Weight |
|-----------|-----|--------|-------------|-----|----------------|----------------|----| -------|---------|
| 0         | 45  | 1      | 1           | 0   | 0             | 1              | 0  | 165    | 70.5    |
| 1         | 62  | 0      | 2           | 1   | 0             | 1              | 1  | 175    | 82.3    |
| 0         | 38  | 1      | 0           | 0   | 0             | 0              | 0  | 160    | 58.7    |
| 1         | 55  | 1      | 1           | 0   | 1             | 1              | 0  | 162    | 65.4    |
| 0         | 42  | 0      | 0           | 0   | 0             | 0              | 0  | 178    | 76.8    |
| 1         | 58  | 1      | 2           | 1   | 1             | 1              | 1  | 159    | 68.2    |
| 0         | 35  | 0      | 0           | 0   | 0             | 0              | 0  | 172    | 71.5    |
| 1         | 65  | 1      | 3           | 1   | 1             | 1              | 1  | 157    | 63.8    |
| 0         | 40  | 1      | 1           | 0   | 0             | 1              | 0  | 164    | 61.2    |
| 1         | 52  | 0      | 2           | 1   | 0             | 1              | 1  | 170    | 79.4    |
```

## Analysis Visualizations

The script generates several visualizations to help understand the relationships between variables:

1. **Categorical Variables Analysis** (categorical_variables_analysis.png)
   - Shows the distribution of categorical variables (Gender, Comorbidity, CAD, etc.) in relation to Gallstone Status
   - Helps identify if certain conditions are more prevalent in patients with gallstones

2. **Demographic Variables Analysis** (demographic_variables_analysis.png)
   - Displays age distribution by gallstone status and gender
   - Helps understand if age is a significant factor in gallstone development
   - Shows gender-specific age patterns

3. **Body Composition Analysis** (body_composition_analysis.png)
   - Scatter plot of Height vs Weight by Gallstone Status
   - Box plot of Weight distribution by Gallstone Status
   - Helps identify if body composition metrics are related to gallstone presence

4. **Body Composition Correlations** (body_composition_correlation.png)
   - Heat map showing correlations between body composition variables
   - Helps understand relationships between Height, Weight, and Age

5. **General Correlation Matrix** (general_correlation_matrix.png)
   - Comprehensive correlation analysis between all numeric variables
   - Helps identify potential relationships between different health indicators

## Requirements

- Python 3.7 or higher
- Required packages listed in `requirements.txt`

## Installation

1. Clone this repository or download the files
2. Install the dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Ensure the `dataset-uci.xlsx` file is in the same directory as the script
2. Run the script:
```bash
python analyze_gallstone_data.py
```

The script will generate all visualizations and provide a detailed analysis of the data in the console output. 

def analizar_bmi_vs_edad(df):
    if 'BMI' in df.columns and 'Age' in df.columns:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df, x='Age', y='BMI', hue='Gallstone Status', palette='Set1')
        plt.title('Relación entre IMC y Edad por Estado de Cálculos Biliares')
        plt.xlabel('Edad')
        plt.ylabel('IMC')
        plt.tight_layout()
        plt.savefig('bmi_vs_age.png')
        plt.show()
        print(df[['Age', 'BMI', 'Gallstone Status']].groupby('Gallstone Status').corr()) 