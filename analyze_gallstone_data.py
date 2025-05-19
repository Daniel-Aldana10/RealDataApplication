import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
import warnings
import os
warnings.filterwarnings('ignore')

# Configure visualization style
plt.style.use('seaborn')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)

# =====================
# FUNCIONES BASE
# =====================
def read_excel(file_path):
    """
    Read Excel file and display basic information about the data.
    """
    try:
        print(f"Reading file: {file_path}")
        df = pd.read_excel(file_path)
        df.columns = df.columns.str.strip()
        
        # Check for common variations of column names and standardize them
        column_mapping = {
            'Body Mass Index': 'BMI',
            'BMI (kg/m²)': 'BMI',
            'Body Mass Index (BMI)': 'BMI',
            'Gallstones': 'Gallstone Status',
            'Gallstone': 'Gallstone Status',
            'Status': 'Gallstone Status',
            'Total Body Fat': 'Total Body Fat Ratio (TBFR) (%)',
            'Fat Ratio': 'Total Body Fat Ratio (TBFR) (%)',
            'TBFR': 'Total Body Fat Ratio (TBFR) (%)',
            'Visceral Fat': 'Visceral Fat Rating (VFR)',
            'VFR': 'Visceral Fat Rating (VFR)',
            'Total Chol': 'Total Cholesterol',
            'Chol': 'Total Cholesterol',
            'HDL': 'High Density',
            'LDL': 'Low Density'
        }
        
        # Standardize column names if matches found
        for old_name, new_name in column_mapping.items():
            if old_name in df.columns:
                df.rename(columns={old_name: new_name}, inplace=True)
        
        # Print column names to diagnose issues
        print("\n=== AVAILABLE COLUMNS IN DATASET ===")
        for i, col in enumerate(df.columns):
            print(f"{i+1}. {col}")
        
        print("\n=== BASIC DATASET INFORMATION ===")
        print(f"Number of records: {df.shape[0]}")
        print(f"Number of columns: {df.shape[1]}")
        print("\n=== FIRST 5 ROWS ===")
        print(df.head())
        print("\n=== DATA TYPES ===")
        print(df.dtypes)
        print("\n=== MISSING VALUES ===")
        missing_values = df.isnull().sum()
        print(missing_values[missing_values > 0])
        print("\n=== DESCRIPTIVE STATISTICS ===")
        print(df.describe())
        
        # Create output directory if it doesn't exist
        os.makedirs('output', exist_ok=True)
        
        return df
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        return None

# =====================
# ANÁLISIS EXPLORATORIO DETALLADO
# =====================
def exploratory_analysis(df):
    """Detailed exploratory data analysis"""
    print("\n=== DATASET INFORMATION ===")
    print(df.info())
    print("\n=== DESCRIPTIVE STATISTICS ===")
    print(df.describe(include='all'))
    print("\n=== VARIABLE ANALYSIS ===")
    for column in df.columns:
        print(f"\nVariable: {column}")
        print(df[column].value_counts(dropna=False))
        print(f"Null values: {df[column].isnull().sum()}")

# =====================
# ANÁLISIS POR ESTADO DE CÁLCULOS BILIARES
# =====================
def analyze_by_gallstone_status(df):
    """Analyze data by gallstone status"""
    if 'Gallstone Status' not in df.columns:
        print("Column 'Gallstone Status' not present in the dataset.")
        return
    try:
        groups = df.groupby('Gallstone Status')
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        print("\n=== STATISTICS BY GALLSTONE STATUS ===")
        print(groups[numeric_cols].mean())
        relevant_cols = ['Age', 'BMI', 'Total Body Fat Ratio (TBFR) (%)',
                        'Visceral Fat Rating (VFR)', 'Glucose']
        existing_cols = [col for col in relevant_cols if col in df.columns]
        if existing_cols:
            print("\n=== COMPARISON OF KEY VARIABLES ===")
            print(groups[existing_cols].mean())
    except Exception as e:
        print(f"Error analyzing by gallstone status: {e}")

# =====================
# VISUALIZACIONES AVANZADAS
# =====================
def create_visualizations(df):
    """Create detailed visualizations for data analysis"""
    
    # 1. Categorical Variables Analysis
    cat_vars = [col for col in ['Gender', 'Comorbidity', 'Coronary Art', 'Hypothyroidi', 'Hyperlipidem', 'Diabetes Mel'] 
                if col in df.columns]
    if cat_vars:
        plt.figure(figsize=(15, 10))
        for i, col in enumerate(cat_vars, 1):
            plt.subplot(2, 3, i)
            sns.countplot(data=df, x=col, hue='Gallstone Status')
            plt.title(f'{col} by Gallstone Status')
        plt.tight_layout()
        plt.savefig('categorical_variables_analysis.png')
        plt.close()
    
    # 2. Demographic Variables Analysis
    if 'Age' in df.columns and 'Gallstone Status' in df.columns:
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        sns.boxplot(data=df, x='Gallstone Status', y='Age')
        plt.title('Age Distribution by Gallstone Status')
        plt.subplot(1, 2, 2)
        if 'Gender' in df.columns:
            sns.histplot(data=df, x='Age', hue='Gender', multiple="stack")
            plt.title('Age Distribution by Gender')
        plt.tight_layout()
        plt.savefig('demographic_variables_analysis.png')
        plt.close()
    
    # 3. Body Composition Analysis
    if all(col in df.columns for col in ['Height', 'Weight', 'Gallstone Status']):
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        sns.scatterplot(data=df, x='Height', y='Weight', hue='Gallstone Status')
        plt.title('Height vs Weight by Gallstone Status')
        plt.subplot(1, 2, 2)
        sns.boxplot(data=df, x='Gallstone Status', y='Weight')
        plt.title('Weight Distribution by Gallstone Status')
        plt.tight_layout()
        plt.savefig('body_composition_analysis.png')
        plt.close()
    
    # 4. Body Composition Correlations
    body_comp_vars = [col for col in ['Height', 'Weight', 'Age'] if col in df.columns]
    if len(body_comp_vars) >= 2:
        plt.figure(figsize=(10, 8))
        sns.heatmap(df[body_comp_vars].corr(), annot=True, cmap='coolwarm')
        plt.title('Body Composition Correlations')
        plt.tight_layout()
        plt.savefig('body_composition_correlation.png')
        plt.close()
    
    # 5. General Correlation Matrix - IMPROVED VERSION
    create_improved_correlation_matrix(df)

def create_improved_correlation_matrix(df):
    """Create an improved correlation matrix that is more readable"""
    # Select only numeric columns
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    
    # Calculate correlation matrix
    corr_matrix = numeric_df.corr()
    
    # Create a more readable correlation matrix by breaking it into smaller chunks if needed
    if corr_matrix.shape[0] > 15:
        # Split into smaller correlation matrices if there are many variables
        for i in range(0, len(corr_matrix.columns), 10):
            end_idx = min(i + 10, len(corr_matrix.columns))
            subset_cols = corr_matrix.columns[i:end_idx]
            
            plt.figure(figsize=(12, 10))
            mask = np.zeros_like(corr_matrix.loc[subset_cols, subset_cols], dtype=bool)
            mask[np.triu_indices_from(mask, k=1)] = True
            
            cmap = sns.diverging_palette(220, 10, as_cmap=True)
            sns.heatmap(corr_matrix.loc[subset_cols, subset_cols], 
                      mask=mask,
                      cmap=cmap, 
                      vmax=1.0, 
                      vmin=-1.0,
                      center=0,
                      square=True, 
                      linewidths=.5, 
                      annot=True, 
                      fmt='.2f',
                      annot_kws={"size": 8})
            
            plt.title(f'Correlation Matrix (Variables {i+1}-{end_idx})')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(f'correlation_matrix_part_{i//10+1}.png')
            plt.close()
    else:
        # Create a single correlation matrix for smaller datasets
        plt.figure(figsize=(12, 10))
        mask = np.zeros_like(corr_matrix, dtype=bool)
        mask[np.triu_indices_from(mask, k=1)] = True
        
        cmap = sns.diverging_palette(220, 10, as_cmap=True)
        sns.heatmap(corr_matrix, 
                  mask=mask,
                  cmap=cmap, 
                  vmax=1.0, 
                  vmin=-1.0,
                  center=0,
                  square=True, 
                  linewidths=.5, 
                  annot=True, 
                  fmt='.2f')
        
        plt.title('General Correlation Matrix')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('general_correlation_matrix.png')
        plt.close()

# =====================
# VISUALIZACIÓN INDIVIDUAL
# =====================
def visualize_distribution(df, variable):
    """Visualize distribution of a variable by gallstone status"""
    # Check if the variable exists in any form
    if variable not in df.columns:
        # Try to find similar column names
        similar_cols = [col for col in df.columns if variable.lower() in col.lower()]
        if similar_cols:
            print(f"Variable '{variable}' not found exactly, but found similar columns: {similar_cols}")
            variable = similar_cols[0]  # Use the first similar column found
            print(f"Using column '{variable}' instead")
        else:
            print(f"Variable '{variable}' not in the dataset and no similar columns found.")
            return
    
    try:
        plt.figure(figsize=(10, 6))
        sns.histplot(data=df, x=variable, hue='Gallstone Status', kde=True,
                     common_norm=False, palette='Set2')
        plt.title(f'Distribution of {variable} by Gallstone Status')
        plt.xlabel(variable)
        plt.ylabel('Frequency')
        plt.tight_layout()
        save_path = os.path.join('output', f'distribution_{variable.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("%", "pct")}.png')
        plt.savefig(save_path)
        print(f"Saved distribution plot to {save_path}")
        plt.close()
    except Exception as e:
        print(f"Error creating histogram for {variable}: {e}")

# =====================
# NEW ANALYSIS FUNCTIONS
# =====================

def analyze_bmi_vs_age(df):
    """Analyze relationship between BMI and Age by Gallstone Status"""
    # Check for BMI column
    bmi_col = None
    if 'BMI' in df.columns:
        bmi_col = 'BMI'
    else:
        # Try to find BMI-like column
        bmi_candidates = [col for col in df.columns if 'bmi' in col.lower() or 'mass index' in col.lower()]
        if bmi_candidates:
            bmi_col = bmi_candidates[0]
            print(f"BMI column not found. Using '{bmi_col}' instead.")
        else:
            # If no BMI column exists, try to calculate it
            if 'Weight' in df.columns and 'Height' in df.columns:
                print("Calculating BMI from Weight and Height...")
                # BMI = weight(kg) / (height(m))²
                df['BMI'] = df['Weight'] / ((df['Height']/100) ** 2)
                bmi_col = 'BMI'
            else:
                print("Cannot analyze BMI vs Age: No BMI column found and cannot calculate it.")
                return
    
    # Check for Age column
    if 'Age' not in df.columns:
        age_candidates = [col for col in df.columns if 'age' in col.lower() or 'edad' in col.lower()]
        if age_candidates:
            age_col = age_candidates[0]
            print(f"Age column not found. Using '{age_col}' instead.")
        else:
            print("Cannot analyze BMI vs Age: No Age column found.")
            return
    else:
        age_col = 'Age'
    
    # Check for Gallstone Status column
    if 'Gallstone Status' not in df.columns:
        gs_candidates = [col for col in df.columns if 'gallstone' in col.lower() or 'status' in col.lower()]
        if gs_candidates:
            gs_col = gs_candidates[0]
            print(f"Gallstone Status column not found. Using '{gs_col}' instead.")
        else:
            print("Cannot analyze BMI vs Age: No Gallstone Status column found.")
            return
    else:
        gs_col = 'Gallstone Status'
    
    try:
        # 1. BMI vs Age scatter plot
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df, x=age_col, y=bmi_col, hue=gs_col, palette='Set1')
        
        # Add regression lines if there are enough data points
        if len(df[df[gs_col] == 0]) > 2:
            sns.regplot(data=df[df[gs_col] == 0], x=age_col, y=bmi_col, 
                      scatter=False, line_kws={"linestyle": "--"})
        
        if len(df[df[gs_col] == 1]) > 2:
            sns.regplot(data=df[df[gs_col] == 1], x=age_col, y=bmi_col, 
                      scatter=False, line_kws={"linestyle": "-."})
        
        plt.title(f'Relationship between {bmi_col} and {age_col} by {gs_col}')
        plt.xlabel(age_col)
        plt.ylabel(bmi_col)
        plt.tight_layout()
        save_path = os.path.join('output', 'bmi_vs_age.png')
        plt.savefig(save_path)
        print(f"Saved BMI vs Age plot to {save_path}")
        plt.close()
        
        # Statistical analysis
        print(f"\n=== {bmi_col} vs {age_col} CORRELATION BY {gs_col} ===")
        corr_df = df[[age_col, bmi_col, gs_col]].groupby(gs_col).corr()
        print(corr_df)
        
        # Age group analysis
        age_min = df[age_col].min()
        age_max = df[age_col].max()
        
        # Create age bins based on data range
        age_bins = [age_min, age_min + (age_max-age_min)*0.2, age_min + (age_max-age_min)*0.4, 
                    age_min + (age_max-age_min)*0.6, age_min + (age_max-age_min)*0.8, age_max]
        age_labels = ['Very Young', 'Young', 'Middle', 'Senior', 'Elderly']
        
        df['Age Group'] = pd.cut(df[age_col], bins=age_bins, labels=age_labels)
        
        print(f"\n=== {bmi_col} BY AGE GROUP AND {gs_col} ===")
        print(df.groupby(['Age Group', gs_col])[bmi_col].mean())
        
        # Visualize BMI by age group
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=df, x='Age Group', y=bmi_col, hue=gs_col)
        plt.title(f'{bmi_col} Distribution by Age Group and {gs_col}')
        plt.xlabel('Age Group')
        plt.ylabel(bmi_col)
        plt.tight_layout()
        save_path = os.path.join('output', 'bmi_by_age_group.png')
        plt.savefig(save_path)
        print(f"Saved BMI by Age Group plot to {save_path}")
        plt.close()
    except Exception as e:
        print(f"Error analyzing BMI vs Age: {e}")

def analyze_height_weight_gender(df):
    """Analyze relationship between Height and Weight by Gender"""
    if all(col in df.columns for col in ['Height', 'Weight', 'Gender']):
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df, x='Height', y='Weight', hue='Gender', style='Gallstone Status', palette='Set2')
        
        # Add regression lines
        sns.regplot(data=df[df['Gender'] == 0], x='Height', y='Weight', 
                   scatter=False, line_kws={"linestyle": "--"})
        sns.regplot(data=df[df['Gender'] == 1], x='Height', y='Weight', 
                   scatter=False, line_kws={"linestyle": "-."})
        
        plt.title('Relationship between Height and Weight by Gender')
        plt.xlabel('Height (cm)')
        plt.ylabel('Weight (kg)')
        plt.tight_layout()
        plt.savefig('height_vs_weight_by_gender.png')
        plt.close()

def analyze_age_gallstone_gender(df):
    """Analyze age distribution by gallstone status and gender"""
    if all(col in df.columns for col in ['Age', 'Gallstone Status', 'Gender']):
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df, x='Gallstone Status', y='Age', hue='Gender')
        plt.title('Age Distribution by Gallstone Status and Gender')
        plt.tight_layout()
        plt.savefig('age_by_gallstone_and_gender.png')
        plt.close()
        
        # Statistical summary
        print("\n=== AGE STATISTICS BY GALLSTONE STATUS AND GENDER ===")
        print(df.groupby(['Gallstone Status', 'Gender'])['Age'].describe())

def analyze_clinical_variables(df):
    """Analyze clinical variables correlation"""
    clinical_vars = [col for col in ['Total Cholesterol', 'Glucose', 'Triglyceride', 
                                    'Low Density', 'High Density', 'Hepatic Fat', 'Total Body Fat Ratio (TBFR) (%)'] 
                     if col in df.columns]
    
    if clinical_vars and len(clinical_vars) >= 2:
        plt.figure(figsize=(12, 10))
        sns.heatmap(df[clinical_vars].corr(), annot=True, cmap='YlGnBu')
        plt.title('Clinical Variables Correlation')
        plt.tight_layout()
        plt.savefig('clinical_variables_correlation.png')
        plt.close()
        
        # Pairplot for key clinical variables (limit to 5 variables to keep plot manageable)
        key_vars = clinical_vars[:min(5, len(clinical_vars))] + ['Gallstone Status']
        plt.figure(figsize=(15, 12))
        sns.pairplot(df[key_vars], hue='Gallstone Status', diag_kind='kde')
        plt.suptitle('Relationships between Key Clinical Variables', y=1.02)
        plt.tight_layout()
        plt.savefig('clinical_variables_pairplot.png')
        plt.close()

def analyze_comorbidity_prevalence(df):
    """Analyze comorbidity prevalence by gallstone status"""
    comorbs = [col for col in ['Comorbidity', 'Coronary Art', 'Hypothyroidi', 'Hyperlipidem', 'Diabetes Mel'] 
              if col in df.columns]
    
    if comorbs and 'Gallstone Status' in df.columns:
        # Create visualization
        plt.figure(figsize=(12, 8))
        for i, col in enumerate(comorbs):
            plt.subplot(2, 3, i+1)
            # Calculate and plot percentages instead of counts
            ax = sns.countplot(data=df, x=col, hue='Gallstone Status')
            
            # Add percentage labels
            for p in ax.patches:
                height = p.get_height()
                ax.text(p.get_x() + p.get_width()/2.,
                        height + 0.1,
                        '{:1.1f}%'.format(100 * height / len(df)),
                        ha="center")
            
            plt.title(f'{col} by Gallstone Status')
        
        plt.tight_layout()
        plt.savefig('comorbidity_prevalence.png')
        plt.close()
        
        # Statistical analysis
        print("\n=== COMORBIDITY PREVALENCE BY GALLSTONE STATUS ===")
        for col in comorbs:
            print(f'\nFrequency of {col} by Gallstone Status (%):\n')
            print(df.groupby('Gallstone Status')[col].value_counts(normalize=True).mul(100).round(1))
            
            # Chi-square test
            try:
                from scipy.stats import chi2_contingency
                contingency_table = pd.crosstab(df['Gallstone Status'], df[col])
                chi2, p, dof, expected = chi2_contingency(contingency_table)
                print(f"Chi-square test: chi2={chi2:.2f}, p-value={p:.4f}")
            except Exception as e:
                print(f"Could not perform chi-square test: {e}")

def analyze_visceral_fat(df):
    """Analyze visceral fat rating by gallstone status"""
    vfr_column = next((col for col in df.columns if 'Visceral Fat' in col), None)
    
    if vfr_column and 'Gallstone Status' in df.columns:
        plt.figure(figsize=(8, 6))
        sns.boxplot(data=df, x='Gallstone Status', y=vfr_column)
        plt.title('Visceral Fat Rating by Gallstone Status')
        plt.tight_layout()
        plt.savefig('visceral_fat_by_gallstone.png')
        plt.close()
        
        # Statistical analysis
        print(f"\n=== VISCERAL FAT STATISTICS BY GALLSTONE STATUS ===")
        print(df.groupby('Gallstone Status')[vfr_column].describe())
        
        # T-test
        try:
            from scipy.stats import ttest_ind
            group0 = df[df['Gallstone Status'] == 0][vfr_column].dropna()
            group1 = df[df['Gallstone Status'] == 1][vfr_column].dropna()
            t_stat, p_val = ttest_ind(group0, group1, equal_var=False)
            print(f"T-test: t={t_stat:.2f}, p-value={p_val:.4f}")
        except Exception as e:
            print(f"Could not perform t-test: {e}")

def multi_variable_analysis(df):
    """Perform multi-variable analysis with key health indicators"""
    # Define the gallstone status column
    gs_col = 'Gallstone Status'
    if gs_col not in df.columns:
        gs_candidates = [col for col in df.columns if 'gallstone' in col.lower() or 'status' in col.lower()]
        if gs_candidates:
            gs_col = gs_candidates[0]
        else:
            print("Cannot perform multi-variable analysis: No Gallstone Status column found")
            return
    
    # Find appropriate health indicators
    potential_indicators = [
        'BMI', 'Body Mass Index', 
        'Total Body Fat Ratio (TBFR) (%)', 'TBFR', 'Body Fat', 
        'Visceral Fat Rating (VFR)', 'VFR', 'Visceral Fat',
        'Glucose', 'Blood Glucose',
        'Total Cholesterol', 'Cholesterol',
        'Triglyceride', 'TG',
        'Age', 'Weight', 'Height'
    ]
    
    # Find which indicators are available in the dataset
    available_indicators = []
    for indicator in potential_indicators:
        if indicator in df.columns:
            available_indicators.append(indicator)
    
    # Remove duplicates (e.g., if we have both 'BMI' and 'Body Mass Index')
    unique_indicators = []
    added_concepts = set()
    for indicator in available_indicators:
        # Extract the core concept
        concept = indicator.split()[0].lower()
        if concept not in added_concepts:
            unique_indicators.append(indicator)
            added_concepts.add(concept)
    
    print(f"Available health indicators for analysis: {unique_indicators}")
    
    # Need at least 3 indicators for 3D analysis
    if len(unique_indicators) < 3 or gs_col not in df.columns:
        print(f"Not enough indicators for 3D analysis. Found {len(unique_indicators)}: {unique_indicators}")
        return
    
    try:
        # Use up to 4 variables for visualization
        plot_vars = unique_indicators[:min(4, len(unique_indicators))]
        
        # Create multiple 3D plots combining different variables
        from itertools import combinations
        for combo in combinations(plot_vars, 3):
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            # Plot points by gallstone status
            for gs_value in df[gs_col].unique():
                subset = df[df[gs_col] == gs_value]
                if not subset.empty:
                    try:
                        ax.scatter(subset[combo[0]], subset[combo[1]], subset[combo[2]], 
                                  label=f'{gs_col} {gs_value}',
                                  alpha=0.7)
                    except Exception as e:
                        print(f"Error plotting 3D with {combo}: {e}")
            
            ax.set_xlabel(combo[0])
            ax.set_ylabel(combo[1])
            ax.set_zlabel(combo[2])
            plt.title(f'3D Relationship: {combo[0]} vs {combo[1]} vs {combo[2]}')
            plt.legend()
            plt.tight_layout()
            
            # Create safe filename
            safe_name = "_".join(c.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("%", "pct") for c in combo)
            save_path = os.path.join('output', f'3d_relationship_{safe_name}.png')
            plt.savefig(save_path)
            print(f"Saved 3D Relationship plot to {save_path}")
            plt.close()
    except Exception as e:
        print(f"Error performing multi-variable analysis: {e}")

# =====================
# MAIN
# =====================
if __name__ == "__main__":
    excel_path = 'dataset-uci.xlsx'
    data = read_excel(excel_path)
    if data is not None:
        # Create a plots directory to store all visualizations
        output_dir = 'output'
        os.makedirs(output_dir, exist_ok=True)
        print(f"Created output directory for plots: {output_dir}")
        
        exploratory_analysis(data)
        analyze_by_gallstone_status(data)
        create_visualizations(data)
        
        print("\n=== RUNNING ADVANCED ANALYSES ===")
        
        # Additional analyses
        analyze_bmi_vs_age(data)
        analyze_height_weight_gender(data)
        analyze_age_gallstone_gender(data)
        analyze_clinical_variables(data)
        analyze_comorbidity_prevalence(data)
        analyze_visceral_fat(data)
        multi_variable_analysis(data)
        
        # Visualize distribution of key variables
        print("\n=== GENERATING VARIABLE DISTRIBUTIONS ===")
        key_variables = ['Age', 'BMI', 'Glucose', 'Total Body Fat Ratio (TBFR) (%)', 
                         'Visceral Fat Rating (VFR)', 'Total Cholesterol', 'Triglyceride']
        for var in key_variables:
            visualize_distribution(data, var)
        
        # List all the generated plots
        print("\nAnalysis completed. The following visualizations have been generated in the 'output' directory:")
        for i, filename in enumerate(sorted(os.listdir(output_dir))):
            if filename.endswith('.png'):
                print(f"{i+1}. {filename}")
        
        print("\nYou can find all visualizations in the 'output' directory.")