import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
import warnings
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
    if variable not in df.columns:
        print(f"Variable '{variable}' not in the dataset.")
        return
    try:
        plt.figure(figsize=(10, 6))
        sns.histplot(data=df, x=variable, hue='Gallstone Status', kde=True,
                     common_norm=False, palette='Set2')
        plt.title(f'Distribution of {variable} by Gallstone Status')
        plt.xlabel(variable)
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.savefig(f'distribution_{variable.lower().replace(" ", "_")}.png')
        plt.close()
    except Exception as e:
        print(f"Error creating histogram: {e}")

# =====================
# NEW ANALYSIS FUNCTIONS
# =====================

def analyze_bmi_vs_age(df):
    """Analyze relationship between BMI and Age by Gallstone Status"""
    if 'BMI' in df.columns and 'Age' in df.columns and 'Gallstone Status' in df.columns:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df, x='Age', y='BMI', hue='Gallstone Status', palette='Set1')
        
        # Add a regression line for each group
        sns.regplot(data=df[df['Gallstone Status'] == 0], x='Age', y='BMI', 
                   scatter=False, line_kws={"linestyle": "--"})
        sns.regplot(data=df[df['Gallstone Status'] == 1], x='Age', y='BMI', 
                   scatter=False, line_kws={"linestyle": "-."})
        
        plt.title('Relationship between BMI and Age by Gallstone Status')
        plt.xlabel('Age')
        plt.ylabel('BMI')
        plt.tight_layout()
        plt.savefig('bmi_vs_age.png')
        plt.close()
        
        # Statistical analysis
        print("\n=== BMI vs AGE CORRELATION BY GALLSTONE STATUS ===")
        print(df[['Age', 'BMI', 'Gallstone Status']].groupby('Gallstone Status').corr())
        
        # Age group analysis
        df['Age Group'] = pd.cut(df['Age'], bins=[0, 30, 40, 50, 60, 100], 
                              labels=['<30', '30-40', '40-50', '50-60', '60+'])
        
        print("\n=== BMI BY AGE GROUP AND GALLSTONE STATUS ===")
        print(df.groupby(['Age Group', 'Gallstone Status'])['BMI'].mean())
        
        # Visualize BMI by age group
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=df, x='Age Group', y='BMI', hue='Gallstone Status')
        plt.title('BMI Distribution by Age Group and Gallstone Status')
        plt.xlabel('Age Group')
        plt.ylabel('BMI')
        plt.tight_layout()
        plt.savefig('bmi_by_age_group.png')
        plt.close()

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
    # Check for key health indicators
    health_indicators = [col for col in ['BMI', 'Total Body Fat Ratio (TBFR) (%)', 'Visceral Fat Rating (VFR)', 
                                        'Glucose', 'Total Cholesterol', 'Triglyceride'] 
                         if col in df.columns]
    
    if health_indicators and len(health_indicators) >= 3 and 'Gallstone Status' in df.columns:
        # Choose up to 4 variables for 3D visualization
        plot_vars = health_indicators[:min(4, len(health_indicators))]
        
        # Create multiple 3D plots combining different variables
        from itertools import combinations
        for combo in combinations(plot_vars, 3):
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            # Plot points
            for gs in [0, 1]:
                subset = df[df['Gallstone Status'] == gs]
                ax.scatter(subset[combo[0]], subset[combo[1]], subset[combo[2]], 
                          label=f'Gallstone Status {gs}',
                          alpha=0.7)
            
            ax.set_xlabel(combo[0])
            ax.set_ylabel(combo[1])
            ax.set_zlabel(combo[2])
            plt.title(f'3D Relationship: {combo[0]} vs {combo[1]} vs {combo[2]}')
            plt.legend()
            plt.tight_layout()
            plt.savefig(f'3d_relationship_{"_".join(c.lower().replace(" ", "_") for c in combo)}.png')
            plt.close()

# =====================
# MAIN
# =====================
if __name__ == "__main__":
    # Reemplaza 'tu_archivo.xlsx' con la ruta de tu archivo
    excel_path = 'dataset-uci.xlsx'

    # Leer los datos
    data = read_excel(excel_path)

    # Si los datos se cargaron correctamente
    if data is not None:
        exploratory_analysis(data)
        analyze_by_gallstone_status(data)
        create_visualizations(data)
        
        # Additional analyses
        analyze_bmi_vs_age(data)
        analyze_height_weight_gender(data)
        analyze_age_gallstone_gender(data)
        analyze_clinical_variables(data)
        analyze_comorbidity_prevalence(data)
        analyze_visceral_fat(data)
        multi_variable_analysis(data)
        
        # Visualize distribution of key variables
        key_variables = ['Age', 'BMI', 'Glucose', 'Total Body Fat Ratio (TBFR) (%)', 
                         'Visceral Fat Rating (VFR)', 'Total Cholesterol', 'Triglyceride']
        for var in key_variables:
            if var in data.columns:
                visualize_distribution(data, var)
                
        print("\nAnalysis completed. The following visualizations have been generated:")
        print("1. Categorical Variables Analysis (categorical_variables_analysis.png)")
        print("2. Demographic Variables Analysis (demographic_variables_analysis.png)")
        print("3. Body Composition Analysis (body_composition_analysis.png)")
        print("4. Body Composition Correlations (body_composition_correlation.png)")
        print("5. Correlation Matrix (general_correlation_matrix.png)")
        print("6. BMI vs Age Analysis (bmi_vs_age.png, bmi_by_age_group.png)")
        print("7. Height vs Weight by Gender (height_vs_weight_by_gender.png)")
        print("8. Age by Gallstone and Gender (age_by_gallstone_and_gender.png)")
        print("9. Clinical Variables Analysis (clinical_variables_correlation.png, clinical_variables_pairplot.png)")
        print("10. Comorbidity Prevalence (comorbidity_prevalence.png)")
        print("11. Visceral Fat Analysis (visceral_fat_by_gallstone.png)")
        print("12. Multi-Variable 3D Analysis (3d_relationship_*.png)")
        print("13. Variable Distributions (distribution_*.png)")