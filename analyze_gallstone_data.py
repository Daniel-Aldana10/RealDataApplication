import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# =====================
# FUNCIONES BASE
# =====================
def leer_excel(ruta_archivo):
    """
    Lee un archivo Excel y muestra información básica sobre los datos.

    Args:
        ruta_archivo (str): Ruta al archivo Excel (.xlsx)

    Returns:
        pandas.DataFrame: DataFrame con los datos del archivo
    """
    try:
        # Leer el archivo Excel
        print(f"Leyendo archivo: {ruta_archivo}")
        df = pd.read_excel(ruta_archivo)
        df.columns = df.columns.str.strip()

        # Mostrar información básica
        print("\n=== INFORMACIÓN BÁSICA DEL DATASET ===")
        print(f"Número de registros: {df.shape[0]}")
        print(f"Número de columnas: {df.shape[1]}")
        print("\n=== PRIMERAS 5 FILAS ===")
        print(df.head())

        # Mostrar tipos de datos
        print("\n=== TIPOS DE DATOS ===")
        print(df.dtypes)

        # Verificar valores faltantes
        print("\n=== VALORES FALTANTES ===")
        missing_values = df.isnull().sum()
        print(missing_values[missing_values > 0])

        # Estadísticas descriptivas de las columnas numéricas
        print("\n=== ESTADÍSTICAS DESCRIPTIVAS ===")
        print(df.describe())

        return df

    except Exception as e:
        print(f"Error al leer el archivo Excel: {e}")
        return None

# =====================
# ANÁLISIS EXPLORATORIO DETALLADO
# =====================
def analisis_exploratorio(df):
    print("\n=== INFORMACIÓN DEL DATASET ===")
    print(df.info())
    print("\n=== ESTADÍSTICAS DESCRIPTIVAS ===")
    print(df.describe(include='all'))
    print("\n=== ANÁLISIS DE VARIABLES ===")
    for column in df.columns:
        print(f"\nVariable: {column}")
        print(df[column].value_counts(dropna=False))
        print(f"Valores nulos: {df[column].isnull().sum()}")

# =====================
# ANÁLISIS POR ESTADO DE CÁLCULOS BILIARES
# =====================
def analizar_por_gallstone_status(df):
    """
    Analiza las diferencias entre pacientes con y sin cálculos biliares

    Args:
        df (pandas.DataFrame): DataFrame con los datos
    """
    if 'Gallstone Status' not in df.columns:
        print("La columna 'Gallstone Status' no está presente en el dataset.")
        return

    try:
        # Agrupar por estado de cálculos biliares
        grupos = df.groupby('Gallstone Status')

        # Obtener columnas numéricas
        columnas_numericas = df.select_dtypes(include=['float64', 'int64']).columns

        # Estadísticas por grupo
        print("\n=== ESTADÍSTICAS POR ESTADO DE CÁLCULOS BILIARES ===")
        estadisticas = grupos[columnas_numericas].mean()
        print(estadisticas)

        # Para las 5 variables numéricas más relevantes, mostrar comparación
        columnas_relevantes = ['Age', 'BMI', 'Total Body Fat Ratio (TBFR) (%)',
                               'Visceral Fat Rating (VFR)', 'Glucose']

        # Asegurarse de que las columnas existan
        columnas_existentes = [col for col in columnas_relevantes if col in df.columns]

        if columnas_existentes:
            print("\n=== COMPARACIÓN DE VARIABLES CLAVE ===")
            print(grupos[columnas_existentes].mean())

    except Exception as e:
        print(f"Error al analizar por estado de cálculos biliares: {e}")

# =====================
# VISUALIZACIONES AVANZADAS
# =====================
def crear_visualizaciones(df):
    # 1. Variables categóricas
    cat_vars = [col for col in ['Gender', 'Comorbidity', 'Coronary Art', 'Hypothyroidi', 'Hyperlipidem', 'Diabetes Mel'] if col in df.columns]
    if cat_vars:
        plt.figure(figsize=(15, 10))
        for i, col in enumerate(cat_vars, 1):
            plt.subplot(2, 3, i)
            sns.countplot(data=df, x=col, hue='Gallstone Status')
            plt.title(f'{col} por Gallstone Status')
        plt.tight_layout()
        plt.savefig('categorical_variables_analysis.png')
        plt.close()
    # 2. Variables demográficas
    if 'Age' in df.columns and 'Gallstone Status' in df.columns:
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        sns.boxplot(data=df, x='Gallstone Status', y='Age')
        plt.title('Distribución de Edad por Gallstone Status')
        plt.subplot(1, 2, 2)
        if 'Gender' in df.columns:
            sns.histplot(data=df, x='Age', hue='Gender', multiple="stack")
            plt.title('Distribución de Edad por Género')
        plt.tight_layout()
        plt.savefig('demographic_variables_analysis.png')
        plt.close()
    # 3. Composición corporal
    if all(col in df.columns for col in ['Height', 'Weight', 'Gallstone Status']):
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        sns.scatterplot(data=df, x='Height', y='Weight', hue='Gallstone Status')
        plt.title('Altura vs Peso por Gallstone Status')
        plt.subplot(1, 2, 2)
        sns.boxplot(data=df, x='Gallstone Status', y='Weight')
        plt.title('Distribución de Peso por Gallstone Status')
        plt.tight_layout()
        plt.savefig('body_composition_analysis.png')
        plt.close()
    # 4. Correlaciones de composición corporal
    body_comp_vars = [col for col in ['Height', 'Weight', 'Age'] if col in df.columns]
    if len(body_comp_vars) >= 2:
        plt.figure(figsize=(10, 8))
        sns.heatmap(df[body_comp_vars].corr(), annot=True, cmap='coolwarm')
        plt.title('Correlaciones de Composición Corporal')
        plt.tight_layout()
        plt.savefig('body_composition_correlation.png')
        plt.close()
    # 5. Matriz de correlación general
    plt.figure(figsize=(12, 10))
    sns.heatmap(df.select_dtypes(include=['float64', 'int64']).corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Matriz de Correlación General')
    plt.tight_layout()
    plt.savefig('general_correlation_matrix.png')
    plt.close()

# =====================
# VISUALIZACIÓN INDIVIDUAL
# =====================
def visualizar_distribucion(df, variable):
    """
    Crea un histograma para visualizar la distribución de una variable.

    Args:
        df (pandas.DataFrame): DataFrame con los datos
        variable (str): Nombre de la columna a visualizar
    """
    if variable not in df.columns:
        print(f"La variable '{variable}' no está en el dataset.")
        return

    try:
        plt.figure(figsize=(10, 6))
        sns.histplot(data=df, x=variable, hue='Gallstone Status', kde=True,
                     common_norm=False, palette='Set2')
        plt.title(f'Distribución de {variable} por estado de cálculos biliares')
        plt.xlabel(variable)
        plt.ylabel('Frecuencia')
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Error al crear el histograma: {e}")

# =====================
# MAIN
# =====================
if __name__ == "__main__":
    # Reemplaza 'tu_archivo.xlsx' con la ruta de tu archivo
    ruta_excel = 'dataset-uci.xlsx'

    # Leer los datos
    datos = leer_excel(ruta_excel)

    # Si los datos se cargaron correctamente
    if datos is not None:
        analisis_exploratorio(datos)
        analizar_por_gallstone_status(datos)
        crear_visualizaciones(datos)
        # Visualizar algunas variables de interés
        variables_interes = ['Age', 'BMI', 'Glucose']
        for var in variables_interes:
            if var in datos.columns:
                visualizar_distribucion(datos, var)