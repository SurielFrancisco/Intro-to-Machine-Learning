# Pipeline
# para predecir el puntaje final de los estudiantes
# utilizando características académicas y personales.

# 1 - Importar librerías
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


# 2 - Lectura de datos
student_file_path = 'StudentsPerformance.csv'
student_data = pd.read_csv(student_file_path)


# 3 - Target y predictores
y = student_data.Final_Exam_Score
X = student_data.drop(['Final_Exam_Score'], axis=1)


# 4 - Separación para entrenamiento y validación
X_train_full, X_valid_full, y_train, y_valid = train_test_split(
    X,
    y,
    train_size=0.8,
    test_size=0.2,
    random_state=0
)


# 5 - Selección de columnas categóricas
categorical_cols = [
    cname
    for cname in X_train_full.columns
    if X_train_full[cname].nunique() < 10
    and X_train_full[cname].dtype == "object"
]


# 6 - Selección de columnas numéricas
numerical_cols = [
    cname
    for cname in X_train_full.columns
    if X_train_full[cname].dtype in ['int64', 'float64']
]


# 7 - Mantener únicamente las columnas seleccionadas
my_cols = categorical_cols + numerical_cols

X_train = X_train_full[my_cols].copy()
X_valid = X_valid_full[my_cols].copy()


# 8 - Preprocesamiento de datos numéricos
numerical_transformer = SimpleImputer(strategy='constant')


# 9 - Preprocesamiento de datos categóricos
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])


# 10 - Unir los preprocesamientos
processor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ]
)


# 11 - Creación del modelo
model = RandomForestRegressor(
    n_estimators=100,
    random_state=0
)


# 12 - Crear el pipeline
my_pipeline = Pipeline(steps=[
    ('preprocessor', processor),
    ('model', model)
])


# 13 - Entrenamiento del modelo
my_pipeline.fit(X_train, y_train)


# 14 - Predicciones
student_preds = my_pipeline.predict(X_valid)


# 15 - Comparación y evaluación
print("Making predictions for the following 5 students:")
print(X_valid.head(), "\n")

print("The predictions are:")
print(student_preds[:5])

print("\nReal values:")
print(y_valid.head().values)

print("\nAverage absolute error:",
      mean_absolute_error(y_valid, student_preds))