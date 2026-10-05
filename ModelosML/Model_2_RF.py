# Random Forest Regressor
# para predecir el puntaje final de los estudiantes basado en sus características académicas y personales.

# 1 - Importar librerías
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# 2 - Lectura y limpieza de datos
student_file_path = 'StudentsPerformance.csv'
student_data = pd.read_csv(student_file_path)
student_data = student_data.dropna(axis=0)

# 3 - Target
y = student_data.Final_Exam_Score

# 4 - choose features
student_features = [
    'Previous_GPA',
    'Number_of_Failed_Courses',
    'Total_Credits_Earned',
    'Weekly_Study_Hours',
    'Attendance_Rate',
    'Library_Visits_Per_Month',
    'Extracurricular_Hours',
    'Sleep_Hours',
    'Social_Media_Usage_Hours',
    'Stress_Level',
    'Motivation_Score',
    'Self_Efficacy_Score',
    'Midterm_Mark'
]
X = student_data[student_features]

# 5 - Separación para entrenamiento y validación
train_X, val_X, train_y, val_y = train_test_split(
    X, y, random_state=0
)

# 6 - Entrenamiento del modelo
forest_model = RandomForestRegressor(random_state=1)
forest_model.fit(train_X, train_y)

# 7 - Predicciones
student_preds = forest_model.predict(val_X)

# 8 - Comparación y evaluación
print("Making predictions for the following 5 students:")
print(val_X.head(), "\n") 
print(mean_absolute_error(val_y, student_preds))