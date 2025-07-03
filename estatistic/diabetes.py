from sklearn.datasets import load_diabetes
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

diabetes = load_diabetes(as_frame=True)
X = diabetes.data[['bmi']]
y = diabetes.target

plt.scatter(X, y, alpha=0.5)
plt.title('Relação entre IMC e Progressão de Diabetes')
plt.xlabel('IMC')
plt.ylabel('Progressão da Doença')
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()

model.fit(X_train, y_train)

beta_0 = model.intercept_
beta_1 = model.coef_[0]

print(f"Modelo: Y = {beta_0:.2f} + {beta_1:.2f} * X")

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Erro Quadrático Médio (MSE): {mse:.2f}")
print(f"Coeficiente de Determinação (R²): {r2:.2f}")

plt.scatter(X_test, y_test, alpha=0.5, label='Dados Reais')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Linha de Regressão')
plt.title('Performance do Modelo de Regressão')
plt.xlabel('IMC')
plt.ylabel('Progressão da Doença')
plt.legend()
plt.show()