import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

titanic = sns.load_dataset('titanic')
print(titanic.head())
print(titanic.describe())

media_idade = titanic['age'].mean()
mediana_idade = titanic['age'].median()

print(f"Idade Média: {media_idade:.2f}")
print(f"Idade Mediana: {mediana_idade:.2f}")

std_tarifa = titanic['fare'].std()
print(f"Desvio Padrão da Tarifa: {std_tarifa:.2f}")

moda_classe = titanic['pclass'].mode()
print(f"Classe mais comum (Moda): {moda_classe}")

plt.figure(figsize=(10, 6))
sns.histplot(titanic['age'].dropna(), kde=True, bins=30)
plt.title('Distribuição de Idades dos Passageiros do Titanic')
plt.xlabel('Idade')
plt.ylabel('Frequencia')
plt.axvline(media_idade, color='red', linestyle='--', label=f'Média: {media_idade:.2f}')
plt.axvline(mediana_idade, color='green', linestyle='-', label=f'Mediana: {mediana_idade:.2f}')
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(x='pclass', y='fare', data=titanic)
plt.title('Distribuição das Tarifas por Classe do Passageiro')
plt.xlabel('Classe')
plt.ylabel('Tarifa')
plt.show()