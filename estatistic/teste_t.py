import numpy as np
from scipy import stats

np.random.seed(42)
grupo_a = np.random.normal(loc=30, scale=8, size=50)
grupo_b = np.random.normal(loc=34, scale=9, size=55)

shapiro_a = stats.shapiro(grupo_a)
shapiro_b = stats.shapiro(grupo_b)
print(f"Shapiro-Wilk Grupo A: p-valor = {shapiro_a.pvalue:.3f}")
print(f"Shapiro-Wilk Grupo B: p-valor = {shapiro_b.pvalue:.3f}")

levene_test = stats.levene(grupo_a, grupo_b)
print(f"Levene Test: p-valor = {levene_test.pvalue:.3f}")

ttest_result = stats.ttest_ind(grupo_a, grupo_b, equal_var=True)

print(f"Ëstatística t: {ttest_result.statistic:.3f}")
print(f"P-valor: {ttest_result.pvalue:.3f}")