import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

n = 10
p = 0.8

dist_binom = stats.binom(n,p)

prob_7_caras = dist_binom.pmf(7)
print(f"P(X=7) em Binomial(10, 0.8): {prob_7_caras:.4f}")

prob_ate_5 = dist_binom.cdf(5)
print(f"P(X<=5) em Binomial(10, 0.8): {prob_ate_5:.4f}")

amostras_binom = dist_binom.rvs(size=1000)

k_valores = np.arange(0, n + 1)
pmf_valores = dist_binom.pmf(k_valores)
plt.bar(k_valores, pmf_valores)
plt.title('PMF da Distribuição Binomial(10, 0.8)')
plt.xlabel('Número de Sucessos (k)')
plt.ylabel('Probabilidade')
plt.show()

lambda_ = 5

dist_poisson = stats.poisson(lambda_)

prob_5_reqs = dist_poisson.pmf(5)
print(f"P(X=5) em Poissom(5): {prob_5_reqs:.4f}")

k_valores = np.arange(0, 15)
pmf_valores = dist_poisson.pmf(k_valores)
plt.bar(k_valores, pmf_valores)
plt.title('PMF da Distribuição de Poisson(5)')
plt.xlabel('Número de Eventos (k)')
plt.ylabel('Probabilidade')
plt.show()

lambda_ = 5
scale = 1 / lambda_

dist_expon = stats.expon(scale=scale)

prob_menor_01s = dist_expon.cdf(0.1)
print(f"P(T < 0.1) em Exponencial(5): {prob_menor_01s:.4f}")

x_valores = np.linspace(0, 1, 100)
pdf_valores = dist_expon.pdf(x_valores)
plt.plot(x_valores, pdf_valores)
plt.title('PDF da Distribuição Exponencial(lambda=5)')
plt.xlabel('Tempo (x)')
plt.ylabel('Densidade de Probabilidade')
plt.show()

mu = 75
sigma = 10

dist_norm = stats.norm(loc=mu, scale=sigma)

prob_menor_60 = dist_norm.cdf(60)
print(f"P(Nota < 60) em Normal(75, 10): {prob_menor_60:.4f}")

nota_top_10 = dist_norm.ppf(0.90)
print(f"Nota de corte para os 10% melhores: {nota_top_10:.2f}")

x_valores = np.linspace(40, 110, 200)
pdf_valores = dist_norm.pdf(x_valores)
plt.plot(x_valores, pdf_valores)
plt.title('PDF da Distribuição Normal(75, 10)')
plt.xlabel('Nota')
plt.ylabel('Densidade de Probabilidade')
plt.show()