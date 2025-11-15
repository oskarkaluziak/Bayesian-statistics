
import pandas as pd
from fractions import Fraction
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

file_name = 'gnra.csv'
df = pd.read_csv(file_name)
df.info()
print(df.head())
total_count = len(df)
print(total_count)

"""**3.0**"""

# P(gnra)
# sum (if grna = true)
gnra_count = df['GNRA'].sum()

# ułamek dla grna
rob_gnra = Fraction(gnra_count, total_count)

print(f"Liczba łańcuchów z motywem GNRA: {gnra_count}")
print(f"P(GNRA) = {gnra_count} / {total_count} = {prob_gnra}")
print("-" * 45)

# P(ribosomal)
# if 'description' contains 'ribosomal'
ribosomal_count = df['Description'].str.contains('ribosomal', case=False, na=False).sum()

# ułamek dla ribosomal
prob_ribosomal = Fraction(ribosomal_count, total_count)

print(f"Liczba łańcuchów z 'ribosomal' w opisie: {ribosomal_count}")
print(f"P(Ribosomal) = {ribosomal_count} / {total_count} = {prob_ribosomal}")
print("-" * 45)

# P(riboswitch)
# if 'description' contains 'riboswitch'
riboswitch_count = df['Description'].str.contains('riboswitch', case=False, na=False).sum()

# ułamek dla riboswitch
prob_riboswitch = Fraction(riboswitch_count, total_count)

print(f"Liczba łańcuchów z 'riboswitch' w opisie: {riboswitch_count}")
print(f"P(Riboswitch) = {riboswitch_count} / {total_count} = {prob_riboswitch}")

"""**3.5**"""

# stworzenie serii boolean dla Ribosomal
is_ribosomal = df['Description'].str.contains('ribosomal', case=False, na=False)
is_ribosomal.name = 'Ribosomal'

# stworzenie serii boolean dla Riboswitch
is_riboswitch = df['Description'].str.contains('riboswitch', case=False, na=False)
is_riboswitch.name = 'Riboswitch'

# GNRA już jest typu boolean, więc jest ready to use

# wyswietlenie liczby wystąpień dla każdej kombinacji
counts_ribosomal = pd.crosstab(df['GNRA'], is_ribosomal)
print("Macierz zliczeń")
print(counts_ribosomal)


# wyswietlenie wspolnego rozkladu prawdopodobienstwa
prob_ribosomal = pd.crosstab(df['GNRA'], is_ribosomal, normalize='all')
print("\nMacierz prawdopodobieństwa")
print(prob_ribosomal)

# wyswietlenie liczby wystąpień dla każdej kombinacji
counts_riboswitch = pd.crosstab(df['GNRA'], is_riboswitch)
print("\nMacierz zliczeń")
print(counts_riboswitch)

# wyswietlenie wspolnego rozkladu prawdopodobienstwa
prob_riboswitch = pd.crosstab(df['GNRA'], is_riboswitch, normalize='all')
print("\nMacierz prawdopodobieństwa")
print(prob_riboswitch)

"""**4.0**"""

def safe_divide(numerator, denominator):
    if denominator == 0:
        return np.nan
    else:
        return numerator / denominator

def are_results_identical(a, b):
    # czy obie są NaN
    if np.isnan(a) and np.isnan(b):
        return True
    # czy są numerycznie bliskie (i nie są NaN)
    try:
        return np.isclose(a, b)
    except TypeError:
        # jeden z nich jest NaN, a drugi nie
        return False

is_gnra = df['GNRA']
is_ribosomal = df['Description'].str.contains('ribosomal', case=False, na=False)
is_riboswitch = df['Description'].str.contains('riboswitch', case=False, na=False)

p_gnra = is_gnra.mean()
p_ribosomal = is_ribosomal.mean()
p_riboswitch = is_riboswitch.mean()
p_gnra_and_ribosomal = (is_gnra & is_ribosomal).mean()
p_gnra_and_riboswitch = (is_gnra & is_riboswitch).mean()

print(f"P(GNRA)       = {p_gnra:.6f}")
print(f"P(Ribosomal)  = {p_ribosomal:.6f}")
print(f"P(Riboswitch) = {p_riboswitch:.6f}")
print(f"P(GNRA ∩ Ribosomal)  = {p_gnra_and_ribosomal:.6f}")
print(f"P(GNRA ∩ Riboswitch) = {p_gnra_and_riboswitch:.6f}")

print("\nobliczenie klasyczne")

p_gnra_given_ribosomal = safe_divide(p_gnra_and_ribosomal, p_ribosomal)
print(f"P(GNRA|Ribosomal) = P(A∩B) / P(B) = {p_gnra_given_ribosomal:.6f}")

p_ribosomal_given_gnra = safe_divide(p_gnra_and_ribosomal, p_gnra)
print(f"P(Ribosomal|GNRA) = P(A∩B) / P(A) = {p_ribosomal_given_gnra:.6f}")

print("\nobliczenie z wykorzystaniem tw. bayesa")

bayes_g_given_r = safe_divide(p_ribosomal_given_gnra * p_gnra, p_ribosomal)
print(f"P(GNRA|Ribosomal) = ( P(B|A) * P(A) ) / P(B) = {bayes_g_given_r:.6f}")

bayes_r_given_g = safe_divide(p_gnra_given_ribosomal * p_ribosomal, p_gnra)
print(f"P(Ribosomal|GNRA) = ( P(A|B) * P(B) ) / P(A) = {bayes_r_given_g:.6f}")

print(f"wyniki dla P(GNRA|Ribosomal) są identyczne: {are_results_identical(p_gnra_given_ribosomal, bayes_g_given_r)}")
print(f"wyniki dla P(Ribosomal|GNRA) są identyczne: {are_results_identical(p_ribosomal_given_gnra, bayes_r_given_g)}")

print("\nobliczenie klasyczne")

p_gnra_given_riboswitch = safe_divide(p_gnra_and_riboswitch, p_riboswitch)
print(f"P(GNRA|Riboswitch) = P(A∩B) / P(B) = {p_gnra_given_riboswitch:.6f}")

p_riboswitch_given_gnra = safe_divide(p_gnra_and_riboswitch, p_gnra)
print(f"P(Riboswitch|GNRA) = P(A∩B) / P(A) = {p_riboswitch_given_gnra:.6f}")

print("\nobliczenie z wykorzystaniem tw. bayesa")

bayes_g_given_s = safe_divide(p_riboswitch_given_gnra * p_gnra, p_riboswitch)
print(f"P(GNRA|Riboswitch) = ( P(B|A) * P(A) ) / P(B) = {bayes_g_given_s:.6f}")

bayes_s_given_g = safe_divide(p_gnra_given_riboswitch * p_riboswitch, p_gnra)
print(f"P(Riboswitch|GNRA) = ( P(A|B) * P(B) ) / P(A) = {bayes_s_given_g:.6f}")

print(f"wyniki dla P(GNRA|Riboswitch) są identyczne: {are_results_identical(p_gnra_given_riboswitch, bayes_g_given_s)}")
print(f"wyniki dla P(Riboswitch|GNRA) są identyczne: {are_results_identical(p_riboswitch_given_gnra, bayes_s_given_g)}")

"""**4.5**"""

bf = bayes_g_given_r/bayes_g_given_s
bf

"""**Zinterpretuj wyniki. Czy obecność motywu GNRA silniej wskazuje na to, że struktura jest rybosomem, czy ryboprzełącznikiem?**
P(Ribosomal|GNRA)  = 0.078313

P(Riboswitch|GNRA) = 0.168675

P(Riboswitch|GNRA) > P(Ribosomal|GNRA).
Obecność motywu GNRA silniej wskazuje na to, że struktura jest ryboprzełącznikiem.

**Traktując motyw GNRA jako obserwację/dane, a strukturę rybosomu lub ryboprzełącznika jako dwie alternatywne hipotezy, oblicz współczynnik Bayesa (Bayes Factor) i skomentuj jego wartość.**

  Prawdopodobieństwo P(GNRA|Ribosomal) [P(D|H1)] = 0.590909

Prawdopodobieństwo P(GNRA|Riboswitch) [P(D|H2)] = 0.466667

Współczynnik Bayesa (BF) = 1.2662 ≈ 1.3

Dane (GNRA) dostarczają słabych ale pozytywnych dowodów
   na korzyść hipotezy Rybosomu (H1) wobec hipotezy Ryboprzełącznika (H2).


Ta nierówność wynika z tego, że ryboprzełączników jest więcej niż rybosomów i zajmują większość wystąpień GNRA, ale GNRA zajmuje tylko niewielką część ryboprzełączników, za to w rybosomach ma zdecydowanie większy udział.

**5.0**
"""

def safe_divide(numerator, denominator):
    if denominator == 0 or np.isnan(denominator) or denominator is None:
        return np.nan
    if np.isnan(numerator) or numerator is None:
         return np.nan
    return float(numerator) / float(denominator)

n_samples = 10000

is_gnra = df['GNRA']
is_ribosomal = df['Description'].str.contains('ribosomal', case=False, na=False)
is_riboswitch = df['Description'].str.contains('riboswitch', case=False, na=False)

# parametry dla rozkladow beta

# H1: Ribosomal
# sukcesy = (Ribosomal ORAZ GNRA)
k_ribo_gnra = (is_ribosomal & is_gnra).sum()
# porażki = (Ribosomal ORAZ NIE GNRA)
k_ribo_not_gnra = (is_ribosomal & ~is_gnra).sum()

# H2: Riboswitch
# "sukcesy" = (Riboswitch ORAZ GNRA)
k_switch_gnra = (is_riboswitch & is_gnra).sum()
# "porażki" = (Riboswitch ORAZ NIE GNRA)
k_switch_not_gnra = (is_riboswitch & ~is_gnra).sum()

# H1: P(GNRA|Ribosomal)
alpha_1 = k_ribo_gnra + 1
beta_1 = k_ribo_not_gnra + 1

# H2: P(GNRA|Riboswitch)
alpha_2 = k_switch_gnra + 1
beta_2 = k_switch_not_gnra + 1

print("\nParametry Rozkładów Beta")
print(f"H1 (Ribosomal): Beta(α={alpha_1}, β={beta_1})")
print(f"  (Na podstawie {k_ribo_gnra} sukcesów i {k_ribo_not_gnra} porażek)")
print(f"H2 (Riboswitch): Beta(α={alpha_2}, β={beta_2})")
print(f"  (Na podstawie {k_switch_gnra} sukcesów i {k_switch_not_gnra} porażek)")

# punktowy bf dla porównania
p_h1_point = k_ribo_gnra / (k_ribo_gnra + k_ribo_not_gnra)
p_h2_point = k_switch_gnra / (k_switch_gnra + k_switch_not_gnra)
bf_point_estimate = safe_divide(p_h1_point, p_h2_point)

print("\nWcześniej obliczony 'punktowy' BF")
print(f"P(D|H1) = {p_h1_point:.6f}")
print(f"P(D|H2) = {p_h2_point:.6f}")
print(f"Współczynnik Bayesa (punktowy) = {bf_point_estimate:.4f}")

# symulacja mmmonte carlitos
print(f"\nmmmonte carlitos dla {n_samples} losowań...")

# losowanie z rozkładów Beta
samples_h1 = beta.rvs(alpha_1, beta_1, size=n_samples)
samples_h2 = beta.rvs(alpha_2, beta_2, size=n_samples)

# obliczenie współczynników Bayesa dla każdej pary
bayes_factors = samples_h1 / (samples_h2 + np.finfo(float).eps)
print (bayes_factors)

# oczyszczenie wyników z ewentualnych nieskończoności
bf_clean = bayes_factors[np.isfinite(bayes_factors)]
print(f"wygenerowano i oczyszczono {len(bf_clean)} próbek BF")

# analiza rozkladu bf
mean_bf = np.mean(bf_clean)
median_bf = np.median(bf_clean)
# obliczenie 95% przedziału ufności
ci_low = np.percentile(bf_clean, 2.5)
ci_high = np.percentile(bf_clean, 97.5)

print("\nWyniki mmmonte carlitosa:")
print(f"Średnia:   {mean_bf:.4f}")
print(f"Mediana:   {median_bf:.4f}")
print(f"95% Przedział Ufności (CI): [{ci_low:.4f}, {ci_high:.4f}]")

is_inside = (bf_point_estimate >= ci_low) and (bf_point_estimate <= ci_high)
print(f"Wcześniej obliczona wartość BF ({bf_point_estimate:.4f}) ...")

plt.figure(figsize=(10, 6))
x = np.linspace(0, 1, 500)
plt.plot(x, beta.pdf(x, alpha_1, beta_1),
         label=f'P(GNRA|Ribosomal)\nBeta(α={alpha_1}, β={beta_1})',
         linewidth=2)
plt.plot(x, beta.pdf(x, alpha_2, beta_2),
         label=f'P(GNRA|Riboswitch)\nBeta(α={alpha_2}, β={beta_2})',
         linewidth=2, linestyle='--')
plt.title('Rozkłady Beta dla P(Dane | Hipoteza)', fontsize=16)
plt.xlabel('Prawdopodobieństwo P(GNRA|H)', fontsize=12)
plt.ylabel('Gęstość prawdopodobieństwa (PDF)', fontsize=12)
plt.legend()
plt.grid(True, linestyle=':', alpha=0.7)

plt.figure(figsize=(10, 6))
plt.hist(bf_clean, bins=100, alpha=0.75,
         label='Rozkład BF z symulacji', log=True)

xlim_upper = np.percentile(bf_clean, 99.9)
plt.xlim(0, xlim_upper)

plt.axvline(median_bf, color='red', linestyle='--', linewidth=2,
            label=f'Mediana BF = {median_bf:.3f}')
plt.axvline(bf_point_estimate, color='black', linestyle=':', linewidth=2,
            label=f'Punktowy BF = {bf_point_estimate:.3f}')
plt.axvline(ci_low, color='orange', linestyle=':', linewidth=2,
            label='95% CI Dolny')
plt.axvline(ci_high, color='orange', linestyle=':', linewidth=2,
            label='95% CI Górny')

plt.title('Rozkład Współczynników Bayesa (z symulacji MC)', fontsize=16)
plt.xlabel('Współczynnik Bayesa (H1_Ribo / H2_Switch)', fontsize=12)
plt.ylabel('Częstość (skala log)', fontsize=12)
plt.legend()
plt.grid(True, linestyle=':', alpha=0.7)

"""Wyniki symulacji Monte Carlo są zgodne z obliczoną wcześniej wartością BF. Różnica między średnią w symulacji a BF wyniosła 0.0036 (1.2698−1.2662), a między BF a medianą 0.0183 (1.2662−1.2479). Natomiast zakres 95% ufności (CI: [0.7820,1.8823]) jest dość szeroki."""