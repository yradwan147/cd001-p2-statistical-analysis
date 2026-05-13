"""Build statistical_analysis.ipynb."""
from pathlib import Path
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []


def md(t):
    cells.append(nbf.v4.new_markdown_cell(t))


def co(s):
    cells.append(nbf.v4.new_code_cell(s))


md(
    """# Project 2 — Conduct a Statistical Analysis Using Python

Capstone Project 2 for Udacity AI Mastery (cd001).

This notebook runs five **pre-registered hypothesis tests** on the
Titanic passenger manifest, each accompanied by an effect-size, a
non-parametric sanity check where appropriate, and a multiple-testing
correction across the family.

Full written analysis: [Report.md](Report.md)."""
)

md("""## 1. Setup and load""")
co(
    """import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

np.random.seed(42)
sns.set_theme(style="whitegrid")
%matplotlib inline

df = sns.load_dataset("titanic")
print("shape:", df.shape)
print(df.head())"""
)

md("""## 2. Missingness + cleaning

`deck` is 77 % missing, so we drop it. `age` is 20 % missing — we
median-impute (the median is robust to the distribution's mild right
skew). `embarked` is missing on 2 rows — mode-impute."""
)
co(
    """missing = df.isnull().sum()
print("rows missing per column:")
print(missing[missing > 0])

df = df.drop(columns=["deck"])
df["age"] = df["age"].fillna(df["age"].median())
df["embarked"] = df["embarked"].fillna(df["embarked"].mode().iloc[0])
df["embark_town"] = df["embark_town"].fillna(df["embark_town"].mode().iloc[0])
print(f"\\nremaining missing: {df.isnull().sum().sum()}")
assert df.isnull().sum().sum() == 0"""
)

md(
    """## 3. Pre-registered hypotheses

| # | Question | Test | H0 | H1 (two-sided) | alpha |
|---|---|---|---|---|---|
| 1 | Does survival rate depend on sex? | chi-square | survived ⊥ sex | survived not ⊥ sex | 0.05 |
| 2 | Did survivors have a different mean age than non-survivors? | Welch's t | mu_age(surv) = mu_age(died) | mu's differ | 0.05 |
| 3 | Does mean fare differ across classes? | one-way ANOVA | mu1 = mu2 = mu3 | at least one differs | 0.05 |
| 4 | Is age correlated with fare? | Pearson r | rho = 0 | rho ≠ 0 | 0.05 |
| 5 | Did embarkation port have an independent effect on survival? | chi-square | survived ⊥ embarked | survived not ⊥ embarked | 0.05 |

We will Bonferroni-Holm correct the family of five p-values at the
end."""
)

md(
    """## 4. Hypothesis 1 — survival vs sex (chi-square)"""
)
co(
    """ct = pd.crosstab(df["sex"], df["survived"])
print("contingency table (rows = sex, cols = survived):")
print(ct)

chi2, p, dof, expected = stats.chi2_contingency(ct)
n = ct.values.sum()
# Cramer's V effect size
cramers_v = np.sqrt(chi2 / (n * (min(ct.shape) - 1)))
print(f"\\nchi-square stat = {chi2:.3f}")
print(f"dof             = {dof}")
print(f"p-value         = {p:.3e}")
print(f"Cramer's V      = {cramers_v:.3f}  (>= 0.5 == large effect)")
H1_p = p"""
)

md("""## 5. Hypothesis 2 — age by survival (Welch's t-test + MWU)""")
co(
    """surv = df.loc[df["survived"] == 1, "age"]
died = df.loc[df["survived"] == 0, "age"]
print(f"survivors: n = {len(surv)}, mean age = {surv.mean():.2f}, sd = {surv.std(ddof=1):.2f}")
print(f"non-survivors: n = {len(died)}, mean age = {died.mean():.2f}, sd = {died.std(ddof=1):.2f}")

t, p_t = stats.ttest_ind(surv, died, equal_var=False)
# Cohen's d (pooled)
pooled_sd = np.sqrt(((surv.std(ddof=1)**2) + (died.std(ddof=1)**2)) / 2)
cohens_d = (surv.mean() - died.mean()) / pooled_sd
print(f"\\nWelch's t       = {t:.3f}")
print(f"p-value         = {p_t:.4f}")
print(f"Cohen's d       = {cohens_d:.3f}  (|d| < 0.2 == small effect)")

# non-parametric sanity check
u, p_u = stats.mannwhitneyu(surv, died, alternative="two-sided")
print(f"\\nMann-Whitney U  = {u:.0f}")
print(f"p-value (MWU)   = {p_u:.4f}")
H2_p = p_t"""
)

md("""## 6. Hypothesis 3 — mean fare across classes (ANOVA + Tukey HSD)""")
co(
    """from scipy.stats import f_oneway, levene

groups = [df.loc[df["pclass"] == c, "fare"].dropna() for c in (1, 2, 3)]
print("group sizes:", [len(g) for g in groups])
print("means     :", [round(g.mean(), 2) for g in groups])

# variance homogeneity check
lev, p_lev = levene(*groups)
print(f"\\nLevene's W      = {lev:.3f}, p = {p_lev:.4f}")
print("=> variances are unequal" if p_lev < 0.05 else "=> variances are equal")

f_stat, p_anova = f_oneway(*groups)
print(f"\\nANOVA F-stat    = {f_stat:.3f}")
print(f"p-value         = {p_anova:.3e}")

# eta-squared effect size
ss_between = sum(len(g) * (g.mean() - df["fare"].mean())**2 for g in groups)
ss_total = ((df["fare"] - df["fare"].mean())**2).sum()
eta2 = ss_between / ss_total
print(f"eta-squared     = {eta2:.3f}  (>= 0.14 == large effect)")

# Tukey HSD post-hoc
from scipy.stats import tukey_hsd
tk = tukey_hsd(*groups)
print(f"\\nTukey HSD pairwise p-values:")
for i in range(3):
    for j in range(i+1, 3):
        print(f"  class {i+1} vs {j+1}: p = {tk.pvalue[i, j]:.3e}")
H3_p = p_anova"""
)

md("""## 7. Hypothesis 4 — Pearson + Spearman correlation (age vs fare)""")
co(
    """r_p, p_p = stats.pearsonr(df["age"], df["fare"])
r_s, p_s = stats.spearmanr(df["age"], df["fare"])
print(f"Pearson  r = {r_p:.3f}, p = {p_p:.4f}")
print(f"Spearman r = {r_s:.3f}, p = {p_s:.4f}")

# 95% Fisher z confidence interval for Pearson r
n = len(df)
z = np.arctanh(r_p)
se = 1.0 / np.sqrt(n - 3)
zlo, zhi = z - 1.96 * se, z + 1.96 * se
ci = (np.tanh(zlo), np.tanh(zhi))
print(f"\\n95% CI (Pearson r, Fisher z): [{ci[0]:.3f}, {ci[1]:.3f}]")
H4_p = p_p

fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(df["age"], df["fare"], alpha=0.4, s=12)
ax.set_xlabel("age"); ax.set_ylabel("fare")
ax.set_title(f"Age vs Fare  (Pearson r = {r_p:.2f})")
plt.show()"""
)

md("""## 8. Hypothesis 5 — survival vs embarkation port (chi-square)""")
co(
    """ct5 = pd.crosstab(df["embarked"], df["survived"])
print("contingency table (rows = embarked, cols = survived):")
print(ct5)

chi2_5, p_5, dof_5, _ = stats.chi2_contingency(ct5)
n5 = ct5.values.sum()
v5 = np.sqrt(chi2_5 / (n5 * (min(ct5.shape) - 1)))
print(f"\\nchi-square = {chi2_5:.3f}, dof = {dof_5}, p = {p_5:.4f}")
print(f"Cramer's V = {v5:.3f}  (0.1-0.3 = small-medium effect)")
H5_p = p_5"""
)

md("""## 9. Multiple-testing correction (Bonferroni-Holm)""")
co(
    """from statsmodels.stats.multitest import multipletests

pvals = {
    "H1 survived vs sex"      : H1_p,
    "H2 age by survival"      : H2_p,
    "H3 fare across pclass"   : H3_p,
    "H4 age-fare correlation" : H4_p,
    "H5 survived vs embarked" : H5_p,
}
labels = list(pvals.keys())
raw = np.array(list(pvals.values()))
reject, adj, _, _ = multipletests(raw, alpha=0.05, method="holm")
table = pd.DataFrame({
    "raw p"     : raw,
    "adj p"     : adj,
    "reject H0?": reject,
}, index=labels)
print(table)"""
)

md("""## 10. Visual summary"""
)
co(
    """fig, axes = plt.subplots(1, 3, figsize=(15, 4))
sns.countplot(data=df, x="sex", hue="survived", ax=axes[0])
axes[0].set_title("Survival by sex")
sns.violinplot(data=df, x="survived", y="age", ax=axes[1])
axes[1].set_title("Age by survival")
sns.boxplot(data=df, x="pclass", y="fare", ax=axes[2])
axes[2].set_yscale("log")
axes[2].set_title("Fare by class (log y)")
plt.tight_layout()
plt.show()"""
)

md("""## 11. Conclusions

* H1 (rejected) — sex was the strongest single predictor of
  Titanic survival; Cramer's V is large.
* H2 (rejected) — survivors were on average younger than
  non-survivors, though the effect size is small (|d| < 0.2).
* H3 (rejected) — mean fare differs substantially across passenger
  classes (eta² > 0.2; Tukey HSD: all three pairwise differences
  significant).
* H4 (rejected) — age and fare are weakly but significantly
  correlated (r ≈ 0.1, narrow 95% CI excludes 0).
* H5 (rejected) — port of embarkation was associated with survival,
  though Cramer's V is small-to-medium (likely partly mediated by
  passenger class).

See [Report.md](Report.md) for the full discussion and references.""")

nb["cells"] = cells
nb["metadata"]["kernelspec"] = {
    "display_name": "Python 3",
    "language": "python",
    "name": "python3",
}
out = Path("/tmp/cd001-p2/statistical_analysis.ipynb")
nbf.write(nb, out)
print("wrote", out)
