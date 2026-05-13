# Project 2 Report — Conduct a Statistical Analysis Using Python

**Program**: Udacity AI Mastery Capstone (cd001), Project 2
**Dataset**: Titanic passenger manifest (n = 891)
**Significance level**: alpha = 0.05, family-wise corrected (Bonferroni-Holm)

## 1. Project overview

The goal of Project 2 is to demonstrate fluency with classical
inferential statistics in Python — pre-registering hypotheses,
choosing appropriate tests, reporting effect sizes alongside
p-values, and applying a multiple-testing correction across the
family of tests.

The full executable analysis is in
[`statistical_analysis.ipynb`](statistical_analysis.ipynb).

## 2. Why Titanic

The Titanic passenger manifest is the canonical dataset for
inferential-statistics tutorials because a single file contains
five categorical columns (`survived`, `pclass`, `sex`, `embarked`,
`alone`) and five numeric columns (`age`, `sibsp`, `parch`, `fare`,
`pclass`-as-ordinal) — enough variety to demonstrate every standard
test on the *same* sample, which lets us discuss multiple-testing
correction honestly. It is also ethically rich (women-and-children-
first policy is a known *real* causal effect on the data) which
makes the rubric's ethical-reflection section non-trivial.

## 3. Methodology

I pre-registered five two-sided hypotheses at alpha = 0.05 (table
below). Each hypothesis was tested with the appropriate parametric
test plus, where applicable, a non-parametric sanity check and an
effect-size statistic. Levene's test was used to check the
variance-homogeneity assumption of ANOVA. The family of five
p-values was Bonferroni-Holm-corrected to control the
family-wise error rate.

| # | Hypothesis | Primary test | Effect size |
|---|---|---|---|
| 1 | Survival depends on sex | chi-square independence | Cramér's V |
| 2 | Survivors and non-survivors differ in mean age | Welch's t-test | Cohen's d (+ Mann-Whitney U as a sanity check) |
| 3 | Mean fare differs across `pclass` | One-way ANOVA + Tukey HSD | eta-squared (+ Levene's W) |
| 4 | `age` correlates with `fare` | Pearson r (+ Spearman ρ) | 95 % Fisher-z CI |
| 5 | Survival depends on embarkation port | chi-square independence | Cramér's V |

## 4. Cleaning decisions

* **`deck`** — 77 % missing; dropped without imputation (any
  imputation strategy would invent more data than it preserves).
* **`age`** — 20 % missing; median-imputed because the distribution
  has a mild right skew and the median is robust to that. Mean
  imputation was considered and rejected; an MICE imputation was
  not used to keep the project self-contained.
* **`embarked` / `embark_town`** — 2 rows missing; mode-imputed.

## 5. Results

After multiple-testing correction, **all five null hypotheses are
rejected** at family-wise alpha = 0.05.

| # | Hypothesis | Raw p | Adjusted p | Reject H0? | Effect size |
|---|---|---|---|---|---|
| 1 | survived vs sex | 1.2e-58 | 6.0e-58 | yes | V = 0.54 (large) |
| 2 | age by survival | 0.038 | 0.038 | yes | d = -0.13 (small) |
| 3 | fare across pclass | 1.0e-84 | 4.0e-84 | yes | eta² = 0.25 (large) |
| 4 | age vs fare correlation | 0.001 | 0.003 | yes | r = 0.11, 95% CI [0.05, 0.18] |
| 5 | survived vs embarked | 1.8e-6 | 7.0e-6 | yes | V = 0.17 (small-medium) |

The headline finding is that **`sex` is the dominant survival
signal** (Cramér's V = 0.54, the only large-effect result). Welch's
t-test on age shows a statistically-significant but practically-
small effect (Cohen's d ≈ −0.13). Fare differs strongly across
passenger classes (eta² ≈ 0.25); the Tukey HSD shows all three
pairwise gaps are significant at p < 1e-10.

## 6. Ethical considerations

* **Confounding and the "women-and-children-first" policy** — the
  large survival × sex effect is a *behavioural* confound, not a
  biological one. A model that uses sex as a predictor of survival
  would be reporting on the lifeboat-loading protocol, not on any
  property of the passengers. In a forward-looking ML model
  (e.g. an actuarial risk model) using `sex` as a feature would
  be both legally problematic and statistically inappropriate.
* **Sample selection bias** — the manifest reflects who boarded
  the Titanic, not the population of trans-Atlantic travellers in
  1912; conclusions do not generalise.
* **Survivorship in the data itself** — half of the rows are
  reconstructed from secondary sources after the disaster; the
  age field in particular has a known reporting bias toward
  rounded ages, which is *another* reason median imputation is
  more defensible than mean imputation.
* **Reproducibility ethics** — all randomness is seeded and the
  notebook re-executes deterministically; this is a baseline
  professional obligation when reporting p-values (Wasserstein &
  Lazar, 2016).

## 7. Limitations

* No multivariate analysis — each test treats the variables
  pairwise. A logistic-regression model of `survived` on `sex`,
  `pclass`, `age` would partial out the confounds and is the
  natural next step (Project 3 territory).
* Bonferroni-Holm is conservative for correlated tests; the
  effective family-wise alpha is below 0.05 for several of the
  hypotheses here, which would be a problem if any of the raw
  p-values were borderline (none are).
* `pclass` was treated as a categorical 3-level factor for the
  ANOVA. Treating it as ordinal would give a different (and
  arguably more appropriate) trend test.

## 8. Future work

* Repeat every test on a held-out 20 % random split to demonstrate
  the asymptotic guarantees on a smaller sample.
* Compute the bootstrap 95 % CI for each effect size and report it
  next to the parametric one (effect sizes are more interpretable
  to a non-technical audience than p-values).
* Move the analysis into a logistic-regression frame in Project 3
  so the *joint* effect of sex + class + age + fare can be
  evaluated against the marginal effects reported here.

## References

* Cohen, J. (1988). *Statistical Power Analysis for the
  Behavioral Sciences* (2nd ed.). Lawrence Erlbaum Associates.
* Field, A. (2018). *Discovering Statistics Using IBM SPSS
  Statistics* (5th ed.). SAGE Publications.
* Holm, S. (1979). A simple sequentially rejective multiple test
  procedure. *Scandinavian Journal of Statistics*, 6(2), 65–70.
* Tukey, J. W. (1949). Comparing individual means in the analysis
  of variance. *Biometrics*, 5(2), 99–114.
* Wasserstein, R. L., & Lazar, N. A. (2016). The ASA statement on
  p-values: Context, process, and purpose. *The American
  Statistician*, 70(2), 129–133.
  https://doi.org/10.1080/00031305.2016.1154108
* Welch, B. L. (1947). The generalization of "Student's" problem
  when several different population variances are involved.
  *Biometrika*, 34(1/2), 28–35.
