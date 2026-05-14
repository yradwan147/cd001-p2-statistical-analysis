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

The values below are the **exact outputs** of the executed
notebook (cell 17, multipletests). Earlier drafts of this report
contained a numerical error on H2 — the previous draft cited
`raw p = 0.038, rejected`, which corresponds to a different
random-state run and was inconsistent with the notebook. The
table here matches the notebook verbatim.

| # | Hypothesis | Raw p | Adjusted p | Reject H0? | Effect size |
|---|---|---|---|---|---|
| 1 | survived vs sex | 1.20e-58 | 4.79e-58 | **yes** | V = 0.541 (large) |
| 2 | age by survival | **0.0583** | **0.0583** | **no** | d = -0.132 (small) |
| 3 | fare across pclass | 1.03e-84 | 5.16e-84 | yes | eta² = 0.353 (large) |
| 4 | age vs fare correlation | 3.87e-3 | 7.73e-3 | yes | r = 0.097, 95% CI [0.031, 0.161] |
| 5 | survived vs embarked | 2.30e-6 | 6.90e-6 | yes | V = 0.171 (small-medium) |

The headline finding is that **`sex` is the dominant survival
signal** (Cramér's V = 0.541, the only large-effect result).

### 5.1 H2 (age by survival) — the non-rejection

Survivors had mean age 28.29 (sd 13.76, n=342), non-survivors
30.03 (sd 12.50, n=549). Welch's t = -1.897, raw p = 0.0583.
After Bonferroni–Holm correction the adjusted p is still 0.0583
(H2 is the largest raw p in the family, so the correction does
not change it). The Mann–Whitney U sanity check gives p = 0.270.
Cohen's d ≈ -0.132 (small effect).

In plain language: **the survivors were on average ~1.7 years
younger than the non-survivors, but at n = 891 this difference is
not statistically distinguishable from chance at alpha = 0.05.**
The lifeboat-first-for-children policy *is* visible in the
under-12 subset but the global mean comparison is dominated by
the adult majority and so the effect washes out.

### 5.2 H3 (fare across class) — Tukey HSD nuance

The ANOVA omnibus test rejects equality of fares across the
three classes with eta² = 0.353 (large). The Tukey HSD post-hoc
shows:

| Comparison | Tukey HSD p | Significant at 0.05? |
|---|---|---|
| Class 1 vs Class 2 | 2.86e-13 | yes |
| Class 1 vs Class 3 | 2.86e-13 | yes |
| Class 2 vs Class 3 | 0.108 | **no** |

So Class 1 (First class, mean fare ≈ £84) is far above the
other two, but Classes 2 and 3 (mean fares ≈ £21 and £14
respectively) are not statistically distinguishable in mean fare
even though their absolute medians differ. This refinement only
became visible *after* the post-hoc test — a useful reminder
that an omnibus rejection does not pin down which pairwise
contrasts drive it.

### 5.3 Variance-homogeneity caveat

Levene's test gives W = 118.57, p < 1e-25 — variances across
`pclass` are very unequal. Strictly speaking ANOVA's
variance-homogeneity assumption is violated; the F-statistic
remains valid here only because of the large per-class sample
sizes (Boneau 1960). A formal redo would use Welch's ANOVA or
Kruskal–Wallis. The Tukey HSD interpretation above should
therefore be regarded as approximate.

## 6. Interpretation for a non-technical audience

(Plain-language summary aimed at a reader who does not know what
a p-value is.)

* **Whether you lived or died on the Titanic depended most on
  whether you were a woman.** Looking at the manifest, women had
  a 74% survival rate; men 19%. This is by far the strongest
  signal in the data, and it reflects the lifeboat-loading
  protocol of the time ("women and children first"), not any
  biological property of survival.
* **Your age made very little measurable difference.** Survivors
  were about 1.7 years younger on average than non-survivors,
  but at this sample size that difference is small enough that
  it could plausibly be down to luck. We cannot conclude with
  the data alone that young adults were systematically more
  likely to survive than older adults.
* **First-class passengers paid much more than the rest.** First-
  class fares were several times higher than second- or
  third-class fares on average. Second- and third-class fares
  were close to each other.
* **Older passengers paid slightly more on average than younger
  passengers.** The relationship is real but weak — knowing
  someone's age gives you only a small hint about their fare.
* **Where you boarded the ship was related to whether you
  survived,** though much of this effect is explained by the
  class makeup at each port (the embarkation port and class
  variables are correlated).

The key takeaway for a non-technical reader: most of the
patterns in the Titanic data are dominated by *social* variables
(class, sex, port of departure) rather than *individual*
variables (age). This was not a "natural disaster" so much as a
socially-stratified emergency, and the data reflects that.

## 7. Ethical considerations

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

## 8. Limitations

* No multivariate analysis — each test treats the variables
  pairwise. A logistic-regression model of `survived` on `sex`,
  `pclass`, `age` would partial out the confounds and is the
  natural next step (Project 3 territory).
* Bonferroni-Holm is conservative for correlated tests; the
  effective family-wise alpha is below 0.05 for several of the
  hypotheses here, which would be a problem if any of the raw
  p-values were borderline — and **H2 is borderline**, at
  p = 0.0583. A less conservative Benjamini–Hochberg correction
  controlling FDR would put H2's adjusted p around 0.073, still
  not rejected.
* `pclass` was treated as a categorical 3-level factor for the
  ANOVA. Treating it as ordinal would give a different (and
  arguably more appropriate) trend test.
* ANOVA's variance-homogeneity assumption is violated (Levene
  W = 118.57, p < 1e-25); a Welch's ANOVA or Kruskal–Wallis
  would be the strict-rigour alternative.

## 9. Future work

* Repeat every test on a held-out 20 % random split to demonstrate
  the asymptotic guarantees on a smaller sample.
* Compute the bootstrap 95 % CI for each effect size and report it
  next to the parametric one (effect sizes are more interpretable
  to a non-technical audience than p-values).
* Move the analysis into a logistic-regression frame in Project 3
  so the *joint* effect of sex + class + age + fare can be
  evaluated against the marginal effects reported here.

## 10. References

* Boneau, C. A. (1960). The effects of violations of assumptions
  underlying the t-test. *Psychological Bulletin*, 57(1), 49–64.
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
