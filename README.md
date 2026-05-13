# Project 2 — Conduct a Statistical Analysis Using Python (cd001-p2)

Final-capstone deliverable for Udacity AI Mastery Capstone Project 2.
A full inferential-statistics walkthrough on the **Titanic** dataset:
formal hypotheses, four classes of statistical test (t-test, Mann–
Whitney U, chi-square, one-way ANOVA), correlation analysis, and an
explicit *effect-size + power* discussion alongside every p-value.

## What's in here

```
statistical_analysis.ipynb   Executable end-to-end notebook
build_notebook.py            Source of statistical_analysis.ipynb
Report.md                    Written analysis (APA citations)
requirements.txt             Pinned package versions
README.md / LICENSE / .gitignore
```

## Dataset

The Titanic passenger manifest (n = 891), via
`seaborn.load_dataset("titanic")`. Five usable categorical columns
(`survived`, `pclass`, `sex`, `embarked`, `alone`) × five numeric
columns (`age`, `sibsp`, `parch`, `fare`, `pclass`) make it ideal
for demonstrating every classical inferential test on the *same*
dataset without juggling multiple sources.

## What the notebook does

1. **Loads + audits** the dataset (missingness, dtypes, balance).
2. **Cleans** — drops `deck` (77 % missing), median-imputes `age`,
   mode-imputes `embarked`.
3. **Formal hypotheses** — five pre-registered hypotheses with
   alternative + null, alpha = 0.05, power target 0.80.
4. **Welch's t-test** — does mean `age` differ between survivors
   and non-survivors? + non-parametric Mann–Whitney U sanity check.
5. **Chi-square test of independence** — is `survived` independent
   of `sex`? + Cramér's V effect-size.
6. **One-way ANOVA + Tukey HSD post-hoc** — does mean `fare`
   differ across `pclass`? + Levene's test for variance equality.
7. **Pearson and Spearman correlation** — `fare` vs `age` and
   `fare` vs `sibsp`, with Fisher z confidence intervals.
8. **Multiple-testing correction** — Bonferroni-Holm across the
   five hypotheses.

## Environment

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook statistical_analysis.ipynb
```

CPU-only, runs in <30 seconds.

## License

MIT. Titanic dataset is in the public domain.
