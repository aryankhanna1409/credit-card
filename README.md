<div align="center">

<img src="assets/banner.svg" width="100%" alt="credit-card: Spending-habit segmentation and reward modelling" />

**Analysing credit-card spending habits in India: customer segmentation and reward-point modelling.**

![Python](https://img.shields.io/badge/Python-3b2416?style=flat-square&logo=python&logoColor=efdfc4)
![Jupyter](https://img.shields.io/badge/Jupyter-notebooks-8a5a22?style=flat-square&logo=jupyter&labelColor=1d120b&logoColor=efdfc4)
![scikit-learn](https://img.shields.io/badge/scikit--learn-clustering-8a5a22?style=flat-square&logo=scikitlearn&labelColor=1d120b&logoColor=efdfc4)
![XGBoost](https://img.shields.io/badge/XGBoost-regression-8a5a22?style=flat-square&labelColor=1d120b)

</div>

---

## Research

This repo holds the notebooks behind my applied-ML research on credit-card rewards, done as a research assistant at Symbiosis Institute of Technology.

The published system delivered **62.5% better reward personalisation**, as reported in the *International Journal of Computational Intelligence Systems* (2025). The notebooks here run on synthetic data; see the note below.

Related publications:

- *Analyzing and Rewarding Credit Card Spending Habits in India: A Machine Learning Approach*
- *Credit Card Hyperpersonalization and Retention Strategies Using AI*

---

## What's inside

Two notebooks, from first exploration to the full pipeline.

| Notebook | What it covers |
|---|---|
| `credit_card.ipynb` | Early experiments: classification baselines, K-Means with silhouette scores, Isolation Forest outlier removal, agglomerative clustering |
| `credit_card_2_0 (1).ipynb` | The full pipeline described below |

### The full pipeline

1. **Explore** the original transaction data (`card.csv`), clean it and handle outliers.
2. **Build a synthetic dataset** with Faker (Indian locale): 4,000 users and 20,000 transactions across Platinum, Gold, Silver and Signature cards, eight spending categories, eight cities and festival periods (Diwali, Holi, Christmas).
3. **Segment users** with K-Means, DBSCAN and Gaussian Mixture models, and test whether card type is itself the best way to segment spenders.
4. **Model reward points.** Points follow a formula of `amount × card base multiplier ÷ 100 × category multiplier × festival boost`. Random Forest and XGBoost regressors then learn to predict points from transaction features.
5. **Evaluate** with feature importance, actual-vs-predicted plots and error distributions.

### Findings

- **Amount** is the most important feature in both models (about 87% of importance in Random Forest, 54% in XGBoost), followed by whether the transaction fell in a festival period.
- Both models reproduce the reward formula almost exactly (correlation above 0.999), so an ML model can stand in for hand-written rules and adapt to new patterns.

> ⚠️ **About the data:** the original dataset has essentially one numeric feature (the amount), which is too little to model rewards. The reward analysis therefore runs on **synthetic data generated in the notebook**. The near-perfect fit shows the models can learn the reward rules, not that they predict real-world customer behaviour.

---

## Run it

```bash
git clone https://github.com/aryankhanna1409/credit-card.git
cd credit-card

pip install pandas numpy matplotlib seaborn scikit-learn xgboost lightgbm catboost faker jupyter
jupyter notebook
```

The notebooks were written for Google Colab and read data from paths such as `/content/fixed_synthetic_credit_card_data.csv`. Run the synthetic-data cell first to generate that file, and update the paths if you run locally. The original `card.csv` is not included in the repo.

---

<div align="center">

<img src="assets/footer.svg" width="100%" alt="Off the menu: from the archive" />

Applied machine learning on Indian credit-card spending

<sub>Part of <a href="https://github.com/aryankhanna1409">Aryan Khanna's portfolio</a>: one loop, five pours, from robots to AI systems.</sub>

</div>
