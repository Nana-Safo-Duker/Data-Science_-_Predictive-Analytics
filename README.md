# Data Science & Predictive Analytics

This repository curates ten end-to-end analytics projects that span marketing, customer intelligence, cybersecurity, NLP, HR analytics, financial crime, energy, classical machine learning education, compensation science, and venture insights. Every project follows the same blueprint—clean, analyze, model, and communicate—so you can jump into any folder and reproduce the workflow with minimal ramp-up.

## Table of Contents

- [About](#about)
- [Repository Highlights](#repository-highlights)
- [Project Catalog](#project-catalog)
- [Project Summaries](#project-summaries)
- [Getting Started](#getting-started)
- [Repository Structure](#repository-structure)
- [Contributing & Support](#contributing--support)
- [License & Data Ethics](#license--data-ethics)

## About
**Data Science & Predictive Analytics** is curated by **Nana Safo Duker** as a demonstrable playground of repeatable analytics blueprints.  

- 🌐 **Website / Portfolio:** [https://nana-safo-duker.github.io/](https://nana-safo-duker.github.io/)  
- 🎯 **Focus:** Transform real datasets into production-ready insights via reproducible EDA, statistical inference, and predictive modeling workflows in both Python and R.  
- 🧰 **Tooling Philosophy:** Opinionated project scaffolding (data → notebooks → scripts → results) that makes it easy to swap datasets, extend models, or port deliverables into client-facing decks.  
- 🤝 **Intended Audience:** Analysts, ML engineers, and educators looking for ready-made exemplars covering marketing, operations, cybersecurity, finance, HR, NLP, energy, and startup analytics.  
- 📈 **Outcomes:** Clear documentation, validated metrics, and visual assets so stakeholders can trace insight provenance from raw CSVs to final recommendations.

## Repository Highlights
- Ten standalone case studies, each with datasets, dual-language notebooks, production-ready scripts, and reporting assets.
- Consistent directory structure (`data/`, `notebooks/`, `scripts/`, `results/`, `docs/`) so analysts can re-use tooling across domains.
- Emphasis on statistical depth (descriptive, inferential, hypothesis testing) and predictive rigor (ML pipelines, evaluation, explainability).
- Visual assets (plots, dashboards-ready figures) and compliance documentation (project summaries, verification reports) included where appropriate.

## Project Catalog

| Folder | Domain & Question | Key Assets | Tech Highlights |
| --- | --- | --- | --- |
| `Consumer Purchase Prediction using Machine Learning` | Who is most likely to buy after seeing an ad? | Advertisement dataset, staged notebooks/scripts, compliance summaries | Logistic regression, tree ensembles, balanced evaluation, Python & R parity |
| `Customer Data Analysis and Insights Project` | How do geography and customer attributes drive segmentation? | Customer CSV, clustering notebooks, reporting assets | PCA, K-Means/Hierarchical clustering, statistical testing |
| `Cybersecurity Threat Detection and Attack Pattern Analysis` | Which traffic patterns indicate attacks in progress? | 178k-row attack log, EDA/ML notebooks, automation scripts | Gradient boosting, time-based features, threat taxonomy dashboards |
| `Email Spam Detection_Data Science Project` | Can we flag spam with interpretable NLP features? | Raw & cleaned corpora, TF-IDF/NLP workflows, evaluation reports | Text preprocessing, Naive Bayes vs SVM vs XGBoost, ROC/PR analysis |
| `Employee Performance and Retention Data Analysis` | What signals foretell retention risk and performance trends? | Raw + processed HR data, statistical tables, visualization suite | Cohort hiring trends, ANOVA/t-tests, tree-based attrition modeling |
| `Financial Fraud Detection using Predictive Analytics` | Which transactions are anomalous or fraudulent? | Kaggle-style fraud data, figure gallery, dual-language notebooks | Imbalanced learning, anomaly detection, SHAP/feature-importance views |
| `Fuel Consumption and Efficiency Analysis Project` | How do specs influence fuel use & emissions? | Canadian fuel dataset, regression workflows, verification pack | Multi-model regression, temporal trend analysis, policy-ready visuals |
| `Iris Flower Classification_Machine Learning Analysis` | Classic iris benchmark with modern pedagogy | Cleaned Iris CSV, six Python & R notebooks, reusable scripts | EDA → ML curriculum, model comparison dashboards, documentation |
| `Salary Prediction_Data Science and Predictive Modeling` | How does seniority translate to compensation? | Position-salary data, processed features, model artifacts | Polynomial regression, SVR, feature engineering for career ladders |
| `Unicorn Companies Growth and Investment Data Analysis` | What drives valuations & funding velocity? | Unicorn CSV, docs, notebooks, plots/models outputs | Valuation regression, stage classification, investor-network insights |

## Project Summaries

### Consumer Purchase Prediction using Machine Learning
- **Dataset:** Advertisement.csv with demographics, salary, and binary purchase labels.
- **Workflow:** Full EDA, statistical testing, univariate/bivariate/multivariate notebooks, ML scripts for Python & R, plus compliance/requirements reports.
- **Models & Metrics:** Logistic Regression, Random Forest, SVM, Naive Bayes, Gradient Boosting evaluated via accuracy, precision, recall, F1, and ROC-AUC.

### Customer Data Analysis and Insights Project
- **Dataset:** Customers.csv capturing geography and company metadata.
- **Highlights:** Segmentation-ready EDA, statistical notebooks, PCA + clustering pipelines (K-Means, hierarchical), and ready-to-publish figures.
- **Use Cases:** Territory planning, market expansion, customer success prioritization.

### Cybersecurity Threat Detection and Attack Pattern Analysis
- **Dataset:** 178k labeled attack events with IPs, ports, protocols, and timestamps.
- **Highlights:** Automated notebook generation scripts, end-to-end ML in Python/R, temporal and protocol analysis, advanced classifiers (XGBoost, LightGBM).
- **Deliverables:** Threat heatmaps, attack timelines, feature-importance plots for SOC teams.

### Email Spam Detection_Data Science Project
- **Dataset:** 5,726 emails (ham vs spam) with raw and cleaned text corpora.
- **Highlights:** NLP preprocessing (tokenization, stop-word removal, TF-IDF), multi-model benchmark (Naive Bayes, SVM, Random Forest, XGBoost), evaluation reports.
- **Deliverables:** Models, visualization-ready figures, requirements/licensing guidance.

### Employee Performance and Retention Data Analysis
- **Dataset:** HR table with demographics, compensation, tenure, and management indicators.
- **Highlights:** Dual-language notebooks/scripts, hypothesis testing (t-tests, chi-square, ANOVA), retention risk modeling, comprehensive plot library.
- **Deliverables:** Cleaned datasets, statistical CSV outputs, automation scripts for reporting.

### Financial Fraud Detection using Predictive Analytics
- **Dataset:** High-dimensional transaction log (card, address, device, engineered features).
- **Highlights:** Heavy-duty preprocessing, imbalance strategies (class weights/SMOTE-ready), ensemble classifiers (XGBoost/LightGBM/RandomForest), explainability tooling.
- **Deliverables:** Saved models, visual diagnostics (correlation, target distribution, time trends), reporting stubs.

### Fuel Consumption and Efficiency Analysis Project
- **Dataset:** FuelConsumption.csv with specs, consumption metrics, and CO₂ emissions.
- **Highlights:** Statistical & ML notebooks in Python/R, regression comparisons (Linear, Random Forest, Gradient Boosting), policy-grade dashboards.
- **Deliverables:** Figures folder, trained regressors, verification checklist.

### Iris Flower Classification_Machine Learning Analysis
- **Dataset:** Classic 150-row Iris dataset with four measurements per species.
- **Highlights:** Six-step notebook curriculum (EDA → ML) for both Python and R, modular scripts, results folder for figures/tables, polished README.
- **Use Cases:** Introductory ML instruction and tooling templates.

### Salary Prediction_Data Science and Predictive Modeling
- **Dataset:** Position vs salary levels for compensation benchmarking.
- **Highlights:** Clean/processed datasets, regression suite (Linear, Polynomial, Random Forest, SVR), feature analysis, environment files for reproducibility.
- **Deliverables:** Notebook set, scripts, ready-to-serve figures/models.

### Unicorn Companies Growth and Investment Data Analysis
- **Dataset:** Unicorn_Companies.csv with valuation, geography, investors, stages.
- **Highlights:** Exploratory docs, Jupyter & R Markdown notebooks, predictive models (valuation regression, stage classification), investor insights.
- **Deliverables:** Plot library, project structure documentation, modeling scripts.

## Getting Started

1. **Clone the repository**
   ```bash
   git clone https://github.com/Nana-Safo-Duker/Data-Science-Predictive-Analytics.git
   cd Data-Science-Predictive-Analytics
   ```
2. **Pick a project folder** and review its local `README.md` (or documentation directory) for project-specific setup.
3. **Create an environment**
   ```bash
   # Python example
   python -m venv .venv
   .venv\Scripts\activate  # or source .venv/bin/activate
   pip install -r <project>/requirements.txt

   # R example
   Rscript <project>/scripts/r/install_packages.R
   ```
4. **Run notebooks or scripts** from the chosen project. Each notebook is numbered to illustrate the recommended execution order (EDA → statistics → modeling).

## Repository Structure

```
├── <project-name>/
│   ├── data/                 # Raw or processed datasets (kept lightweight)
│   ├── docs/                 # Project summaries, analysis guides, compliance notes
│   ├── notebooks/
│   │   ├── python/           # Ordered .ipynb files
│   │   └── r/                # Mirrored R notebooks/Rmd files
│   ├── scripts/
│   │   ├── python/           # CLI-friendly analysis scripts
│   │   └── r/                # R equivalents
│   ├── results/              # Figures, tables, model artifacts
│   └── requirements*/        # Dependency manifests (where applicable)
└── README.md                 # You are here
```

## Contributing & Support
- Issues and pull requests are welcome for bug fixes, documentation improvements, or new analytical modules.
- Please review each project’s contribution guidelines (where available) before submitting changes.
- For questions, open an issue or reach out via the website listed in the About section.

## License & Data Ethics
- Licensing files reside within individual project folders. Respect dataset source licenses and usage restrictions before redistribution.
- All datasets are intended for educational/research use. Remove or anonymize sensitive information when adapting these workflows for production environments.

---

Curated with ❤️ to accelerate real-world applied analytics projects. Happy exploring!

