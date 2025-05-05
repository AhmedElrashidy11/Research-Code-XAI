# Research-Code-XAI

This repository contains the mid-progress submissions for the DSAI 305 course. It includes data preprocessing, exploratory data analysis (EDA), and various machine learning models developed by different team members. The goal is to explore diverse modeling techniques on the diabetes health indicators dataset.

---

##  About the Project

Millions of people around the world are affected by diabetes mellitus, a chronic metabolic disease that requires early and accurate classification for effective management. This project explores the use of diverse machine learning (ML) and deep learning (DL) models for classifying diabetes using health indicators from the BRFSS 2015 dataset.

The objective is to compare and evaluate a variety of ML and DL algorithms—including ensemble techniques, neural networks, and hybrid models—to improve predictive accuracy and interpretability. Each team member contributed models ranging from traditional classifiers (like logistic regression and decision trees) to advanced methods (such as conditional GANs and convolutional neural networks).

Key aspects of the project include:

* Preprocessing and handling imbalanced data
* Model training using ensemble and deep learning techniques
* Performance evaluation and comparison
* Exploring model interpretability for explainable AI (XAI)

This work draws on recent research that highlights the effectiveness of hybrid models and deep learning in medical diagnosis, emphasizing the importance of data quality, feature selection, and advanced modeling frameworks in improving diabetes classification systems.

---

##  Project Structure

* `Preprocessing_and_EDA.ipynb`: General data preprocessing and EDA notebook.
* `diabetes_012_health_indicators_BRFSS2015.csv`: Dataset used for all models.

Model implementations are available in individual branches (see below).

---

##  Branches Structure

Each branch contains contributions from a specific team member, including different ML models:

###  `main`
* `Preprocessing_and_EDA.ipynb`: General data preprocessing and EDA notebook.
* `diabetes_012_health_indicators_BRFSS2015.csv`: Dataset used for all models.
* `preprocessed_diabetes_data.csv`: Dataset after preprocessing and cleaning.
* `requirements.txt`: The all libraries for code. 
* `README.md`

###  `Ahmed-Elrashidy`

* `Ahmed_Mohammed_202202168_(Nueral_Network_Model).ipynb` – Neural Network
* `Ahmed_Mohammed_202202168(AdaBoost_(Adaptive_Boosting)_Model).ipynb` – AdaBoost
* `Ahmed_Mohammed_202202168(XGBoost_&_DT_by_Conditional_GAN).ipynb` – XGBoost & Decision Tree using Conditional GAN

###  `Mohamed-Alaa`

* `Logistic_dsai305(mohamedalaa).ipynb` – Logistic Regression
* `dsai305_SVM(mohamedalaa).ipynb` – Support Vector Machine
* `CNN_dsai305(mohamedalaa).ipynb` – Convolutional Neural Network

###  `Mostafa-Adam`

* `Team_17__CatBoost_DSAI_305_Mostafa_Adam_Proj_Phase_2.ipynb` – CatBoost Model
* `Team_17_DecisionTree_DSAI_305_Mostafa_Adam_Proj_Phase_2.ipynb` – Decision Tree
* `KNN_Team_17_DSAI_305_Mostafa_Adam_Proj_Phase_2.ipynb` – K-Nearest Neighbors (KNN)

###  `Muhammed-Kamal`

* `Muhammad_Kamal_202200899(GaussianNB).ipynb` – Gaussian Naive Bayes
* `Muhammad_Kamal_202200899(Gradient_Boosting).ipynb` – Gradient Boosting
* `Muhammad_Kamal_202200899(Random_forst).ipynb` – Random Forest

---

##  Setup Instructions

###  Requirements

You can install all required dependencies using the provided `requirements.txt` file:

```bash
pip install -r requirements.txt
```

Or install packages manually:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost catboost lightgbm tensorflow keras notebook
```

1. **Clone the repository:**

   ```bash
   git clone https://github.com/AhmedElrashidy11/Research-Code-XAI.git
   cd Research-Code-XAI
   ```

2. **(Optional) Create and activate a virtual environment:**

   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   If `requirements.txt` is not available, install common packages manually:

   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn xgboost catboost lightgbm tensorflow keras
   ```

4. **Run Jupyter notebooks:**

   ```bash
   jupyter notebook
   ```

---

##  Dataset Information

* **Name:** `diabetes_012_health_indicators_BRFSS2015.csv`
* **Source:** Behavioral Risk Factor Surveillance System (BRFSS)
* **Target:** Binary classification for diabetes presence based on health indicators

---

##  Team Members

This project is a collaborative effort by Team 17 from the DSAI 305 course:

* **Ahmed Elrashidy** – AdaBoost, Neural Network, XGBoost with Conditional GAN
* **Mohamed Alaa** – Logistic Regression, Support Vector Machine, Convolutional Neural Network
* **Mostafa Adam** – CatBoost, Decision Tree, K-Nearest Neighbors
* **Muhammad Kamal** – Gaussian Naive Bayes, Gradient Boosting, Random Forest

Each member explored different machine learning techniques and contributed through dedicated branches.

