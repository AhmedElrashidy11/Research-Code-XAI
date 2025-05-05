# Research-Code-XAI

This repository contains the mid-progress submissions for the DSAI 305 course. It includes data preprocessing, exploratory data analysis (EDA), and various machine learning models developed by different team members. The goal is to explore diverse modeling techniques on the diabetes health indicators dataset.

---

## 📁 Project Structure

* `Preprocessing_and_EDA.ipynb`: General data preprocessing and EDA notebook.
* `diabetes_012_health_indicators_BRFSS2015.csv`: Dataset used for all models.
* `README.md`

Model implementations are available in individual branches (see below).

---

## 🌿 Branches Structure

Each branch contains contributions from a specific team member, including different ML models:

### 🔵 `main`

* `README.md`

### 🟢 `Ahmed-Elrashidy`

* `Ahmed_Mohammed_202202168_(Nueral_Network_Model).ipynb` – Neural Network
* `Ahmed_Mohammed_202202168(AdaBoost_(Adaptive_Boosting)_Model).ipynb` – AdaBoost
* `Ahmed_Mohammed_202202168(XGBoost_&_DT_by_Conditional_GAN).ipynb` – XGBoost & Decision Tree using Conditional GAN

### 🟡 `Mostafa-Adam`

* `Team_17__CatBoost_DSAI_305_Mostafa_Adam_Proj_Phase_2.ipynb` – CatBoost Model
* `Team_17_DecisionTree_DSAI_305_Mostafa_Adam_Proj_Phase_2.ipynb` – Decision Tree
* `KNN_Team_17_DSAI_305_Mostafa_Adam_Proj_Phase_2.ipynb` – K-Nearest Neighbors (KNN)

### 🟣 `Muhammed-Kamal`

* `Muhammad_Kamal_202200899(GaussianNB).ipynb` – Gaussian Naive Bayes
* `Muhammad_Kamal_202200899(Gradient_Boosting).ipynb` – Gradient Boosting
* `Muhammad_Kamal_202200899(Random_forst).ipynb` – Random Forest

### 🔴 `Mohamed-Alaa`

* `Logistic_dsai305(mohamedalaa).ipynb` – Logistic Regression
* `dsai305_SVM(mohamedalaa).ipynb` – Support Vector Machine
* `CNN_dsai305(mohamedalaa).ipynb` – Convolutional Neural Network

---

## 🛠️ Setup Instructions

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

## 📊 Dataset Information

* **Name:** `diabetes_012_health_indicators_BRFSS2015.csv`
* **Source:** Behavioral Risk Factor Surveillance System (BRFSS)
* **Target:** Binary classification for diabetes presence based on health indicators

---

## 🤝 Contributing

Want to contribute? Fork the repo, make your changes, and open a pull request. Be sure to branch off `main` or work within your designated feature branch.

---


