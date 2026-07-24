# 🔍 Network Traffic Analysis with Streamlit and Machine Learning

Web application developed with **Streamlit** that detects network intrusions from a CSV file using several **Machine Learning** models.

The application trains different classification algorithms on a training dataset and then analyzes a user-provided file to detect normal or malicious connections.

---

# 📸 Overview

Main features:

- 📂 Test CSV file import
- ⚙️ Automatic data preprocessing
- 🤖 Training of multiple Machine Learning models
- 🔎 Network intrusion detection
- 📊 Results display in tabular format
- 📈 Summary chart
- 📉 Analysis progress bar

---

# 🚀 Technologies Used

- Python 3
- Streamlit
- Pandas
- NumPy
- Scikit-Learn

---

# 📦 Installation

## 1. Clone the project

```bash
git clone https://github.com/Rahkillah/Network_Analyse

cd Network_Analyse
```

## 2. Install dependencies

```bash
pip install streamlit pandas numpy scikit-learn
```

---

# ▶️ Run the Application

```bash
streamlit run app.py
```

*(Replace `app.py` with the name of your Python file.)*

---

# 📁 Project Structure

```
.
│
├── app.py
├── Train_data.csv
├── requirements.txt
├── README.md
└── images/
    └── capture.png
```

---

# 📄 Dataset

The project uses two CSV files.

## Train_data.csv

Dataset used to train the models.

It must contain in particular:

- numerical variables
- categorical variables
- the **class** column (normal / anomaly)

---

## test_data.csv

File uploaded by the user.

It must have the same columns as the training dataset (except for the target column if applicable).

---

# ⚙️ Data Preprocessing

Before analysis, several operations are performed automatically:

- dropping the column:

```
num_outbound_cmds
```

- normalization of numerical variables with:

```
StandardScaler
```

- encoding of categorical variables with:

```
LabelEncoder
```

---

# 🧠 Machine Learning Models

The following models are trained:

| Model | Library |
|---------|--------------|
| K-Nearest Neighbors | Scikit-Learn |
| Logistic Regression | Scikit-Learn |
| Bernoulli Naive Bayes | Scikit-Learn |
| Decision Tree | Scikit-Learn |

Currently, only the **KNeighborsClassifier** model is used to make the final predictions.

---

# 📊 Displayed Results

After the analysis, the application displays:

- the first analyzed rows
- the status of each connection
- a progress bar
- the percentage of normal connections
- a distribution chart of the predictions

Example:

| Row | Result |
|--------|----------|
| 1 | Normal |
| 2 | Intrusion detected |
| 3 | Normal |
| ... | ... |

---

# 🔍 Analysis Workflow

```text
Loading the CSV file
          │
          ▼
Data Preprocessing
          │
          ▼
Normalization
          │
          ▼
Encoding
          │
          ▼
Model Training
          │
          ▼
Prediction
          │
          ▼
Display Results
```

---

# 📈 Algorithm Used

The model currently used for predictions is:

```
KNeighborsClassifier
```

The other models are trained to easily allow future comparison:

- Logistic Regression
- Bernoulli Naive Bayes
- Decision Tree

---

# 💡 Possible Improvements

- Performance comparison between different models
- Confusion matrix calculation
- Accuracy, Recall, Precision, F1-score
- ROC Curve
- Saving results in CSV format
- Downloading PDF reports
- Dynamic selection of the model to use
- Hyperparameter tuning
- Cross-validation
- More interactive interface

---

# 📚 Dependencies

```
streamlit
pandas
numpy
scikit-learn
```

---

# 🛠 Example requirements.txt

```
streamlit
pandas
numpy
scikit-learn
```

---

# 📌 Usage Example

1. Launch the application.

2. Upload a **test_data.csv** file.

3. Click on:

```
Analyze Traffic
```

4. View:

- the predictions table
- the progress
- the statistics
- the chart

---

# ⚠️ Note

The application currently considers two classes:

- **normal**
- **anomaly**

The displayed accuracy corresponds to the percentage of predictions classified as **normal** and does not represent a standard model evaluation metric. To properly measure performance, it is recommended to use a labeled test set along with evaluation metrics such as accuracy, precision, recall, or F1-score.

---

# 👨‍💻 Author

**Andrandraina RANDRIANAIVO**

Project created as part of a network intrusion detection project using Python, Streamlit, and Machine Learning.

---

# 📜 License

This project is distributed under the MIT License.

You are free to use, modify, and share it.