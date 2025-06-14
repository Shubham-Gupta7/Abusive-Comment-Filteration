**Abusive Comment Filtration**

📌 Project Overview
The Abusive Comment Filtration system is a multi-label text classification project aimed at detecting and categorizing online toxic comments. Built using the Jigsaw Toxic Comment Classification Challenge dataset, this project utilizes Natural Language Processing (NLP), Machine Learning (ML), and class imbalance handling techniques to classify user comments into one or more of the following categories:

Toxic
Severe Toxic
Obscene
Threat
Insult
Identity Hate
This project is especially relevant in the era of increasing online interactions, where automatic content moderation is crucial for creating safer digital communities.

🔍 Objectives
Load and preprocess a large corpus of online user comments.
Perform Exploratory Data Analysis (EDA) to understand class imbalance and comment characteristics.
Clean and lemmatize the text using NLP techniques.
Vectorize text using TF-IDF.
Handle class imbalance using ADASYN oversampling.
Train a Gradient Boosting Classifier wrapped in a MultiOutputClassifier to predict multiple labels.
Evaluate performance using accuracy, classification reports, and confusion matrices for each class.

📂 Dataset
Source: Jigsaw Toxic Comment Classification Challenge
Dataset files used:
train.csv
test.csv
test_labels.csv
sample_submission.csv

🛠️ Tech Stack & Libraries
Python
Pandas, NumPy
Matplotlib, Seaborn
NLTK for text preprocessing
scikit-learn for modeling
imbalanced-learn (ADASYN) for handling imbalanced data
TF-IDF Vectorizer for feature extraction
Gradient Boosting for classification
ydata-profiling for automated data profiling

📊 Key Features
✅ Exploratory Data Analysis (EDA)
Visualized label distribution using bar charts, pie charts, and box plots.
Generated correlation heatmaps and statistical summaries.
✅ NLP Preprocessing
Lowercasing, punctuation removal, stop word removal, and lemmatization using WordNetLemmatizer.
✅ Feature Engineering
Comment length and word count distribution visualized.
Applied TF-IDF vectorization with 15,000 max features.
✅ Class Imbalance Handling
Used ADASYN oversampling technique on each label independently to balance the dataset.
✅ Model Building
Trained a MultiOutput Gradient Boosting Classifier to support multi-label predictions.
✅ Evaluation
Reported per-label accuracy.
Displayed classification report with precision, recall, and F1-score.
Plotted individual confusion matrices for each label.

📈 Results
The model achieved strong accuracy and balanced performance across multiple labels despite the inherent imbalance in the dataset.
Visual diagnostics provided detailed insights into model behavior and class-level performance.

📌 Future Improvements
Experiment with deep learning models like LSTM, BERT, or RoBERTa for better contextual understanding.
Deploy the model via a web app using Streamlit for real-time comment filtering.
Implement Explainable AI (XAI) techniques to interpret predictions.

👨‍💻 Developed by
Shubham Gupta
Final Year B.Tech CSE (Data Science) Student
Passionate about NLP, Machine Learning & Ethical AI
LinkedIn Profile: https://www.linkedin.com/in/shubham-gupta777
