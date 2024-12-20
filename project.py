#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install kaggle


# In[2]:


get_ipython().system('mkdir -p ~/.kaggle')
get_ipython().system('cp kaggle.json ~/.kaggle/')


# In[3]:


get_ipython().system('chmod 600 ~/.kaggle/kaggle.json')


# In[4]:


get_ipython().system('kaggle competitions download -c jigsaw-toxic-comment-classification-challenge')


# In[5]:


get_ipython().system('unzip -o jigsaw-toxic-comment-classification-challenge.zip -d ./toxic-comment-classification')


# In[6]:


get_ipython().system('cd ./toxic-comment-classification')


# In[7]:


get_ipython().system('unzip -o ./toxic-comment-classification/train.csv.zip -d ./toxic-comment-classification')
get_ipython().system('unzip -o ./toxic-comment-classification/test.csv.zip -d ./toxic-comment-classification')
get_ipython().system('unzip -o ./toxic-comment-classification/test_labels.csv.zip -d ./toxic-comment-classification')
get_ipython().system('unzip -o ./toxic-comment-classification/sample_submission.csv.zip -d ./toxic-comment-classification')


# In[8]:


get_ipython().system('ls ./toxic-comment-classification')


# In[9]:


cd toxic-comment-classification/


# In[10]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# Load the train dataset
df_train = pd.read_csv("train.csv.zip")
# Display the first few rows
print(df_train.head())


# In[11]:


print(df_train.describe())


# In[12]:


print(df_train.info())


# In[13]:


#bar plot
labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
plt.figure(figsize=(14, 8))

for i, label in enumerate(labels):
    plt.subplot(2, 3, i + 1)
    df_train[label].value_counts().plot(kind='bar', color=['skyblue', 'salmon'])
    plt.title(f'Distribution of {label}')
    plt.xlabel('Label')
    plt.ylabel('Count')
    plt.xticks(ticks=[0, 1], labels=['Not Present', 'Present'])

plt.tight_layout()
plt.show()


# In[14]:


plt.figure(figsize=(12, 6))
data = [df_train[label] for label in labels]
plt.boxplot(data, labels=labels)
plt.title('Boxplots of Label Scores')
plt.ylabel('Score')
plt.xticks(rotation=45)
plt.show()


# In[15]:


plt.figure(figsize=(14, 8))

for i, label in enumerate(labels):
    plt.subplot(2, 3, i + 1)
    df_train[label].value_counts().plot(kind='pie', autopct='%1.1f%%', startangle=90, colors=['lightgreen', 'salmon'])
    plt.title(f'Proportion of {label}')
    plt.ylabel('')

plt.tight_layout()
plt.show()


# In[16]:


import seaborn as sns
corr = df_train[labels].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()


# In[17]:


get_ipython().system('pip install ydata-profiling')
from ydata_profiling import ProfileReport

# Generate a profiling report
profile = ProfileReport(df_train, title="Pandas Profiling Report", explorative=True)

# To view the report in a Jupyter notebook or save it as an HTML file
profile.to_file("your_report.html")


# In[18]:


pip install pandas nltk


# In[19]:


import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK resources (only need to run once)
nltk.download('stopwords')
nltk.download('wordnet')

# Load your dataset
df = pd.read_csv('train.csv.zip')  # Change the path to your dataset


# In[20]:


def preprocess_text(text):
    # Lowercase the text
    text = text.lower()
    
    # Remove punctuation and special characters
    text = re.sub(r'[^\w\s]', '', text)
    
    # Tokenize the text
    tokens = text.split()
    
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    # Lemmatize the tokens
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    # Join tokens back into a single string
    return ' '.join(tokens)


# In[21]:


# Apply preprocessing
df['cleaned_comment_text'] = df['comment_text'].apply(preprocess_text)

# Display the first few rows to verify the changes
print(df[['comment_text', 'cleaned_comment_text']].head())


# In[22]:


df_train['comment_length'] = df_train['comment_text'].apply(len)
plt.figure(figsize=(10, 6))
sns.histplot(df_train['comment_length'], bins=50)
plt.title('Distribution of Comment Lengths')
plt.show()


# In[23]:


df_train['word_count'] = df_train['comment_text'].apply(lambda x: len(x.split()))
plt.figure(figsize=(10, 6))
sns.histplot(df_train['word_count'], bins=50)
plt.title('Distribution of Word Counts')
plt.show()


# In[24]:


sns.pairplot(df_train, vars=['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate'])
plt.show()


# In[25]:


# Select only numeric columns
numeric_columns = df_train.select_dtypes(include=['float64', 'int64'])
# Calculate correlation matrix
correlation = numeric_columns.corr()
# Plot the heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(correlation, annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()


# In[26]:


duplicates = df_train[df_train.duplicated(subset=['comment_text'])]
print(f'Number of duplicate comments: {duplicates.shape[0]}')


# In[27]:


get_ipython().system('pip install scikit-learn imbalanced-learn')


# In[28]:


# Extract target columns
target_columns = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
y = df[target_columns].values  # Get the values as a NumPy array


# In[29]:


pip install imbalanced-learn


# In[30]:


import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split


# In[31]:


def preprocess_and_vectorize(df):
    vectorizer = TfidfVectorizer(max_features=15000, stop_words='english')
    X_vectorized = vectorizer.fit_transform(df['comment_text'])
    return X_vectorized


# In[32]:


X_vectorized = preprocess_and_vectorize(df_train)


# In[33]:


# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.25, random_state=42)

# Print shapes to verify
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)


# In[34]:


from imblearn.over_sampling import ADASYN
# Initialize ADASYN
adasyn = ADASYN(random_state=42)

# Prepare to hold resampled datasets
X_train_balanced_list = []
y_train_balanced_list = []

# Iterate through each label and apply ADASYN
for i in range(y_train.shape[1]):  # Loop over each label
    print(f"Balancing for label {i}...")
    
    # Get the current label's binary classification
    y_current_label = y_train[:, i]
    
    # Only proceed if there are at least two classes in the current label
    if len(np.unique(y_current_label)) > 1:
        # Apply ADASYN
        X_resampled, y_resampled = adasyn.fit_resample(X_train, y_current_label)

        # Collect balanced data
        X_train_balanced_list.append(X_resampled)
        y_train_balanced_list.append(y_resampled)
    else:
        print(f"Skipping label {i} due to a single-class situation.")
        # If there's only one class, append the original data
        X_train_balanced_list.append(X_train)
        y_train_balanced_list.append(y_current_label)

# Convert lists of arrays to single sparse matrix
from scipy.sparse import vstack

# Stack all the balanced features vertically
X_train_balanced = vstack(X_train_balanced_list)

# Concatenate all the balanced labels into a single array
y_train_balanced = np.concatenate(y_train_balanced_list)

# Print the shapes of the balanced datasets
print("Balanced training data shape:", X_train_balanced.shape)
print("Balanced labels shape:", y_train_balanced.shape)


# In[35]:


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_class_distribution(y, title):
    """
    Plot the distribution of classes in the given labels.

    Parameters:
    - y: Array of class labels.
    - title: Title for the plot.
    """
    plt.figure(figsize=(8, 6))
    sns.countplot(x=y, palette='viridis')
    plt.title(title)
    plt.xlabel('Class Labels')
    plt.ylabel('Count')
    plt.xticks(ticks=np.arange(len(np.unique(y))), labels=np.unique(y), rotation=0)
    plt.grid(axis='y')
    plt.show()

# Assuming you have balanced each label and stored them in separate variables
toxic_balanced = y_train_balanced_list[0]  # Using the first element for toxic
severe_toxic_balanced = y_train_balanced_list[1]  # Second for severe toxic
obscene_balanced = y_train_balanced_list[2]  # Third for obscene
threat_balanced = y_train_balanced_list[3]  # Fourth for threat
insult_balanced = y_train_balanced_list[4]  # Fifth for insult
identity_hate_balanced = y_train_balanced_list[5]  # Sixth for identity hate

# Visualize each label without reshaping
plot_class_distribution(toxic_balanced, 'Balanced Class Distribution (Toxic)')
plot_class_distribution(severe_toxic_balanced, 'Balanced Class Distribution (Severe Toxic)')
plot_class_distribution(obscene_balanced, 'Balanced Class Distribution (Obscene)')
plot_class_distribution(threat_balanced, 'Balanced Class Distribution (Threat)')
plot_class_distribution(insult_balanced, 'Balanced Class Distribution (Insult)')
plot_class_distribution(identity_hate_balanced, 'Balanced Class Distribution (Identity Hate)')


# In[36]:


print(toxic_balanced)


# In[37]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.multioutput import MultiOutputClassifier

# Assuming y_train_balanced_list contains your balanced labels
# Find the minimum size among all label arrays and the feature set
min_size = min([y.shape[0] for y in y_train_balanced_list] + [X_train.shape[0]])

# Truncate all label arrays to the same size
y_train_balanced_truncated = [y[:min_size] for y in y_train_balanced_list]

# Combine the balanced labels vertically and transpose to get shape (samples, labels)
y_train_balanced = np.vstack(y_train_balanced_truncated).T

# Truncate features to match the smallest label size
X_train_balanced = X_train[:min_size]  # Ensure features have the same number of rows as labels

# Ensure both X_train_balanced and y_train_balanced have the same number of samples
print(f"Shape of X_train_balanced: {X_train_balanced.shape}")
print(f"Shape of y_train_balanced: {y_train_balanced.shape}")

# Split into training and test sets
X_train_final, X_test, y_train_final, y_test = train_test_split(X_train_balanced, y_train_balanced, test_size=0.25, random_state=42)

# Initialize and train the Gradient Boosting model using MultiOutputClassifier for multi-label
gb_model = MultiOutputClassifier(GradientBoostingClassifier(n_estimators=100, random_state=42))
gb_model.fit(X_train_final, y_train_final)

# Predictions
y_pred = gb_model.predict(X_test)

# Evaluate the model
print("Accuracy for each class:")
for i, label in enumerate(['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']):
    accuracy = accuracy_score(y_test[:, i], y_pred[:, i])
    print(f"{label}: {accuracy:.2f}")

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']))


# In[38]:


import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Function to plot confusion matrix for each label
def plot_confusion_matrices(y_true, y_pred, labels):
    num_labels = len(labels)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))  # Adjust the figure size as needed
    axes = axes.flatten()  # Flatten the axes array for easy indexing
    
    for i, label in enumerate(labels):
        # Calculate confusion matrix
        cm = confusion_matrix(y_true[:, i], y_pred[:, i])
        
        # Plot confusion matrix
        sns.heatmap(cm, annot=True, fmt='d', ax=axes[i], cmap='Blues',
                    xticklabels=['Not ' + label, label],
                    yticklabels=['Not ' + label, label])
        
        axes[i].set_title(f'Confusion Matrix for {label}')
        axes[i].set_xlabel('Predicted Label')
        axes[i].set_ylabel('True Label')
    
    plt.tight_layout()
    plt.show()

# Call the function to plot confusion matrices
plot_confusion_matrices(y_test, y_pred, ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate'])

