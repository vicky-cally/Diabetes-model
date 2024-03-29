import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# Load dataset
dataset = pd.read_csv("diabetes2.csv")
df = dataset.copy()

def show_explore_page(df):
    st.title("Explore Diabetes Data")

    st.write(
        """
        ### Diabetes Dataset Analysis
        """
    )

    # Plot of the count
    outcome_labels = {0: 'Non-Diabetic', 1: 'Diabetic'}
    st.write(
        """
        #### Outcome Counts
        """
    )
    plt.figure(figsize=(6, 4))
    df['Outcome'].value_counts().plot(kind='bar', color=['green', 'yellow'])
    plt.title('Outcome Counts')
    plt.xlabel('Outcome')
    plt.ylabel('Count')
    plt.xticks(ticks=[0, 1], labels=[outcome_labels[0], outcome_labels[1]], rotation=0)
    st.pyplot()

    # Pairplot also indicating the binary classes
    st.write(
        """
        #### Pairplot with Outcome Label
        """
    )
    df['Outcome_Label'] = df['Outcome'].map(outcome_labels)
    sns.pairplot(df, hue='Outcome_Label')
    df = df.drop('Outcome_Label', axis=1)
    st.pyplot()

    # PLotting the correlation matrix
    st.write(
        """
        #### Correlation Matrix
        """
    )
    plt.figure(figsize=(7, 5))
    sns.heatmap(df.corr(), annot=True, cmap='viridis', fmt=".2f")
    plt.title('Correlation Matrix')
    st.pyplot()

    # Plotting the relationships between some of the features
    st.write(
        """
        #### Scatterplots of Feature Relationships
        """
    )
    plt.figure(figsize=(20, 15))
    plt.subplot(3, 3, 1)
    sns.scatterplot(data=df, x="BMI", y="SkinThickness", hue="Outcome", palette={0: 'seagreen', 1: 'tomato'})
    plt.title('BMI Vs Skinthickness')
    plt.subplot(3, 3, 2)
    sns.scatterplot(data=df, x="BloodPressure", y="Age", hue="Outcome", palette={0: 'purple', 1: 'deeppink'})
    plt.title('Age Vs BloodPressure')
    plt.subplot(3, 3, 3)
    sns.scatterplot(data=df, x="Glucose", y="Insulin", hue="Outcome", palette={0: 'royalblue', 1: 'darkgoldenrod'})
    plt.title('Insulin Vs Glucose')
    st.pyplot()

    # Classification report
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Classification report and confusion matrix for KNN classifier (example)
    st.write(
        """
        #### Classification Report and Confusion Matrix for KNN Classifier
        """
    )
    # Add your classification model here and its predictions
    # For example:
    model_knn = KNeighborsClassifier(metric='manhattan', n_neighbors=20, weights= 'distance')
    model_knn.fit(xtrain, ytrain)
    model_knn_predictions = model_knn.predict(xtest)

    # Classification report
    report = classification_report(ytest, model_knn_predictions, output_dict=True)
    sns.heatmap(pd.DataFrame(report).iloc[:-1, :].T, annot=True, cmap="Pastel1")
    plt.title("Precision, Recall, F1-Score for the KNN Algorithm")
    st.pyplot()

    # Confusion matrix
    conf_matrix_knn = confusion_matrix(ytest, model_knn_predictions)
    sns.heatmap(conf_matrix_knn, annot=True, fmt='d', cmap='cividis')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix for K-Nearest Neighbors Classifier')
    st.pyplot()

# Call the function to show the explore page
show_explore_page(df)
