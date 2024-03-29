import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler,StandardScaler

# Load dataset
def load_data():
    dataset = pd.read_csv("diabetes2.csv")
    df = dataset.copy()
    return df

df=load_data()
def show_explore_page():
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
    fig, ax = plt.subplots(figsize=(6, 4))
    df['Outcome'].value_counts().plot(kind='bar', color=['green', 'yellow'], ax=ax)
    ax.set_title('Outcome Counts')
    ax.set_xlabel('Outcome')
    ax.set_ylabel('Count')
    ax.set_xticks([0, 1])
    ax.set_xticklabels([outcome_labels[0], outcome_labels[1]], rotation=0)

    st.pyplot(fig)

        # Create the Outcome_Label column
    df['Outcome_Label'] = df['Outcome'].map(outcome_labels)

    # Pairplot also indicating the binary classes
    st.write(
        """
        #### Pairplot with Outcome Label
        """
    )
    # Create a Pairplot with the Outcome_Label
    fig = sns.pairplot(df, hue='Outcome_Label', height=4)
    st.pyplot(fig)



    # PLotting the correlation matrix
    st.write(
        """
        #### Correlation Matrix
        """
    )
    # Filter out non-numeric columns
    numeric_df = df.select_dtypes(include=['number'])

    # Create a figure and axis object
    fig, ax = plt.subplots(figsize=(7, 5))

    # Plot the heatmap on the axis
    sns.heatmap(numeric_df.corr(), annot=True, cmap='viridis', fmt=".2f", ax=ax)

    # Set the title
    ax.set_title('Correlation Matrix')

    # Display the plot using Streamlit
    st.pyplot(fig)

    # Plotting the relationships between some of the features
    st.write(
        """
        #### Scatterplots of Feature Relationships
        """
    )

    # Plotting the relationships between some of the features
    fig, ax = plt.subplots(figsize=(20, 15))

    plt.subplot(3, 3, 1)
    sns.scatterplot(data=df, x="BMI", y="SkinThickness", hue="Outcome", palette={0: 'seagreen', 1: 'tomato'})
    plt.title('BMI Vs Skinthickness')

    plt.subplot(3, 3, 2)
    sns.scatterplot(data=df, x="BloodPressure", y="Age", hue="Outcome", palette={0: 'purple', 1: 'deeppink'})
    plt.title('Age Vs BloodPressure')

    plt.subplot(3, 3, 3)
    sns.scatterplot(data=df, x="Glucose", y="Insulin", hue="Outcome", palette={0: 'royalblue', 1: 'darkgoldenrod'})
    plt.title('Insulin Vs Glucose')

    # Show the plot using st.pyplot()
    st.pyplot(fig)

# Call the function to show the explore page
#show_explore_page(df)
