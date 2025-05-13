# EXPERIMENT NO: 08
# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import necessary libraries: pandas, scikit-learn modules for model building, and evaluation.

2. Load the dataset using pandas read_csv() with encoding set to 'Windows-1252'.

3. Display the first few rows and check the shape of the dataset.

4. Extract input features (text messages) and output labels (spam/ham) from the dataset.

5. Split the data into training and testing sets using train_test_split().

6. Initialize CountVectorizer to convert text data into numerical form.

7. Fit the vectorizer on the training data and transform both training and test data.

8. Initialize the Support Vector Machine (SVM) classifier using the SVC class.

9. Train the SVM model using the training data with the fit() method.

10. Make predictions on the test data using the predict() method.

11. Evaluate the model using:
    
    a. accuracy_score() for accuracy,
    
    b. confusion_matrix() for confusion matrix,
    
    c. classification_report() for precision, recall, and F1-score.

13. Print the predicted output, accuracy, confusion matrix, and classification report.

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: HARI PRIYA M
RegisterNumber: 212224240047
*/
```

    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.svm import SVC
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
    df = pd.read_csv("/content/spam.csv", encoding='Windows-1252')
    print(df)
    
    print("Dataset Shape:", df.shape)
    
    messages = df['v2'].values
    labels = df['v1'].values
    print("Messages shape:", messages.shape)
    print("Labels shape:", labels.shape)
    
    X_train, X_test, Y_train, Y_test = train_test_split(messages, labels, test_size=0.2, random_state=0)
    X_train
    X_train.shape
    
    vectorizer = CountVectorizer()
    X_train_vector = vectorizer.fit_transform(X_train)
    X_test_vector = vectorizer.transform(X_test)
    svm_model = SVC()
    svm_model.fit(X_train_vector, Y_train)
    Y_pred = svm_model.predict(X_test_vector)
    print("Predicted Output:\n", Y_pred)
    
    accuracy = accuracy_score(Y_test, Y_pred)
    print("Accuracy:", accuracy)
    
    conf_matrix = confusion_matrix(Y_test, Y_pred)
    print("Confusion Matrix:\n", conf_matrix)
    
    class_report = classification_report(Y_test, Y_pred)
    print("Classification Report:\n", class_report)

## Output:

Displaying the Dataset

![Screenshot 2025-05-13 090950](https://github.com/user-attachments/assets/5673b777-04d6-4609-9dfc-d02eb45db7e5)

df.shape

![Screenshot 2025-05-13 090939](https://github.com/user-attachments/assets/898f2373-0eee-4fd9-80c3-3dcd139a065c)

Printing the shape of Messages and Labels

![Screenshot 2025-05-13 090923](https://github.com/user-attachments/assets/f8cadd9d-8eab-4c85-aba2-76128925cc84)

Displaying X_train and it's shape

![Screenshot 2025-05-13 091803](https://github.com/user-attachments/assets/ef48b963-70c2-4196-9972-d077b5183d8a)

![Screenshot 2025-05-13 090916](https://github.com/user-attachments/assets/8837c87c-da80-4c6c-8065-06370eaf5d69)

svm_model

![Screenshot 2025-05-13 090837](https://github.com/user-attachments/assets/cffe7ea4-5848-404a-a20d-2e1a25e3dcf4)

Predicting the output

![Screenshot 2025-05-13 090829](https://github.com/user-attachments/assets/48fd84d0-61f5-438a-85e6-d9fbfc37dea4)

Evaluating the Accuracy

![Screenshot 2025-05-13 090824](https://github.com/user-attachments/assets/76cd604e-c3ff-4513-98d2-d7c392d5333b)

Confusion Matrix

![Screenshot 2025-05-13 091914](https://github.com/user-attachments/assets/a51e5907-06fa-495d-b835-f59ffaa1dd4c)

Classification Report

![Screenshot 2025-05-13 092331](https://github.com/user-attachments/assets/3ad2face-77d6-4ca8-939a-e838842f8a36)

## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
