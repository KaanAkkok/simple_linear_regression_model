SUMMARY:
-----------------------------------------------------
This model predict salary based on yearly experience.

THE EQUATION :

Salary = 9312.57 x YearsExperience + 26780.09

X = Years of Experience

y = Salary

STEPS:
-----------------------------------------------------
1- Importing libraries:

    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    
2- Importing dataset. 

    dataset = pd.read_csv('Salary_Data.csv')
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values

3- Spliting the dataset into the training and testing set with sklearn. (test_size=0.2)

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

4- Training the simple linear regression model on the training set.

    from sklearn.linear_model import LinearRegression
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)

5- Visualising the training and test set result.

    plt.scatter(X_train, y_train, color='red')
    plt.plot(X_train, regressor.predict(X_train), color='blue')
    plt.title('Salary vs Experience (Training set)')
    plt.xlabel('Years of experience')
    plt.ylabel('Salary')
    plt.show()

    plt.scatter(X_test, y_test, color='red')
    plt.plot(X_train, regressor.predict(X_train), color='blue')
    plt.title('Salary vs Experience (Test set)')
    plt.xlabel('Years of experience')
    plt.ylabel('Salary')
    plt.show()
