#Importing libraries
import math
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

def Prediction(rd,ad,ms):
    design = 50 * "-"
    #Load the dataset
    df=pd.read_csv("50_Startups.csv",encoding="unicode_escape")
    print(design)
    print(df.head())

    df=df.rename(columns={"R&D Spend":"rd","Administration":"ad","Marketing Spend":"ms"},inplace=False)
    print(design)
    print(df.head())

    Investment=df.loc[:,["rd","ad","ms"]]
    Investment.head()

    Investment["Total_Investment"]=Investment.sum(axis=1)

    Investment["Total_Investment"]=Investment.mean()

    df["Profit"].mean()

    print(design,"\n")
    print("Information of dataset is given below:")
    print(design)
    df.info()
    print("The shape of dataset is:",df.shape)
    print(design)

    df.isna().sum()
    df.corr()

    sns.set(rc={'figure.figsize':(15,8)})
    sns.heatmap(df.corr(),annot=True,cmap="Blues")

    sns.scatterplot(x='rd',y='Profit',data=df)
    sns.scatterplot(x='ad',y='Profit',data=df)
    sns.scatterplot(x='ms',y='Profit',data=df)

    plt.show()

    sns.pairplot(data=df)
    plt.show()

    df.describe().T

    outliers=["Profit"]
    plt.rcParams['figure.figsize']=[8,8]
    sns.boxplot(data=df[outliers],orient="v",palette="Set2",width=0.7)

    plt.title("outliers Variable Distribution")
    plt.ylabel("Profit Range")
    plt.xlabel("Continuous variable")
    plt.show()

    #Divide our data frame into dependent and independent variables
    X=df.drop("Profit",axis=1)
    Y=df["Profit"]

    X.head()
    Y.head()

    #Create four parts,train and test,from these dependent and independent variable
    X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.20)


    #Include LinearRegression in kernel and build the model
    lm=LinearRegression()
    model=lm.fit(X_train,Y_train)

    Y_pred=lm.predict(X_test)


    df_comp=pd.DataFrame({'Actual Values':Y_test,'Estimate':Y_pred})


    MAE=mean_absolute_error(Y_test,Y_pred)
    print("The average magnitute of erroes in a set forecast:",MAE)
    print("")

    MSE = mean_squared_error(Y_test, Y_pred)
    print("The mean squared difference between the estimated values and the actual value:",MSE)
    print("")


    RMSE=math.sqrt(MSE)
    print("The root mean squared error is:",RMSE)
    print("")

    print("R squared value of the model over training data:",model.score(X,Y))
    print("")

    new_data=pd.DataFrame({'rd':rd,'ad':ad,'ms':ms},index=[1])

    Profit=model.predict(new_data)
    return Profit

def main():
    print("---------------------------------Shraddha Patil-------------------------------------------")

    print("Supervised Machine Learning")

    print("Linear Regression on 50_Startup dataset")
    print("")

    print("Enter value of R&D Spend:")
    rd=float(input())

    print("Enter the value of administration:")
    ad=float(input())

    print("Enter the value of marketing spend:")
    ms=float(input())
    print("")

    Profit=Prediction(rd,ad,ms)

    print("**********************************************************************")
    print("Prediction of Profit value in startup:",Profit)
    print("**********************************************************************")

if __name__=="__main__":
    main()


