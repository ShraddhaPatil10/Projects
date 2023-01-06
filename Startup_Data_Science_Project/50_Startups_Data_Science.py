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
    print(df.head())    #It returns specified number of rows from top

    df=df.rename(columns={"R&D Spend":"rd","Administration":"ad","Marketing Spend":"ms"},inplace=False)     #rename() fun alter all the labels in dataset
    print(design)
    print(df.head())

    Investment=df.loc[:,["rd","ad","ms"]]       #It will return specified rows and columns given as rd,ad,ms
    Investment.head()

    Investment["Total_Investment"]=Investment.sum(axis=1)   #add all rows of dataframe

    Investment["Total_Investment"]=Investment.mean()        #mean() method returns a series  with mean value of each column

    df["Profit"].mean()

    print(design,"\n")
    print("Information of dataset is given below:")
    print(design)
    df.info()
    print("The shape of dataset is:",df.shape)
    print(design)

    df.isna().sum()     #Return the number of missing values in each column
    df.corr()           #Calculate relationship between each column in your dataset

    sns.set(rc={'figure.figsize':(15,8)})       #seaborn library--->Allows us to specify width and height of a figure in unit-inches
    sns.heatmap(df.corr(),annot=True,cmap="Blues")  #heatmap is a plot of rectangular data as a color-encoded matrix.As parameter it takes 2D dataset

    sns.scatterplot(x='rd',y='Profit',data=df)      #A Scatter plot displays data between two continous data.
    sns.scatterplot(x='ad',y='Profit',data=df)      #It is used to plot data points on horizontal and a vertical axis in the attempt to show how much one variable is affected by another
    sns.scatterplot(x='ms',y='Profit',data=df)

    plt.show()      #It will displays the graph

    sns.pairplot(data=df)   #To plot multiple pairwise distributions in a dataset,you can use the .pairplot() function
    plt.show()

    df.describe().T #This method used for calculating some statistical data like mean,percentile of numerical values of the series or DataFrame

    outliers=["Profit"]
    plt.rcParams['figure.figsize']=[8,8]    #matplotlib----> It defines a runtime configuration(rc) containing the default styles for every plot element you create
    sns.boxplot(data=df[outliers],orient="v",palette="Set2",width=0.7)  #orient="v:Vertical boxplot----->Thats very useful when you want to compare data between two groups

    plt.title("outliers Variable Distribution") #Title of graph
    plt.ylabel("Profit Range")                  #y-axis
    plt.xlabel("Continuous variable")           #x-axis
    plt.show()

    #Divide our data frame into dependent and independent variables
    X=df.drop("Profit",axis=1)      #removes the specified row or column---> axis=1 is vertical axis. if you specify axis=1 you will removing columns
    Y=df["Profit"]

    X.head()
    Y.head()

    #Create four parts,train and test,from these dependent and independent variable
    #This fun return 4 multidimensional list -----> test_size:: This is spliting size
    X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.20)


    #Include LinearRegression in kernel and build the model
    lm=LinearRegression()           #Selection of model
    model=lm.fit(X_train,Y_train)   #Using fit we will build the model

    Y_pred=lm.predict(X_test)       #Using predict test the model

    #Collect the estimates and actual values in data frame
    df_comp=pd.DataFrame({'Actual Values':Y_test,'Estimate':Y_pred})
    print("")

    #Include metrics hosted by sklearn in the kernel and display MAE,MSE,RMSE
    MAE=mean_absolute_error(Y_test,Y_pred)
    print("The average magnitute of erroes in a set forecast:",MAE)
    print("")

    MSE=mean_squared_error(Y_test,Y_pred)
    print("The mean squared difference between the estimated values and the actual value:",MSE)
    print("")


    RMSE=math.sqrt(MSE)
    print("The quality of prediction:",RMSE)
    print("")

    #print R squared value of the model over training data
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


