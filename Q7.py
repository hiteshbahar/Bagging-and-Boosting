import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn. ensemble import  BaggingClassifier, AdaBoostClassifier
from bayes_opt import BayesianOptimization
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder as lb


input_data_train = pd.read_csv("data/income.train.txt",delimiter =',',header = None)
input_data_test = pd.read_csv("data/income.test.txt",delimiter =',',header = None)
input_data_dev = pd.read_csv("data/income.dev.txt",delimiter =',',header = None)

# Combining data 
combined_data = pd.concat([input_data_train, input_data_test,input_data_dev], keys = ['train','test','dev'])


decision_vector = combined_data[[9]]
combined_data1 = combined_data.drop([9], axis =1)

#Binning 
combined_data1[0] = pd.cut(combined_data1[0],
                           bins = [0,20,31,41,61,101],
                           labels = ['Teenagers','Young Adults', 'Adults','Middle Age','Elderly'])
combined_data1[7] = pd.cut(combined_data1[7],
                           bins = [0,20,41,61,81,101],
                           labels = ['Zero to Twewnty','Twenty to Forty', 'Forty to Sixty','Sixty to Eighty','Eighty to Hundred'])

combined_data1 = pd.get_dummies(combined_data1, columns=[0,1,2,3,4,5,6,7,8])

# Extracting Train, Test and Dev data for X from combined data
X_train = combined_data1.xs('train')
X_test = combined_data1.xs('test')
X_dev = combined_data1.xs('dev')


# Extracting Train, Test and Dev data for Y from combined data
Y_train = decision_vector.xs('train')
Y_test = decision_vector.xs('test')
Y_dev = decision_vector.xs('dev')

# Passing data through ILoc to convert panda data-frame to numpy-array

X_train = X_train.iloc[:,:].values
X_test = X_test.iloc[:,:].values
X_dev = X_dev.iloc[:,:].values

# Label encoding on Y values
labelenco = lb()
Y_train = labelenco.fit_transform(Y_train)
Y_test = labelenco.fit_transform(Y_test)
Y_dev = labelenco.fit_transform(Y_dev)

# Replace all the zero's with -1 in Y 

Y_train[Y_train == 0] = -1
Y_test[Y_test == 0] = -1
Y_dev[Y_dev == 0] = -1



def decisionTreee(depth, numberOfbags):
    decisionTre = DecisionTreeClassifier(max_depth = int(depth))
    baggClass = BaggingClassifier(decisionTre,
                            n_estimators=int(numberOfbags), 
                            max_samples= 0.5, 
                            max_features = 1.0)
    baggClass.fit(X_train,Y_train)
    return baggClass.score(X_dev,Y_dev)

def decisionTreee1(depth1, numberOfbags):
    decisionTre1 = DecisionTreeClassifier(max_depth = int(depth1))
    bostClass = AdaBoostClassifier(decisionTre1,
                            n_estimators=int(numberOfbags), 
                            learning_rate=1)
    bostClass.fit(X_train,Y_train)
    return bostClass.score(X_dev,Y_dev)


if __name__ == '__main__':
    gp_params = {"alpha": 1e-5}
    baggBO = BayesianOptimization(
            decisionTreee,
            {'depth' : [1,10],
             'numberOfbags' : (10,100)
             })
    baggBO.maximize(n_iter = 50, **gp_params)
    file = open('output.txt','a+')
    print("\n Question 7 results", file=file)
    print('Results - Bagging',file=file)
    print('-' * 53,file=file)
    print('Dtree: %f' % baggBO.res['max']['max_val'],file=file)
    baggBOAcc = baggBO.Y
    remmove = 5
    baggBOAcc = baggBOAcc[remmove:]
    
    for i in range(len(baggBO.X)):
        print('Depth=',baggBO.X[i][0],'\t n-estimator=',baggBO.X[i][1],file=file)
    
    iter = list(range(1,51))
    
    
    bostBO = BayesianOptimization(
            decisionTreee,
            {'depth' : [1,3],
             'numberOfbags' : (10,100)
             })
    bostBO.maximize(n_iter = 50, **gp_params)
    print('Results - Boosting',file=file)
    print('-' * 53,file=file)
    print('Dtree: %f' % bostBO.res['max']['max_val'], file=file)
    bostBOAcc = bostBO.Y
    remmove = 5
    bostBOAcc = bostBOAcc[remmove:]
    iter = list(range(1,51))
    
    for i in range(len(bostBO.X)):
        print('Depth=',bostBO.X[i][0],'\t n-estimator=',bostBO.X[i][1],file=file)
    file.close()
    #Plotting
    plt.xlabel("Iterations")
    plt.ylabel("Accuracies")
    plt.title("Bayesian Optimization Dev Plot - Boosting")
    plt.plot(iter,bostBOAcc)
    plt.savefig('Bayesian_Optimization_Dev_Plot_Boosting.jpg')
    plt.show()
    plt.close(1)
    
    
    #Plotting
    plt.xlabel("Iterations")
    plt.ylabel("Accuracies")
    plt.title("Bayesian Optimization Dev Plot - Bagging")
    plt.plot(iter,baggBOAcc)
    plt.savefig('Bayesian_Optimization_Dev_Plot_Bagging.jpg')
    plt.show()
    plt.close(1)
    