if __name__ == '__main__':
    import pandas as pd 
    from sklearn.tree import DecisionTreeClassifier
    from sklearn. ensemble import  BaggingClassifier, AdaBoostClassifier
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import LabelEncoder as lb
    
    # Preprocessig the data 
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
    
    #Accuracy Arrays
    bag_train_acc = []
    bag_test_acc = []
    bag_dev_acc = []
    boost_train_acc = []
    boost_test_acc = []
    bost_test_acc = []
    boost_dev_acc = []
    
    # Bag Size
    bag_size = (10,20,40,60,80,100)
    # Decision Tree depth
    depths_bagging = (1,2,3,5,10)
    depths_boosting = (1,2,3)
    # Nested For loop to traverse therough tree depth and the bag size
    # Bagging Code
    for depth in depths_bagging:
        temp_train = []
        temp_test = []
        temp_dev = []
        for iter in bag_size:
            decisionTree = DecisionTreeClassifier(max_depth = depth)
            decisionTree.fit(X_train,Y_train)
            baging = BaggingClassifier(decisionTree, max_samples= 0.5, max_features = 1.0, n_estimators = iter)
            baging.fit(X_train,Y_train)
            temp_train.append(baging.score(X_train,Y_train))
            temp_test.append(baging.score(X_test,Y_test))
            temp_dev.append(baging.score(X_dev,Y_dev))
        bag_train_acc.append(temp_train)
        bag_test_acc.append(temp_test)
        bag_dev_acc.append(temp_dev)
    file = open('output.txt','wt')
    print("\n Question 6 results", file=file)   
    #Printing the Data: 
    print("\nBagging on Train data",file=file)
    for i in depths_bagging:
        for j in bag_size:
            for x in bag_train_acc:
                for y in x:
                    print('Tree Depth: ',i,'Bag_szie: ',j,'Accuracy: ',y,file=file)
    print("\nBagging on Test data",file=file)
    for i in depths_bagging:
        for j in bag_size:
            for x in bag_test_acc:
                for y in x:
                    print('Tree Depth: ',i,'Bag_szie: ',j,'Accuracy: ',y,file=file)
    print("\nBagging on Dev data",file=file)
    for i in depths_bagging:
        for j in bag_size:
            for x in bag_dev_acc:
                for y in x:
                    print('Tree Depth: ',i,'Bag_szie: ',j,'Accuracy: ',y,file=file)

    
    #Boosting Code
    for depth in depths_boosting:
        temp_train = []
        temp_test = []
        temp_dev = []
        for iter in bag_size:
            decisionTree = DecisionTreeClassifier(max_depth = depth)
            decisionTree.fit(X_train,Y_train)
            bosting = AdaBoostClassifier(decisionTree,n_estimators = iter, learning_rate = 1)
            bosting.fit(X_train,Y_train)
            temp_train.append(bosting.score(X_train,Y_train))
            temp_test.append(bosting.score(X_test,Y_test))
            temp_dev.append(bosting.score(X_dev,Y_dev))
        boost_train_acc.append(temp_train)
        boost_test_acc.append(temp_test)
        boost_dev_acc.append(temp_dev)
        
    print("\nBoosting on Train data",file=file)
    for i in depths_bagging:
        for j in bag_size:
            for x in boost_train_acc:
                for y in x:
                    print('Tree Depth: ',i,'Bag_szie: ',j,'Accuracy: ',y,file=file)
    print("\nBoosting on Test data",file=file)
    for i in depths_bagging:
        for j in bag_size:
            for x in boost_test_acc:
                for y in x:
                    print('Tree Depth: ',i,'Bag_szie: ',j,'Accuracy: ',y,file=file)
    print("\nBoosting on Dev data",file=file)
    for i in depths_bagging:
        for j in bag_size:
            for x in boost_dev_acc:
                for y in x:
                    print('Tree Depth: ',i,'Bag_szie: ',j,'Accuracy: ',y,file=file)
    file.close()
    #Plotting the Graphs for Bagging
    
    plt.xlabel("Tree Depth")
    plt.ylabel("Accuracies")
    plt.title("Bagging Trainig Plot")
    for i in range(len(bag_train_acc)):
        plt.plot(depths_bagging,[pt[i] for pt in bag_train_acc],label = 'Depths %s'%depths_bagging[i])
    plt.legend()
    plt.savefig('BaggingTrainAccuraciesGraph.jpg')
    plt.show()
    plt.close(1)
    
    plt.xlabel("Tree Depth")
    plt.ylabel("Accuracies")
    plt.title("Bagging Testing Plot")
    for i in range(len(bag_test_acc)):
        plt.plot(depths_bagging,[pt[i] for pt in bag_test_acc],label = 'Depths %s'%depths_bagging[i])
    plt.legend()
    plt.savefig('BaggingTestAccuraciesGraph.jpg')
    plt.show()
    plt.close(1)
    
    plt.xlabel("Tree Depth")
    plt.ylabel("Accuracies")
    plt.title("Bagging Dev Plot")
    for i in range(len(bag_dev_acc)):
        plt.plot(depths_bagging,[pt[i] for pt in bag_dev_acc],label = 'Depths %s'%depths_bagging[i])
    plt.legend()
    plt.savefig('BaggingDevAccuraciesGraph.jpg')
    plt.show()
    plt.close(1)
    
     #Plotting the Graphs for Boosting
    
    plt.xlabel("Tree Depth")
    plt.ylabel("Accuracies")
    plt.title("Boosting Trainig Plot")
    for i in range(len(boost_train_acc)):
        plt.plot(depths_boosting,[pt[i] for pt in boost_train_acc],label = 'Depths %s'%depths_boosting[i])
    plt.legend()
    plt.savefig('BoostingTrainAccuraciesGraph.jpg')
    plt.show()
    plt.close(1)
    
    plt.xlabel("Tree Depth")
    plt.ylabel("Accuracies")
    plt.title("Boosting Testing Plot")
    for i in range(len(boost_test_acc)):
        plt.plot(depths_boosting,[pt[i] for pt in boost_test_acc],label = 'Depths %s'%depths_boosting[i])
    plt.legend()
    plt.savefig('BoostingTestAccuraciesGraph.jpg')
    plt.show()
    plt.close(1)
    
    plt.xlabel("Tree Depth")
    plt.ylabel("Accuracies")
    plt.title("Boosting Dev Plot")
    for i in range(len(boost_dev_acc)):
        plt.plot(depths_boosting,[pt[i] for pt in boost_dev_acc],label = 'Depths %s'%depths_boosting[i])
    plt.legend()
    plt.savefig('BoostingDevAccuraciesGraph.jpg')
    plt.show()
    plt.close(1)