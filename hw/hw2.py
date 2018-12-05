import numpy as np
import matplotlib.pyplot as plt
import cvxopt
from models import LinearRegression, LogisticRegression, SVM
from elice_utils import EliceUtils

elice_utils = EliceUtils()


def main():
    """
    This is the main function. Use this area as you like.
    Below is an example of a simple workflow. Get the data, train the model,
    predict based on the model, then evaluate the performance.
    The code is just for demonstration, you can just erase it and make your own.
    """
    import numpy as np
    from models import LinearRegression, LogisticRegression, SVM

    # Synthesize some data for linear regression.
    f = open('./data/linreg_train.txt')
    lines = f.readlines()
    X = []
    y = []
    for line in lines:
        tmp_line = list(map(float, line.strip().split()))
        X.append(tmp_line[1:])
        y.append(tmp_line[:1])
        
    # y = aX + b
    X = np.asarray(X) # input_data
    y = np.asarray(y) # real_value
    X_train = X[:300,:]
    y_train = y[:300]
    X_test = X[300:,:]
    y_test = y[300:]
    # Make a model and train it.
    model = LinearRegression()
    mse_train = model.fit(X_train, y_train)
    # Predict some target values and evaluate the performance.
    y_pred = model.predict(X_test)
    error = y_pred - y_test
    mse = np.dot(np.transpose(error), error)/len(y_test)
    print("linear regression " + str(mse[0][0]))
    f.close()
    
    g = open('./data/logreg_10d_train.txt')
    lines = g.readlines()
    X = []
    y = []
    for line in lines:
        tmp_line = list(map(float, line.strip().split()))
        X.append(tmp_line[1:])
        y.append(tmp_line[:1])
        
    X = np.asarray(X) # input_data
    y = np.asarray(y) # real_value
    X_train_list = []
    y_train_list = []
    X_train_1 = X[:200,:]
    y_train_1 = y[:200].reshape(200)
    X_train_list.append(X_train_1)
    y_train_list.append(y_train_1)
    X_train_2 = X[200:400,:]
    y_train_2 = y[200:400].reshape(200)
    X_train_list.append(X_train_2)
    y_train_list.append(y_train_2)
    X_train_3 = X[400:600,:]
    y_train_3 = y[400:600].reshape(200)
    X_train_list.append(X_train_3)
    y_train_list.append(y_train_3)
    X_train_4 = X[600:800,:]
    y_train_4 = y[600:800].reshape(200)
    X_train_list.append(X_train_4)
    y_train_list.append(y_train_4)
    X_train_5 = X[800:1000,:]
    y_train_5 = y[800:1000].reshape(200)
    X_train_list.append(X_train_5)
    y_train_list.append(y_train_5)
    
    # Cross validation
    accuracy = []
    for i in range(5):
        valid_X = X_train_list[i]
        valid_y = y_train_list[i]
        train_X = np.zeros((800, 10))
        train_y = np.zeros(800)
        cnt = 0
        for j in range(5):
            if j != i:
                train_X[cnt:cnt + 200] = X_train_list[j]
                train_y[cnt:cnt + 200] = y_train_list[j]
                cnt = cnt + 200
        model = LogisticRegression(0.05, 100)
        model.fit(train_X, train_y)
        y_pred = model.predict(valid_X)
        error = y_pred - np.transpose(valid_y)
        accuracy.append(1 - (np.dot(error, np.transpose(error))/len(valid_y)))
    print("logistic regression " + str(sum(accuracy)/len(accuracy)))
    g.close()
    
    h = open('./data/svm_10d_train.txt')
    lines = h.readlines()
    X = []
    y = []
    for line in lines:
        tmp_line = list(map(float, line.strip().split()))
        X.append(tmp_line[1:])
        y.append(tmp_line[:1])
        
    X = np.asarray(X) # input_data
    y = np.asarray(y) # real_value
    X_train_list = []
    y_train_list = []
    X_train_1 = X[:100,:]
    y_train_1 = y[:100].reshape(100)
    X_train_list.append(X_train_1)
    y_train_list.append(y_train_1)
    X_train_2 = X[100:200,:]
    y_train_2 = y[100:200].reshape(100)
    X_train_list.append(X_train_2)
    y_train_list.append(y_train_2)
    X_train_3 = X[200:300,:]
    y_train_3 = y[200:300].reshape(100)
    X_train_list.append(X_train_3)
    y_train_list.append(y_train_3)
    X_train_4 = X[300:400,:]
    y_train_4 = y[300:400].reshape(100)
    X_train_list.append(X_train_4)
    y_train_list.append(y_train_4)
    X_train_5 = X[400:500,:]
    y_train_5 = y[400:500].reshape(100)
    X_train_list.append(X_train_5)
    y_train_list.append(y_train_5)
    
    # Cross validation
    accuracy = []
    for i in range(5):
        valid_X = X_train_list[i]
        valid_y = y_train_list[i]
        train_X = np.zeros((400, 10))
        train_y = np.zeros(400)
        cnt = 0
        for j in range(5):
            if j != i:
                train_X[cnt:cnt + 100] = X_train_list[j]
                train_y[cnt:cnt + 100] = y_train_list[j]
                cnt = cnt + 100
        model = SVM(2, 1.75)
        model.fit(train_X, train_y)
        y_pred = model.predict(valid_X)
        error = y_pred - np.transpose(valid_y)
        accuracy.append(1 - (np.dot(error, np.transpose(error))/len(valid_y)))
    print("svm " + str(sum(accuracy)/len(accuracy)))
    h.close()


if __name__ == "__main__":
    main()
