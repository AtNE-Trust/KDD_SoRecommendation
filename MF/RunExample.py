import matplotlib.pyplot as plt
import numpy as np
from LoadData import load_rating_data, spilt_rating_dat
from sklearn.model_selection import train_test_split
from ProbabilisticMatrixFactorization import PMF

if __name__ == "__main__":
    file_path = "C:/Users/45016577/Desktop/PMF-master/data/ml-100k/user_review_more_than_5_copy4.dat"
    pmf = PMF()
    pmf.set_params({"num_feat": 35, "epsilon": 75, "_lambda": 0.5, "momentum": 0.8, "maxepoch": 30, "num_batches": 33,
                    "batch_size": 10000})
    ratings = load_rating_data(file_path)
    print(len(np.unique(ratings[:, 0])), len(np.unique(ratings[:, 1])), pmf.num_feat)
    train, test = train_test_split(ratings, test_size=0.01)  # spilt_rating_dat(ratings)
    pmf.fit(train, test)
    
    # Check performance by plotting train and test errors
    '''plt.plot(range(pmf.maxepoch), pmf.rmse_train, marker='o', label='Training Data')
    plt.plot(range(pmf.maxepoch), pmf.rmse_test, marker='v', label='Test Data')
    plt.title('The epinions Dataset Learning Curve')
    plt.xlabel('Number of Epochs')
    plt.ylabel('RMSE')
    plt.legend()
    plt.grid()
    plt.show()
    print("precision_acc,recall_acc:" + str(pmf.topK(test)))
'''