import pandas as pd
import numpy as np
import os
import random
import copy
import json
import glob
import utils
import matplotlib.pyplot as plt


class RecS_advanced_class():
    """Class for advanced recommender system model.



    Attributes:
        pivot_predictions (dataframe): dataframe with ratings


    """
    def __init__(self, pivot_predictions, items_array, users_array, utility, save_prefix="Test", load_precomputed_matrix=None):
        """
            Args:
                pivot_predictions (dataframe): ratings in dataframe format
                items_array (array): array with all items itemId's
                users_array (array): array with all users userId's
                utility: items-users utility matrix
                save_prefix (str): (optional) prefix for saving files during the training
                load_precomputed_matrix (str): (optional) path to precomputed pivot_predictions matrix

        """
        self.__save_prefix__ = save_prefix
        self.pivot_predictions = pivot_predictions
        self.__items_array__ = items_array
        self.__users_array__ = users_array
        self.__utility__ = utility

        if load_precomputed_matrix is None:
            self.__train__()
        else:
            print("load pivot_predictions")
            self.pivot_predictions = pd.read_csv(load_precomputed_matrix, index_col="userId",
                                                 delimiter=",", decimal=".", header=0)

    def __train__(self):
        """ method for calculating whole utility matrix using U-V matrix factorization and biased SGD

        """

        alpha, l, U, V, bias_u, bias_v, global_bias, epochs = self.__prepare_data__()
        # calculate matrix factorization and predictions

        losses = []
        for e in range(epochs):
            loss = 0

            for j, user_ratings in enumerate(self.__utility__):
                for i, item_ratings in enumerate(self.__utility__.T):

                    if self.__utility__[j, i] != 0:
                        prediction = global_bias + bias_u[j] + bias_v[i] + U[j].dot(V[i])
                        
                        error = (self.__utility__[j, i] - prediction)
                        squared_error = error ** 2
                        loss += squared_error
                        
                        bias_u[j] = alpha * (error - l * bias_u[j])
                        bias_v[i] = alpha * (error - l * bias_v[i])
                        
                        temp_u = U[j] + alpha * (2 * error * V[i] - l * U[j])
                        U[j] = temp_u

                        temp_v = V[i] + alpha * (2 * error * U[j] - l * V[i])
                        V[i] = temp_v

            losses.append(loss)

        predictions = U.dot(V.T)

        # set predictions back to the pivot table
        for u, user in enumerate(self.__users_array__):
            for i, item in enumerate(self.__items_array__):
                if self.pivot_predictions.values[u][i] == 0:
                    self.pivot_predictions.values[u][i] = predictions[u][i]
                else:
                    # for training items we set "-1" in order to easily ignore them for topK predictions
                    self.pivot_predictions.values[u][i] = -1

        # we save a plot with a loss function
        if not os.path.exists("./Plots/"):
            os.makedirs("./Plots/")

        fig = plt.figure()
        plt.plot(losses)
        plt.savefig("./Plots/" + self.__save_prefix__ + ".png")
        plt.close(fig)

        # clean the space
        self.__utility__ = None
        # save results
        self.pivot_predictions.to_csv("./out/" + self.__save_prefix__ + "_SGD_predictions.csv")

    def predictRating(self, userId, itemId):
        """
        Method for returning prediction of user's rating for item
        Args:
            userId (str): unique user id
            itemId (str): unique item id
        Returns:
            float: predicted rating

        """

        # add try-catch if we dont't have user/item...
        try:
            return self.pivot_predictions.at[userId,itemId]
        except:
            print("Advanced recommender system can't predict value for this pair user: " + str(userId) + "; item: "
                  + str(itemId))
            return 0.0

    def predictTopKRecommendations(self, userId, k):
        """
        Method for returning list of best k recommendations for userId
        Args:
            userId (str): unique user id
            k (int): number of items to be recommended
        Returns:
            array: array of triples (itemId, predicted_rating, 0)

        """
        topKarr = []
        p = self.pivot_predictions.loc[userId].sort_values(ascending=False)

        for ip in p[:k].iteritems():
            # add checking that prediction != -1.0 (check if it's in train set)
            if float(ip[1]) > -1.0:
                topKarr.append([ip[0], ip[1], 0])
        return topKarr

    def __prepare_data__(self):
        """ method for initializing required variables and setting hyperparameters

        """

        # hyperparameters
        alpha = 0.03
        l = 0.009
        rank = 16
        epochs = 20
        np.random.seed(6)

        # variables
        U = np.random.random((len(self.__users_array__),rank))
        V = np.random.random((len(self.__items_array__),rank))

        bias_u = np.zeros((len(self.__users_array__), 1))
        bias_v = np.zeros((len(self.__items_array__), 1))
        
        global_bias = self.__utility__.mean()

        return alpha, l, U, V, bias_u, bias_v, global_bias, epochs
