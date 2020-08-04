import pandas as pd
import numpy as np
import os
import random
import copy
import json
import glob


class RecS_baseline_class():
    """Class for baseline recommender system model.



        Attributes:
            current_df (dataframe): dataframe with ratings
            sim_matrix (Series): similarities for each item from current_df


    """
    def __init__(self, current_df, current_items_array, save_prefix="Test", load_precomputed_matrix=None):
        """
            Args:
                current_df (dataframe): ratings in dataframe format
                current_items_array (array): array with all items itemId's
                utility: items-users utility matrix
                save_prefix (str): (optional) prefix for saving files during the training
                load_precomputed_matrix (str): (optional) path to precomputed pivot_predictions matrix

        """

        self.__save_prefix__ = save_prefix
        self.current_df = current_df
        self.__items_array__ = current_items_array

        # calculate similarity
        if load_precomputed_matrix is None:
            self.__train__()
        else:
            self.sim_matrix = pd.read_csv(load_precomputed_matrix, index_col=None, header=0)

    def __train__(self):
        """ method for calculating whole similarity matrix

        """

        self.__prepare_data__()
        matrix_columns = ['item1', 'item2', 'sim']
        self.sim_matrix = pd.DataFrame(columns=matrix_columns)

        for k, j in enumerate(self.__items_array__):
            self.sim_matrix = self.sim_matrix.append(self.__compute_similarities__(j))

        # clean the space
        self.adjusted_ratings = None
        # save results
        self.sim_matrix.to_csv("./out/" + self.__save_prefix__ + "_similarities.csv")

    def predictRating(self, userId, itemId):
        """
        Method for returning prediction of user's rating for item
        Args:
            userId (str): unique user id
            itemId (str): unique item id
        Returns:
            float: predicted rating

        """
        items_of_user = self.current_df.loc[(self.current_df["userId"] == userId)]

        pred_nom = 0
        pred_denom = 0

        # check whether user already bought this item. if so, don't compute
        if itemId in items_of_user["itemId"].values:
            print("User ", str(userId), " already purchased item ", str(itemId), ". Please choose another item.")
            return 0.0
        else:

            L = self.sim_matrix.loc[(self.sim_matrix["item1"] == itemId)]
            L = L[L['item2'].isin(items_of_user["itemId"].values)]
            L = L.loc[pd.to_numeric(L["sim"]) > 0.0]

            if L.shape[0] == 0:
                # we can't predict anything
                return 0.0

            # sort the similarities first
            L_sorted = L.sort_values(by=["sim"], ascending=False)

            # then extract the most similar ones
            L_new = L_sorted.head(10)

            # compute prediction. don't use mean free ratings, but the original ones
            for l in L_new["item2"]:
                rat_l = \
                    self.current_df.loc[(self.current_df["itemId"] == l) & (self.current_df["userId"] == userId)][
                        "rating"].values[0]

                if rat_l > 0.0:
                    pred_nom += float(L_new.loc[L_new["item2"] == l]["sim"].values[0]) * rat_l
                    pred_denom += float(np.abs(float(L_new.loc[L_new["item2"] == l]["sim"].values[0])))

            pred = pred_nom / pred_denom

        return pred

    def __prepare_data__(self):
        """ method for generating mean free adjusted ratings dataframe

        """
        mean_user_rating = self.current_df.groupby(["userId"],
                                                   as_index=False,
                                                   sort=False).mean().rename(columns={'rating': 'mean_rating'})[
            ['userId',
             'mean_rating']]
        self.adjusted_ratings = pd.merge(self.current_df, mean_user_rating, on='userId', how='left', sort=False)
        self.adjusted_ratings['rating_adjusted'] = self.adjusted_ratings['rating'] - self.adjusted_ratings[
            'mean_rating']

    def __compute_similarities__(self, item):
        """ method for calculating whole similarity between input item and other items in training set

        Args:
            item (str): unique item id
        Returns:
            Series: partially filled similarity matrix

        """
        # prepare titles of columns and sim matrix
        matrix_columns = ['item1', 'item2', 'sim']
        self.sim_matrix = pd.DataFrame(columns=matrix_columns)

        users_who_rated_item = self.adjusted_ratings.loc[self.adjusted_ratings["itemId"] == item]

        distinct_users = np.unique(users_who_rated_item['userId'])

        # each item-item pair, which were purchased together will be stored
        titles = ['userId', 'item1', 'item2', 'rating1', 'rating2']
        record_1_2 = pd.DataFrame(columns=titles)

        # for all users, who bought the item, find all the other items bought together
        for user in distinct_users:
            items_of_user = self.adjusted_ratings.loc[
                (self.adjusted_ratings["userId"] == user) & (self.adjusted_ratings["itemId"] != item)]

            # how our item was rated by this user:
            rating1 = self.adjusted_ratings.loc[
                (self.adjusted_ratings["itemId"] == item) & (self.adjusted_ratings["userId"] == user)][
                "rating_adjusted"].values[0]

            # look at other items that this user bought
            for other_item in items_of_user["itemId"]:
                # how this second item was rated by this user:
                rating2 = self.adjusted_ratings.loc[
                    (self.adjusted_ratings["itemId"] == other_item) & (self.adjusted_ratings["userId"] == user)][
                    "rating_adjusted"].values[0]

                # store everything
                record = pd.Series([user, item, other_item, rating1, rating2], index=titles)
                record_1_2 = record_1_2.append(record, ignore_index=True)

        # a list of all other items
        distinct_others = np.unique(record_1_2['item2'])

        for other in distinct_others:
            # get info of the other item
            paired_1_2 = record_1_2.loc[record_1_2['item2'] == other]

            # prepare the nominator as always
            sim_value_numerator = float((paired_1_2['rating1'] * paired_1_2['rating2']).sum())

            # for denominator we get all the ratings for items to avoid 1.0 similarities
            sim_value_denominator = float(
                np.sqrt(np.square(
                    self.adjusted_ratings.loc[self.adjusted_ratings["itemId"] == item]["rating_adjusted"].values).sum())
                *
                np.sqrt(np.square(self.adjusted_ratings.loc[self.adjusted_ratings["itemId"] == other][
                                      "rating_adjusted"].values).sum()))
            sim_value_denominator = sim_value_denominator if sim_value_denominator != 0 else 1e-8
            
            # adjusted weird cosine similarity
            sim_value = sim_value_numerator / sim_value_denominator

            # get rid of 1.0000000002 and -1.000000002 if they still exist
            if sim_value > 1.0:
                sim_value = 1.0
            if sim_value < -1.0:
                sim_value = -1.0

            # append to sim matrix ['item1', 'item2', 'sim']
            self.sim_matrix = self.sim_matrix.append(pd.Series([item, other, sim_value], index=matrix_columns),
                                                     ignore_index=True)

        return self.sim_matrix

    def __get_neighbors__(self, userId):
        """ method for calculating recommendation candidates for user

        Args:
            userId (str): unique user id
        Returns:
            array: array of items id which are from neighborhood area

        """
        n_arr = []
        items_of_user = self.current_df.loc[(self.current_df["userId"] == userId)]

        for itemId in items_of_user["itemId"].values:

            L = self.sim_matrix.loc[(self.sim_matrix["item1"] == itemId)]

            L = L.loc[pd.to_numeric(L["sim"]) > 0.0]

            L_sorted = L.sort_values(by=["sim"], ascending=False)
            L_new = L_sorted.head(10)

            for l in L_new["item2"]:
                if (l not in n_arr) and (l not in items_of_user["itemId"].values):
                    n_arr.append(l)

        return n_arr

    def predictTopKRecommendations(self, userId, k):
        """
        Method for returning list of best k recommendations for userId
        Args:
            userId (str): unique user id
            k (int): number of items to be recommended
        Returns:
            array: array of triples (itemId, predicted_rating, 0)

        """

        # because it is very time consuming we predict ratings only for "neighboring" items for each user
        # not for full-catalog minus train set
        # more details in the report
        n_arr = self.__get_neighbors__(userId)

        all_predictions = []

        for i, item in enumerate(n_arr):

            predicted_rating = float(self.predictRating(userId, item))

            if predicted_rating > 0.0:
                all_predictions.append([item, predicted_rating, 0])

        sorted_all_predictions = sorted(all_predictions, key=lambda x: (x[1], x[1], x[1]), reverse=True)

        return sorted_all_predictions[:k]
