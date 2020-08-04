import numpy as np
import os
import sys
import json
from RecS_baseline import RecS_baseline_class
from RecS_advanced import RecS_advanced_class
import utils
import evaluation_metrics


# calculate all metrics and save them
def calculate_and_save_metrics(save_path="metrics_results.txt"):
    """calculate_and_save_metrics function calculates all metrics according to the information
        in global holy_dict and top100_predictions_array.

            Args:
                save_path (str): path to the file where all metrics will be saved.

    """
    
    f_out = open(save_path, "w")
    f_out.write("Recommended system: " + str(rec_s_system) + "\n\n")
    M_arr = [10, 25, 50, 100]
    for M in M_arr:
        precision_m_arr = []
        recall_m_arr = []
        mrr_arr = []
        ndcg_arr = []
        for fold, holy_dict2 in enumerate(top100_predictions_array):
            precision_m, recall_m, mrr = evaluation_metrics.calculate_precision_recall_mrr(holy_dict2, M)
            ndcg = evaluation_metrics.calculate_normalized_dcg(holy_dict2, M)
            precision_m_arr.append(precision_m)
            recall_m_arr.append(recall_m)
            mrr_arr.append(mrr)
            ndcg_arr.append(ndcg)

        precision_m = np.mean(precision_m_arr)
        mrr = np.mean(mrr_arr)
        recall_m = np.mean(recall_m_arr)
        ndcg = np.mean(ndcg_arr)
        f_out.write("M = " + str(M) + "\n")
        f_out.write("precision_m = " + str(precision_m) + "\n")
        f_out.write("recall_m = " + str(recall_m) + "\n")
        f_out.write("mrr = " + str(mrr) + "\n")
        f_out.write("ndcg = " + str(ndcg) + "\n")
        f_out.write("*********************\n")

    rmse_arr = []
    abse_arr = []
    for fold in range(1,6):
        rmse, abse = evaluation_metrics.RMSE_and_ABS(holy_dict, fold)
        rmse_arr.append(rmse)
        abse_arr.append(abse)
    rmse = np.mean(rmse_arr)
    abse = np.mean(abse_arr)
    f_out.write("RMSE = " + str(rmse) + "\n")
    f_out.write("MAE = " + str(abse) + "\n")
    f_out.close()
    print("Metrics results can be found here: " + str(save_path))


def help_text_and_exit():
    """help_text_and_exit function prints information how to use the current .py file and call exit()

    """
    print("Input format:")
    print("python3 EvaS.py <rating_file> <meta_file> <rec_s_system> ")
    print("rec_s_system - recommended system (baseline/advanced)")
    print("Example: python3 EvaS.py ./Examples/small_ratings.csv ./Examples/meta_Electronics_50.json baseline")
    exit()


def check_input(arguments_arr):
    """check_input function checks the input formats.

    """
    if len(arguments_arr) < 4:
        help_text_and_exit()
    else:
        input_file_path_ratings = arguments_arr[1]
        if not os.path.isfile(input_file_path_ratings):
            print("The file " + str(input_file_path_ratings) + " does not exist!")
            help_text_and_exit()

        input_file_path_meta = arguments_arr[2]
        if not os.path.isfile(input_file_path_meta):
            print("The file " + str(input_file_path_meta) + " does not exist!")
            help_text_and_exit()

        rec_s_system = arguments_arr[3]
        if rec_s_system != "baseline" and rec_s_system != "advanced":
            print("Incorrect recomended system!")
            print("Recommended system should be baseline or advanced!")
            help_text_and_exit()


if __name__ == "__main__":

    arguments = sys.argv
    # check the input arguments. If it is incorrect - exit
    check_input(arguments)

    # store checked arguments
    input_file_path_ratings = arguments[1]
    input_file_path_meta = arguments[2]
    rec_s_system = arguments[3]

    # folder for saving all usefull files
    if not os.path.exists("./out/"):
        os.makedirs("./out/")

    ####################################################################
    pivot_utility, ratings_df = utils.loadRatings(input_file_path_ratings)
    print("Ratings file is loaded")

    # meta_electronics file is loaded only for it was required in project specifications, we don't use it anyhere
    meta_electronics = utils.loadItemsProperty(input_file_path_meta)
    print("Meta file is loaded")

    # these arrays for ARS matrix
    users_array = np.array(pivot_utility.index)
    items_array = np.array(pivot_utility.columns)

    ####################################################################
    # create and initiate base structure
    holy_dict = utils.fill_users_items_dict(ratings_df, items_array)
    holy_dict = utils.set_folders_cv5(holy_dict)
    holy_dict = utils.set_relevance(holy_dict)
    print("Cross-validation splittings are done")

    json2 = json.dumps(holy_dict)
    splitting_results_path = "./out/" + rec_s_system + "_zero_dataset_holy_dict.json"
    f = open(splitting_results_path,"w")
    f.write(json2)
    f.close()
    print("Splitting results can be found here: " + splitting_results_path)

    # if it's necessary to use precomputed values
    # we can load holy_dict which contains the information about splitting
    #f_in = open("dict_zeros.json","r")
    #holy_dict = json.load(f_in)
    #f_in.close()

    ####################################################################
    # training part
    models = []
    print("Choosen model: " + rec_s_system)

    if rec_s_system == "baseline":

        for fold in range(1,6):

            save_prefix = rec_s_system + "_test_folder" + str(fold)
            # for each split create a dataset (according to the splitting wich is stored in holy_dict)
            # we delete test values from the copy of a dataset which we will provide to the model
            current_df = utils.generate_ratings_df_without_folder_k(ratings_df, holy_dict, fold)

            # for each split create a model
            models.append(RecS_baseline_class(current_df, items_array, save_prefix=save_prefix))

            # or we can load precomputed similarity matrix (according to the splitting wich is stored in holy_dict)
            # if you want to use this option, you need also to load a holy_dict with splittings
            # models.append(RecS_baseline(current_df, items_array, load_precomputed_matrix="similarities_baseline.csv"))

            print("Training for cv" + str(fold) + " is finished!")

    elif rec_s_system == "advanced":

        for fold in range(1,6):

            save_prefix = rec_s_system + "_test_folder" + str(fold)

            # for each split create a dataset (according to the splitting which is stored in holy_dict)
            # we delete test values from the copy of a dataset which we will provide to the model
            pivot_predictions = utils.generate_pivot_without_folder_k(pivot_utility, holy_dict, fold)
            utility = pivot_predictions.values

            # for each split create a model
            models.append(RecS_advanced_class(pivot_predictions, items_array, users_array, utility,
                                              save_prefix=save_prefix))

            # or we can load precomputed pivot table
            # if you want to use this option, you need also to load a holy_dict with splittings
            #models.append(RecS_advanced_class(pivot_predictions, items_array, users_array, utility,
            #                                  load_precomputed_matrix = (save_prefix + "_SGD_predictions.csv")))
            print("Training for cv" + str(fold) + " is finished!")

    ####################################################################
    # predict test items for all folders. Our holy_dict allows us to do it
    for userId, items in holy_dict.items():
        for item in items:
            folder = int(item['folder']) - 1
            item['predicted_rating'] = float(models[folder].predictRating(userId, item['itemId']))

    # right now holy_dict contains all predictions for test items with respect to cross-validation folders
    # save all prediction results
    json2 = json.dumps(holy_dict)
    test_predict_results_path = "./out/" + rec_s_system + "_filled_dataset_holy_dict.json"
    f = open(test_predict_results_path,"w")
    f.write(json2)
    f.close()

    print("Predictions for test sets are done!")
    print("Results for all test sets can be found here: " + str(test_predict_results_path))

    ####################################################################
    # predict top100 recommendations.
    # We always calculate 100 and after that compute different metrics for top10, top25, top50 and top100
    top100_predictions_array = []

    for fold in range(1,6):
        # for each fold we create a dictionary with key = user, values = [#relevant_items, top100_predictions_array]
        holy_dict2 = {}
        for i, userId in enumerate(users_array):

            top_100 = models[fold-1].predictTopKRecommendations(userId, 100)

            # now we will collect items which are relevant for user in test set
            # when we will predict TOP100 recommendations, we will store relevance of predicted items
            # for future metrics
            rel_arr = []
            items = holy_dict[userId]
            for item in items:
                if item["folder"] == fold and item["relevance"] > 0:
                    rel_arr.append(item["itemId"])


            # check the relevance of topK recommedatios
            for item in top_100:
                if item[0] in rel_arr:
                    # we store relevace here
                    item[2] = 1


            holy_dict2[userId] = [len(rel_arr), top_100]

        top100_predictions_array.append(holy_dict2)

    print("Predictions TOP100 are finished!")

    ####################################################################
    # save all prediction results
    for fold, holy_dict2 in enumerate(top100_predictions_array):
        json2 = json.dumps(holy_dict2)
        topk_predict_results_path = "./out/" + rec_s_system + "_top100_folder" + str(fold+1) + "_holy_dict2.json"
        f = open(topk_predict_results_path,"w")
        f.write(json2)
        f.close()
        print("Results for a test set " + str(fold+1) + " can be found here: " + str(topk_predict_results_path))

    ####################################################################
    # calculate all metrics and save them
    metrics_results_path = "./out/" + str(rec_s_system)+"_metrics_results.txt"
    calculate_and_save_metrics(metrics_results_path)




