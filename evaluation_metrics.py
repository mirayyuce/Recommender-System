# calculate Precision@M, recall@M, mrr
import numpy as np


def calculate_precision_recall_mrr(holy_dict2, M):
    """function calculates precision_m, recall_m, mrr according to the information in dictionary

        Args:
            holy_dict2 (dictionary): dictionary with top 100 predictions for all users
            M - parameter how many top items consider in evaluation
        Returns:
            precision_m (float)
            recall_m (float)
            mrr (float)
    """
    precision_m_arr = []
    recall_m_arr = []
    mrr_arr = []

    for user, values in holy_dict2.items():

        total_number_of_relevant_items_to_u = holy_dict2[user][0]
        
        items = holy_dict2[user][1]
        sorted_m_predictions = items[:M]

        # for Precision@M and recall@M
        recommended_items_m_relevant_to_u = 0
        # the position of the first relevant item
        p_u = -1

        for i, item in enumerate(sorted_m_predictions):
            # for Precision@M and recall@M
            relevance = item[2]
            recommended_items_m_relevant_to_u += relevance

            # for MRR
            if p_u == -1 and relevance == 1:
                p_u = i+1

        precision_m_to_one_u = recommended_items_m_relevant_to_u / float(M)  
        precision_m_arr.append(precision_m_to_one_u)

        # recall@M
        if total_number_of_relevant_items_to_u > 0:
            recall_m_to_one_u = recommended_items_m_relevant_to_u / float(total_number_of_relevant_items_to_u)
        else:
            recall_m_to_one_u = 0
        recall_m_arr.append(recall_m_to_one_u)

        # MRR
        if p_u != -1:
            mrr_to_u = 1.0 / p_u     
        else:  # if we didn't find relevant item in top-M
            mrr_to_u = 0.0

        mrr_arr.append(mrr_to_u)

    precision_m = np.mean(np.array(precision_m_arr)) 
    recall_m = np.mean(np.array(recall_m_arr)) 
    mrr = np.mean(np.array(mrr_arr)) 
    
    return precision_m, recall_m, mrr


def calculate_normalized_dcg(holy_dict2, M):
    """function calculates normalized_dcg according to the information in dictionary

        Args:
            holy_dict2 (dictionary): dictionary with top 100 predictions for all users
            M - parameter how many top items consider in evaluation
        Returns:
            n_dcg (float)
    """
    # ideal order 
    i_dcg = 0

    for m in range(1, M+1):
        i_dcg += 1 / float(np.log2(1+m))

    n_dcg = []

    for user, items in holy_dict2.items():
        items = holy_dict2[user][1]

        sorted_m_predictions = items[:M]

        dcg = 0

        for i, item in enumerate(sorted_m_predictions, 1):

            relevance = item[2]

            dcg += float((2 ** relevance) - 1) / float(np.log2(1 + i))

        n_dcg.append([float(dcg) / float(i_dcg)])

    return np.mean(n_dcg)


def RMSE_and_ABS(holy_dict, fold):
    """function calculates RMSE and MAE for test splits and avarage them (according to the information in dictionary)

        Args:
            holy_dict (dictionary): dictionary with keys are users and values are information about items
                        which were rated by this user.
            M - parameter how many top items consider in evaluation
        Returns:
            rmse (float)
            abse (float)

    """
    rmse_error = []
    abs_error = []
    for user, items in holy_dict.items():
        for item in items:
            if item["folder"] == fold:
                
                rmse_error.append(float((item["real_rating"] - item["predicted_rating"]) ** 2))
                abs_error.append(np.abs(item["real_rating"] - item["predicted_rating"]))

    rmse = np.sqrt(np.mean(rmse_error))
    abse = np.mean(abs_error)
        
    return rmse, abse
