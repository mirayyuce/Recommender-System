README

Name: EvaS
Authors: Miray Yüce, Olesya Tsapenko
Date: 13.07.2018


(1) Requirements:

1- Python3
2- Pandas 0.20.3 or higher
3- Numpy 1.13.3 or higher
4- Matplotlib 2.1.0 or higher

(1) Package files:

1- utils.py
2- RecS_baseline.py
3- RecS_advanced.py
4- evaluation_metrics.py
5- EvaS.py
6- (Optional) ./Examples/ directory required
7- (Optional) ./Examples/small_ratings.csv
8- (Optional) ./Examples/meta_Electronics_50.json

(2) Instructions:

During execution, the program should be able to create folders and files in local directory.

Input format: 
	python3 EvaS.py <rating_file_path> <meta_file> <rec_s_system> 
    
Example: 
	python3 EvaS.py ./Examples/small_ratings.csv ./Examples/meta_Electronics_50.json baseline

	python3 EvaS.py ./Examples/small_ratings.csv ./Examples/meta_Electronics_50.json advanced

(3) Execution flow:

1- Load items rating and properties files, return the results
2- Prepare userId and itemId arrays, which contain unique ids.
3- Initiate holy_dict, which is the base structure of EvaS.
4- Generate cross validation splits, and fill respective fields in holy_dict.
5- Compute items’ relevance of test set items according to the average user rating from the training set. Later, fill respective fields for relevance in holy_dict.

6- If baseline recommender system is selected, for each dataset pair (training set, test set) do the followings (by checking the fold number)
	a- Generate training set and return it.
	b- Create baseline recommender system instance.
	c- Compute similarity matrix of the training set, and store it in the instance.

7- If advanced recommender system is selected, for each dataset pair (training set, test set) do the followings (by checking the fold number)
	a- Generate a pivot table, and a utility matrix according training set 
	b- Create advanced recommender system instance.
	c- Compute utility matrix, and store it in the instance.
	
8- For each user, item pair of test set, predict ratings
9- Update respective holy_dict entries with predictions of test set items.
10 - Save holy_dict.json

11- Compute top 100 recommendations for each user, and save them.
12- Set relevance of top 100 recommendations. 
13- Calculate Precision@M, Recall@M, MRR@M, nDCG@M, RMSE and MAE for [10, 25, 50, 100] 
14- After computing everything for all dataset pairs, take average of evaluation metrics. Store the results in txt files.
