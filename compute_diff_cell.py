# nohup python -u compute_diff_cell.py --distance_threshold 0.2 --cell_num 5000 --method cellhint --emb geneformer --dataset Lung --allocation_flag True > compute_diff_cell.log 2>&1 &
# nohup python -u compute_diff_cell.py --distance_threshold 0.7 --cell_num 3000 --method cosine --emb emb_langcell --dataset PBMC --allocation_flag True > compute_diff_cell.log 2>&1 &

# nohup python -u compute_diff_cell.py --cell_num 3000 --dataset PBMC > compute_diff_cell.log 2>&1 &
# nohup python -u compute_diff_cell.py --method cosine --distance_threshold 0.6 --emb emb_langcell --dataset PBMC_10K --allocation_flag > compute_diff_cell.log 2>&1 &

# nohup python -u compute_diff_cell.py --dataset PBMC_10K > get_similarity.log 2>&1 &
# nohup python -u compute_diff_cell.py --dataset PBMC > get_similarity.log 2>&1 &


# nohup python -u compute_diff_cell.py --dataset Blood > get_similarity.log 2>&1 &
# nohup python -u compute_diff_cell.py --distance_threshold 0.4 --cell_num 3000 --method euclidean --emb emb_pooling_langcell --dataset Blood --allocation_flag > compute_diff_cell.log 2>&1 &


# nohup python -u compute_diff_cell.py --dataset Blood_all > get_similarity.log 2>&1 &
# nohup python -u compute_diff_cell.py --distance_threshold 0.4 --cell_num 3000 --method euclidean --emb emb_pooling_langcell --dataset Blood_all --allocation_flag > compute_diff_cell.log 2>&1 &
# nohup python -u compute_diff_cell.py --distance_threshold 0.6 --cell_num 3000 --method euclidean --emb emb_langcell --dataset Blood_all --allocation_flag > compute_diff_cell.log 2>&1 &

import pandas as pd
import numpy as np
import os
import pickle
import ast
import argparse
import os
import time
from datetime import datetime

from utils import calculate_distance, get_similarity_matrix

import warnings
warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser(description='Process some thresholds.')
parser.add_argument('--distance_threshold', type=float, default=0.6)
parser.add_argument('--cell_num', type=int, default=5000)

parser.add_argument('--method', type=str, default="euclidean") # cosine, euclidean, Manhattan, Chebyshev, Hamming, Pearson
parser.add_argument('--emb', type=str, default="emb_pooling_langcell") # 'emb_geneformer', 'emb_langcell', 'emb_pooling_geneformer', 'emb_pooling_langcell'
parser.add_argument('--dataset', type=str, default="Lung")
parser.add_argument('--mode', type=str, default="stage_1")

parser.add_argument('--allocation_flag', action='store_true', default=False)

parser.add_argument('--max_workers', type=int, default=10)

args = parser.parse_args()

if args.allocation_flag:
    print(f"Args method: {args.method}; emb: {args.emb}; distance_threshold: {args.distance_threshold}; Cell_num: {args.cell_num}; Dataset: {args.dataset}; max_workers: {args.max_workers}")
else:
    print(f"allocation_flag: {args.allocation_flag}; Cell_num: {args.cell_num}; Dataset: {args.dataset}; max_workers: {args.max_workers}")


def store_sum_and_count(row):
    return [row['sum'], row['count']]


def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Creating {path}")
    else:
        print(f"{path} already exists")
   
print("Loading data...")

if args.mode == "stage_1":
    read_data_dir = "processed_data"
else:
    read_data_dir = "comp_data"

orchestra_cell = pd.read_csv(f"{read_data_dir}/{args.dataset}/orchestra_cell.csv")
# orchestra_cell = orchestra_cell.iloc[:20]
comp_orchestra = pd.read_csv(f"{read_data_dir}/{args.dataset}/comp_orchestra.csv")

make_dir(f"result/{args.dataset}")
make_dir(f"result/{args.dataset}/{args.method}_{args.emb}")
make_dir(f"result/{args.dataset}/{args.method}_{args.emb}/matching")
make_dir(f"result/{args.dataset}/{args.method}_{args.emb}/similarity_matrix")

# 
# 初始化一个字典来存储所有的mapping文件
matching_dict = {}
for i in range(len(comp_orchestra)):
    select_index =  ast.literal_eval(comp_orchestra['Sample2'][i])
    # print(i,select_index)
    mapping = pd.DataFrame(0, index = range(comp_orchestra['cell_num'][i]) ,
                           columns = orchestra_cell['orchestra'][select_index]
                           )
    num = comp_orchestra['Sample1'][i]
    matching_dict[f'mapping_{num}'] = mapping

s_time = time.time()
print("Satrt time:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


# _, _, cost, matching, all_index_1, all_index_2,save_index = calculate_distance(0, 1, method = args.method,emb = args.emb,distance_threshold = args.distance_threshold, dataset = args.dataset)
# exit()

import concurrent.futures

# 使用进程池并发执行计算
with concurrent.futures.ProcessPoolExecutor(max_workers=args.max_workers) as executor:
    if args.allocation_flag:
        futures = {
            executor.submit(
                calculate_distance, i, j, dataset_1 = args.dataset, dataset_2 = args.dataset,method = args.method,emb = args.emb,distance_threshold = args.distance_threshold, read_data_dir = read_data_dir): (i, j) for i in comp_orchestra[comp_orchestra['disease'] != "normal"]['Sample1'] for j in ast.literal_eval(comp_orchestra.loc[comp_orchestra['Sample1'] == i,'Sample2'].values[0])
                    }

        # 处理每个任务的结果
        for future in concurrent.futures.as_completed(futures):
            i, j = futures[future]
            try:
                _, _, cost, matching, all_index_1, all_index_2,save_index = future.result()
               
                matching_dict[f'mapping_{i}'].index = all_index_1
                matching_dict[f'mapping_{j}'].index = all_index_2
                
                if cost is not None:

                    matching_dict[f'mapping_{i}'].loc[matching['Left_Node'].values, orchestra_cell['orchestra'][j]] = matching['Right_Node'].values
                    matching_dict[f'mapping_{i}'].to_csv(f"result/{args.dataset}/{args.method}_{args.emb}/matching/{i}.csv")
                    matching_dict[f'mapping_{j}'].loc[matching['Right_Node'].values, orchestra_cell['orchestra'][i]] = matching['Left_Node'].values
                    matching_dict[f'mapping_{j}'].to_csv(f"result/{args.dataset}/{args.method}_{args.emb}/matching/{j}.csv")

                else:
                    print(f"cost is None: {i}, {j}")
                
                with open(f"result/{args.dataset}/{args.method}_{args.emb}/matching/matching_dict_{args.emb}_{args.distance_threshold}.pkl", 'wb') as f:
                    pickle.dump(matching_dict, f)
                print(f"Task ({i}, {j}) finished.")
            except Exception as e:
                print(f"Task ({i}, {j}) generated an exception: {e}")
    else:
        futures = {
            executor.submit(get_similarity_matrix, i, j, method, emb, dataset_1 = args.dataset,dataset_2 = args.dataset,save = True,read_data_dir = read_data_dir): (i, j, method, emb) for i in comp_orchestra[comp_orchestra['disease'] != "normal"]['Sample1']  for j in ast.literal_eval(comp_orchestra.loc[comp_orchestra['Sample1'] == i,'Sample2'].values[0]) for method in ['cosine', 'euclidean'] for emb in ['emb_langcell','emb_pooling_langcell','emb_pooling_geneformer','emb_geneformer','X_pca','X_umap']
                    }
        for future in concurrent.futures.as_completed(futures):
            i, j, method, emb = futures[future]
            _, _, similarity_matrix = future.result()
            print(f"Task ({i}, {j}, {method}, {emb}) finished.")

f_time = time.time()
elapsed_time = f_time - s_time
# 转换为小时、分钟和秒
hours, rem = divmod(elapsed_time, 3600)
minutes, seconds = divmod(rem, 60)

# print(distances)
print("Finish time: ",datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
print(f"All finished. Read in data ran for {int(hours)} hours, {int(minutes)} minutes and {seconds:.2f} seconds.\n")