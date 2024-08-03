import pandas as pd
import numpy as np

import scanpy as sc
import cellhint

import networkx as nx
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import cosine_similarity

import ast
import os
import time

def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Creating {path}")
    else:
        print(f"{path} already exists")

def transfer_to_dummies(df):
    # 使用 ast.literal_eval 将字符串转换为列表，然后使用 get_dummies 函数将 "col" 列转换为一个 0/1 矩阵
    # df.to_csv("df.csv")
    dummies = df.apply(lambda x: pd.Series([1] * len(ast.literal_eval(x)), index=ast.literal_eval(x))).fillna(0, downcast='infer')

    # 获取所有可能的列名
    all_columns = pd.Index(sorted(set(x for l in df for x in ast.literal_eval(l))))

    # 使用 reindex 函数添加缺失的列
    dummies = dummies.reindex(columns=all_columns, fill_value=0)

    return dummies

def adata_pca(adata):
    sc.pp.normalize_total(adata, target_sum=1e4)  # 归一化总基因表达为1e4
    sc.pp.log1p(adata)  # 对数转换，log1p表示log(x+1)，确保0的对数转换不会是负无穷
    sc.pp.scale(adata, max_value=10)  # 缩放数据，使均值为0，方差为1
    sc.tl.pca(adata, svd_solver='arpack') 
    return adata

def adata_umap(adata,  n_neighbors=10, n_pcs=40):
    adata = adata_pca(adata)
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs)
    sc.tl.umap(adata) 
    return adata

def construct_B_graph(transfer_matrix, duplicates):
    B = nx.Graph()
    # transfer_matrix.to_csv("transfer_matrix.csv")
    # 获取矩阵的行和列名称
    row_names = transfer_matrix.index
    col_names = transfer_matrix.columns

    # 添加行节点
    for row_name in row_names:
        if transfer_matrix.loc[row_name].any():  # 如果该行有任何非零元素
            B.add_node(row_name, bipartite=0)

    # 添加列节点，考虑重复次数
    for j, col_name in enumerate(col_names):
        if transfer_matrix[col_name].any():  # 如果该列有任何非零元素
            for i, row_name in enumerate(row_names):
                if transfer_matrix.iloc[i, j] != 0:
                    if duplicates is None:
                        B.add_node(col_name, bipartite=1)
                        B.add_edge(row_name, col_name)
                    else:
                        num_duplicates = duplicates.get(j, 1)  # 获取重复次数，默认为1
                        for dup in range(num_duplicates):
                            col_node = f"{col_name}_{dup}"  # 为每个重复创建独立的节点
                            B.add_node(col_node, bipartite=1)
                            B.add_edge(row_name, col_node)
    # 删除没有任何边的节点
    B.remove_nodes_from(list(nx.isolates(B)))
    return B

def maximum_matching_for_each_component(B):
    components = nx.connected_components(B)
    matching = {}

    for component in components:
        subgraph = B.subgraph(component)
        component_matching = nx.bipartite.maximum_matching(subgraph)
        matching.update(component_matching)

    return matching

def max_matching_from_transfer_matrix(transfer_matrix, duplicates):
    
    B = construct_B_graph(transfer_matrix, duplicates)
    matching = maximum_matching_for_each_component(B)
    # 过滤匹配结果，只保留从左侧到右侧的匹配
    matching = {k: v for k, v in matching.items() if B.nodes[k]['bipartite'] == 0}

    matched_rows = len([node for node in matching.keys() if B.nodes[node]['bipartite'] == 0])
    cost = 1 - matched_rows / len(transfer_matrix)

    # 转换为DataFrame
    matching = pd.DataFrame(list(matching.items()), columns=['Left_Node', 'Right_Node'])

    return cost, B, matching


def normalize_excluding_self(distance_matrix, group, axis):
    normalized_matrix = distance_matrix.copy()
    for cell_type in group.unique():
        mask = group == cell_type
        if axis == 0: # meta_1['cell_type] (行)
            mean = distance_matrix.loc[~mask, :].mean(axis=0)
            std = distance_matrix.loc[~mask, :].std(axis=0)
            normalized_matrix.loc[mask, :] = (distance_matrix.loc[mask, :] - mean[None, :]) / std[None, :]
        else:
            mean = distance_matrix.loc[:, ~mask].mean(axis=1)
            std = distance_matrix.loc[:, ~mask].std(axis=1)
            normalized_matrix.loc[:, mask] = (distance_matrix.loc[:, mask] - mean[:, None]) / std[:, None]
            
    return normalized_matrix


def get_mean_std(distance_matrix, group, axis, sample_size=None, exclude_group=None):
    if sample_size is None:
        sample_size = round(3000/len(group.unique()))
    sampled_distance_matrix = pd.DataFrame()
    for cell_type in group.unique():
        if cell_type == exclude_group:
            continue
        if axis == 0: # meta_1['cell_type] (行)
            samples = distance_matrix.loc[group == cell_type, :].sample(n=sample_size, axis=0, replace=True, random_state=42)
            sampled_distance_matrix = pd.concat([sampled_distance_matrix, samples], axis=0)
        else:
            samples = distance_matrix.loc[:, group == cell_type].sample(n=sample_size, axis=1, replace=True, random_state=42)
            sampled_distance_matrix = pd.concat([sampled_distance_matrix, samples], axis=1)
    mean = sampled_distance_matrix.mean(axis=axis)
    std = sampled_distance_matrix.std(axis=axis)
    return mean, std

def normalize_distance_matrix(distance_matrix, group, axis, sample_size=None):
    mean, std = get_mean_std(distance_matrix, group, axis, sample_size=sample_size)
    
    if axis == 0: # meta_1['cell_type] (行)
        normalized_matrix = (distance_matrix - mean[None, :]) / std[None, :]
    else:
        normalized_matrix = (distance_matrix - mean[:, None]) / std[:, None]
    return normalized_matrix


# def normalize_distance_matrix(distance_matrix, group, axis, sample_size=None):
    # normalized_matrix = distance_matrix.copy()
    # for cell_type in group.unique():
    #     mask = group == cell_type
    #     mean, std = get_mean_std(distance_matrix, group, axis, sample_size=sample_size,exclude_group = cell_type)

    #     if axis == 0: # meta_1['cell_type] (行)
    #         mean = distance_matrix.loc[~mask, :].mean(axis=0)
    #         std = distance_matrix.loc[~mask, :].std(axis=0)
    #         normalized_matrix.loc[mask, :] = (distance_matrix.loc[mask, :] - mean[None, :]) / std[None, :]
    #     else:
    #         mean = distance_matrix.loc[:, ~mask].mean(axis=1)
    #         std = distance_matrix.loc[:, ~mask].std(axis=1)
    #         normalized_matrix.loc[:, mask] = (distance_matrix.loc[:, mask] - mean[:, None]) / std[:, None]
            
    # return normalized_matrix

def get_num(m,num=1):
    indices_3 = np.where(m.values == num)

    # 根据索引获取对应的行列名，并存到一个 DataFrame 中
    matched_CT = pd.DataFrame({'Row': m.index[indices_3[0]].astype(str), 'Col': m.columns[indices_3[1]].astype(str)
            })

    # matched_CT['Com'] = matched_CT['Row'] + '_' + matched_CT['Col']
    a = pd.DataFrame(matched_CT.groupby(['Row','Col']).size(),columns = ['num']).reset_index()
    return a

def get_matched_cells(distance_matrix,matched_CT,sort_num = 100,set_value = 1,transpose_flag = False):
    # 初始化结果列表
    if transpose_flag:
        distance_matrix = distance_matrix.T
    result = []

    visited_matrix = pd.DataFrame(0, index=distance_matrix.index, columns=distance_matrix.columns)
    dm_copy = distance_matrix.copy()
    dm_copy.columns = range(len(dm_copy.columns))
    # 遍历每一行
    for i in range(len(distance_matrix)):
        # 对当前行进行排序
        sorted_row = dm_copy.iloc[i].sort_values()[:sort_num]
        
        # 获取当前行的名称
        row_name = distance_matrix.index[i]
        if row_name not in matched_CT['Row Name'].values and row_name != "unknown":
            m_ct = ['no']
        else:
            m_ct = matched_CT[matched_CT['Row Name'] == row_name]['Column Name']
            if type(m_ct) == str:
                m_ct = [m_ct]
            else:
                m_ct = m_ct.values
        
        cell_set = set()
        wrong_num = 0
        right_num = 0
        for j, value in sorted_row.iteritems():
            column_name = distance_matrix.columns[j]
            cell_set.add(column_name)
            if row_name == "unknown":
                visited_matrix.iloc[i, j] = set_value
                if len(cell_set) == 5:
                    result.append([i,row_name, column_name, value])
                    break
            else:
                if column_name not in m_ct:
                    wrong_num = wrong_num+1
                else:
                    right_num = right_num+1 
                    visited_matrix.iloc[i, j] = set_value
                if wrong_num == 5 or len(cell_set) == 5:
                    ratio = right_num / (right_num + wrong_num)
                    result.append([i,row_name, column_name, value,ratio])
                    break
        else:
            if row_name == "unknown":
                result.append([i,row_name, column_name, value])
            else:
                ratio = right_num / (right_num + wrong_num)
                result.append([i,row_name, column_name, value,ratio])
    
    result = pd.DataFrame(result, columns=["Num",'Row Name', 'Column Name', 'Value',"Ratio"])
    
    if transpose_flag:
        visited_matrix = visited_matrix.T
    return visited_matrix, result   


def get_transfer_matrix(i,j,expr_data1,expr_data2,meta_1,meta_2,method = "euclidean",emb = "emb_langcell"):
    CT_TS = -0.4 #底线，防止有些细胞（行）完全不匹配另一个数据集的细胞（列），一行的数值都很小，那么选取前几个也有噪音

    dm = pd.DataFrame(cdist(expr_data1, expr_data2, 'euclidean'))
    dm = dm / dm.max().max()

    dm.columns = meta_2['cell_type']
    dm.index = meta_1['cell_type']

    dm1 = normalize_distance_matrix(dm,group = meta_1['cell_type'].values, axis = 0)
    dm2 = normalize_distance_matrix(dm,group = meta_2['cell_type'].values, axis = 1)

    Ave_1 = dm1.groupby(dm1.columns, axis=1).mean().groupby(dm1.index, axis=0).mean()
    Ave_2 = dm2.groupby(dm2.columns, axis=1).mean().groupby(dm2.index, axis=0).mean()

    CT_1 = pd.DataFrame(0,index = Ave_1.index, columns = Ave_1.columns)
    CT_2 = pd.DataFrame(0,index = Ave_2.index, columns = Ave_2.columns)

    top_n_1 = max( round(Ave_1.shape[1] / 10), 2)
    top_n_2 = max( round(Ave_1.shape[0] / 10), 2)

    indices_1 = Ave_1.apply(lambda row: row.nsmallest( top_n_1 ).index, axis=1)
    for index, values in indices_1.iteritems():
        CT_1.loc[index, values] = 1
    CT_1[ Ave_1 > CT_TS ] = 0

    indices_2 = Ave_2.apply(lambda col: col.nsmallest( top_n_2 ).index, axis=0)
    for index, values in indices_2.iteritems():
        CT_2.loc[ values, index ] = 2
    CT_2[ Ave_2 > CT_TS ] = 0
    CT = CT_1 + CT_2

    # 遍历 DataFrame 的每个元素
    for row in CT.index:
        for col in CT.columns:
            # 如果行名和列名相同且不是 "unknown"，则赋值为3
            if row == col and row != 'unknown':
                CT.loc[row, col] = 3

    # 找到数值等于3的位置的索引
    indices_3 = np.where(CT.values == 3)

    # 根据索引获取对应的行列名，并存到一个 DataFrame 中
    matched_CT = pd.DataFrame({'Row Name': CT.index[indices_3[0]], 'Column Name': CT.columns[indices_3[1]]})

    matched_CT['Row Name'] = matched_CT['Row Name'].astype(str)
    matched_CT['Column Name'] = matched_CT['Column Name'].astype(str)
    matched_CT['Same'] = matched_CT['Row Name'] == matched_CT['Column Name']
    matched_CT = matched_CT.sort_values(by='Same', ascending=False)

    mc_1 = matched_CT
    mc_2 = mc_1.copy()
    mc_2.columns = ['Column Name','Row Name','Same']

    vm_1,df_1 = get_matched_cells(dm1, mc_1,sort_num = 100,set_value = 1,transpose_flag = False)
    vm_2,df_2 = get_matched_cells(dm2, mc_2,sort_num = 100,set_value = 1,transpose_flag = True)
    vm = vm_1 | vm_2
    
    cp_per = round(vm.sum(axis=0).sum()/3000/3000 *100, 2)
    a = get_num(vm,num=1) # 哪些CT1和CT2的匹配被选出来了，方便统计看
    return vm,top_n_1,top_n_2, cp_per, a


def cal_sc_distance(i,j,adata1,adata2,method = "cosine",emb = "geneformer",distance_threshold = 0.8):
    expr_data1 = adata1.obsm[emb]
    expr_data2 = adata2.obsm[emb]
    # 确保数据是numpy数组
    expr_data1 = np.array(expr_data1)
    expr_data2 = np.array(expr_data2)

    # 计算adata1中每个细胞与adata2中所有细胞之间的欧几里得距离
    if distance_threshold is None:
        # print("Setting into Auto threshold mode...")
        s_time = time.time()
        meta_1 = adata1.obs
        meta_2 = adata2.obs
        transfer,t1,t2,cp_per, CT_num_table = get_transfer_matrix(i,j,expr_data1,expr_data2,meta_1,meta_2,method=method,emb=emb)

        f_time = time.time()
        elapsed_time = f_time - s_time
        elapsed_time = round(elapsed_time, 2)

        print(f"Select min num for sample {i} and sample {j}: ,{t1} , {t2} with { cp_per } % cell pair matched in {elapsed_time}s ...")

    else:
        if method == "euclidean":
            distance_matrix = pd.DataFrame(cdist(expr_data1, expr_data2, 'euclidean'))
            distance_matrix = distance_matrix / distance_matrix.max().max()

            transfer = distance_matrix.applymap(lambda x: 0 if x > distance_threshold else 1)
           
        elif method == "cosine":
            similarity_matrix = pd.DataFrame(cosine_similarity(expr_data1, expr_data2))
            similarity_matrix = 0.5 * (similarity_matrix + 1)  # 归一化到[0,1]

            transfer = similarity_matrix.applymap(lambda x: 0 if x < distance_threshold else 1)
    
    transfer.index = [f"{adata1.obs['orchestra'][0]}: " + i for i in adata1.obs.index]
    transfer.columns = [f"{adata2.obs['orchestra'][0]}: " + i for i in adata2.obs.index]

    save_index = transfer.apply(lambda row: list(row[row != 0].index), axis=1).to_frame(name='transfer')
   

    if transfer.mean().mean() <= 0 :
        cost = "None"
        matching = "None"
    else:
        # counts = adata2.obs['cell_type'].value_counts()
        # counts_in_order = counts.loc[transfer_1.columns.map(lambda x: x.split(": ")[1]) ]
        # duplicates_12 = {j: counts_in_order[j] for j in range(len(counts_in_order))}
        cost, B, matching = max_matching_from_transfer_matrix(transfer, None)
    
    return cost, matching, save_index, CT_num_table



def calculate_distance(i, j, dataset_1, dataset_2, method = "cosine",emb = "geneformer",distance_threshold = 0.2,read_data_dir = "processed_data"):
    print(f"Calculating distance for {dataset_1}:{i} and {dataset_2}:{j} with emb {emb} and method {method}...")
    # global mapping_dict
    adata1 = sc.read(f"{read_data_dir}/{dataset_1}/{i}.h5ad")
    # mapping_dict[f'mapping_{i}'].index = adata1.obs.index
    adata2 = sc.read(f"{read_data_dir}/{dataset_2}/{j}.h5ad")
    # mapping_dict[f'mapping_{j}'].index = adata2.obs.index
    # adata1 = adata[meta['orchestra'] == orchestra_cell['orchestra'][i]]
    # adata2 = adata[meta['orchestra'] == orchestra_cell['orchestra'][j]]
    if emb == "X_pca":
        adata1 = adata_pca(adata1)
        adata2 = adata_pca(adata2)
    if emb == "X_umap":
        adata1 = adata_umap(adata1)
        adata2 = adata_umap(adata2)
    
    if method == "euclidean" or method == "cosine":
        cost, matching , save_index, CT_num_table = cal_sc_distance(i,j,adata1, adata2,method=method,emb = emb,distance_threshold = distance_threshold)
    else:
        cost = None
        matching = None
        save_index = None
        print("method not found")

    index_1 = [f"{adata1.obs['orchestra'][0]}: " + i for i in adata1.obs.index]
    index_2 = [f"{adata2.obs['orchestra'][0]}: " + i for i in adata2.obs.index]
    # print("return")

    return (i, j, cost,matching,index_1,index_2, save_index, CT_num_table)

def get_similarity_matrix(i, j, method, emb, dataset = "Lung", save = True, read_data_dir = "processed_data"):
    print(f"Calculating similarity matrix for {i} and {j} with emb {emb} and method {method}...")

    make_dir(f"result/{dataset}/{method}_{emb}")
    make_dir(f"result/{dataset}/{method}_{emb}/similarity_matrix")

    adata1 = sc.read(f"{read_data_dir}/{dataset}/{i}.h5ad")
    adata2 = sc.read(f"{read_data_dir}/{dataset}/{j}.h5ad")
    
    if emb == "X_pca":
        adata1 = adata_pca(adata1)
        adata2 = adata_pca(adata2)
    if emb == "X_umap":
        adata1 = adata_umap(adata1)
        adata2 = adata_umap(adata2)
        
    if method == "euclidean": #距离
        expr_data1 = adata1.obsm[emb]
        expr_data2 = adata2.obsm[emb]
        # 确保数据是numpy数组
        expr_data1 = np.array(expr_data1)
        expr_data2 = np.array(expr_data2)

        similarity_matrix = pd.DataFrame(cdist(expr_data1, expr_data2, 'euclidean'))
        similarity_matrix = similarity_matrix / similarity_matrix.max().max()

        similarity_matrix.index = [f"{adata1.obs['orchestra'][0]}: " + i for i in adata1.obs.index]
        similarity_matrix.columns = [f"{adata2.obs['orchestra'][0]}: " + i for i in adata2.obs.index]
    elif method == "cosine": #相似性
        expr_data1 = adata1.obsm[emb]
        expr_data2 = adata2.obsm[emb]
        # 确保数据是numpy数组
        expr_data1 = np.array(expr_data1)
        expr_data2 = np.array(expr_data2)

        similarity_matrix = pd.DataFrame(cosine_similarity(expr_data1, expr_data2))
        similarity_matrix = 0.5 * (similarity_matrix + 1)  # 归一化到[0,1]

        similarity_matrix.index = [f"{adata1.obs['orchestra'][0]}: " + i for i in adata1.obs.index]
        similarity_matrix.columns = [f"{adata2.obs['orchestra'][0]}: " + i for i in adata2.obs.index]

    else:
        similarity_matrix = None
        print(f"method {method} not found")
    
    if save:
        similarity_matrix.to_csv(f"result/{dataset}/{method}_{emb}/similarity_matrix/{i}_{j}.csv")
    return (i, j, similarity_matrix)

