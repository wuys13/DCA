# nohup python -u data_pre.py --input_path "pbmc_4638k.h5ad" --dataset PBMC --cell_num 3000 --trim_flag > data_pre.log 2>&1 &
# nohup python -u data_pre.py --input_path "lung.h5ad" --dataset Lung --cell_num 5000 --trim_flag > data_pre.log 2>&1 &
# nohup python -u data_pre.py --input_path "blood.h5ad" --dataset Blood --cell_num 3000 --trim_flag > data_pre.log 2>&1 &
# nohup python -u data_pre.py --input_path "blood_all.h5ad" --dataset Blood_all --cell_num 3000 --trim_flag > data_pre.log 2>&1 &


# nohup python -u data_pre.py --input_path "raw_data/Lung/lung.h5ad" --dataset Lung --cell_num 3000 --control_num 3 --trim_flag > data_pre.log 2>&1 &
# nohup python -u data_pre.py --input_path "raw_data/COVID/covid.h5ad" --dataset COVID --cell_num 3000 --control_num 3 --trim_flag > data_pre1.log 2>&1 &
#  nohup python -u data_pre.py --input_path "raw_data/COVID/SLE.h5ad" --dataset SLE --cell_num 3000 --control_num 3 --trim_flag > data_pre5.log 2>&1 &
#  nohup python -u data_pre.py --input_path "raw_data/COVID/nasal.h5ad" --dataset Nasal --cell_num 3000 --control_num 3 --trim_flag > data_pre2.log 2>&1 &
#  nohup python -u data_pre.py --input_path "raw_data/COVID/trachea.h5ad" --dataset Trachea --cell_num 3000 --control_num 3 --trim_flag > data_pre3.log 2>&1 &
#  nohup python -u data_pre.py --input_path "raw_data/COVID/bronchus.h5ad" --dataset Bronchus --cell_num 3000 --control_num 3 --trim_flag > data_pre4.log 2>&1 &




# nohup python -u data_pre.py --input_path "pbmc_10k.h5ad" --dataset PBMC_10K  > data_pre.log 2>&1 &

import scanpy as sc
import pandas as pd
import argparse
from tqdm import tqdm
from utils_emb import get_emb_adata
import os

import warnings
warnings.filterwarnings("ignore")


def find_closest_normal_indices(disease_df, normal_df, n=3):
    # 计算当前疾病样本与所有正常样本的 cell_num 差的绝对值
    differences = normal_df['cell_num'].apply(lambda x: abs(x - disease_df['cell_num']))
    # 返回差值最小的 n 个正常样本的索引
    return differences.nsmallest(n).index.tolist()

parser = argparse.ArgumentParser(description='Process some thresholds.')
parser.add_argument('--input_path', type=str, default="pbmc_10k.h5ad")
parser.add_argument('--control_num', type=int, default=0) 

parser.add_argument('--dataset', type=str, default="PBMC_10K")
parser.add_argument('--cell_num', type=float, default=3000)
parser.add_argument('--cuda_num', type=str, default='cuda:7')
parser.add_argument('--trim_flag', action='store_true', default=False)

args = parser.parse_args()
if args.trim_flag:
    print(f"Args input_path: {args.input_path}; dataset: {args.dataset}; cell_num: {args.cell_num}")
else:
    print(f"Args input_path: {args.input_path}; dataset: {args.dataset}; trim_flag: {args.trim_flag}")

if not os.path.exists(f"processed_data/{args.dataset}"):
    os.makedirs(f"processed_data/{args.dataset}")
    print(f"Creating processed_data/{args.dataset}")
else:
    print(f"processed_data/{args.dataset} already exists")


print(f"\nLoading data from {args.input_path}...")
adata = sc.read(args.input_path)
meta = adata.obs
meta = meta[['cell_type','disease','donor_id','tissue']]
# meta = meta.loc[meta['tissue'] == args.dataset.lower()]
meta['orchestra'] = ""
for i in ['disease','donor_id']:
    # print(type(meta[i][0]))
    meta['orchestra'] = meta['orchestra'].astype(str) + meta[i].astype(str) + "__"
adata.obs = meta

orchestra_cell = meta.groupby('orchestra').agg(cell_num=('orchestra', 'count')).reset_index()
# orchestra_cell = orchestra_cell[(orchestra_cell['cell_num'] > 3000) & (orchestra_cell['cell_num'] < 5000)]
orchestra_cell.sort_values(by='cell_num',ascending=True,inplace=True)
orchestra_cell.index = range(len(orchestra_cell))

orchestra_cell['disease'] = orchestra_cell['orchestra'].apply(lambda x: x.split("__")[0])
orchestra_cell['donor_id'] = orchestra_cell['orchestra'].apply(lambda x: x.split("__")[1])
orchestra_cell.reset_index(drop=True, inplace=True)

print(orchestra_cell)
orchestra_cell.to_csv(f"processed_data/{args.dataset}/orchestra_cell.csv",index=False)

print("\nStarting comp_orchestra:")

normal_index = orchestra_cell.loc[orchestra_cell['disease'] == 'normal'].index
disease_index = orchestra_cell.loc[orchestra_cell['disease'] != 'normal'].index

normal_df = orchestra_cell.loc[orchestra_cell['disease'] == 'normal']
disease_df = orchestra_cell.loc[orchestra_cell['disease'] != 'normal']

if args.control_num == 0:
    comp_orchestra_1 = pd.DataFrame({
        'Sample1': disease_index,
        'Sample2': [list(normal_index)] * len(disease_index)
    })
    comp_orchestra_2 = pd.DataFrame({
        'Sample1': normal_index,
        'Sample2': [list(disease_index)] * len(normal_index)
    })
else:
    comp_orchestra_1 = pd.DataFrame({
        'Sample1': disease_index,
        'Sample2': disease_df.apply(find_closest_normal_indices, axis=1, args=(normal_df, args.control_num))
    })
    comp_orchestra_2 = pd.DataFrame({
        'Sample1': normal_index,
        'Sample2': [list(disease_index)] * len(normal_index)
    })


comp_orchestra_1['disease'] = orchestra_cell['disease'][disease_index].values
comp_orchestra_1['cell_num'] = (meta['disease'] != "normal").sum()
comp_orchestra_1['Sample_name'] = orchestra_cell['orchestra'][disease_index].values

comp_orchestra_2['disease'] = orchestra_cell['disease'][normal_index].values
comp_orchestra_2['cell_num'] = (meta['disease'] == "normal").sum()
comp_orchestra_2['Sample_name'] = orchestra_cell['orchestra'][normal_index].values
print(comp_orchestra_1,comp_orchestra_2)

comp_orchestra = pd.concat([comp_orchestra_1,comp_orchestra_2],axis=0)
comp_orchestra.reset_index(drop=True, inplace=True)

if args.trim_flag:
    comp_orchestra['cell_num'] = int(args.cell_num)


print(comp_orchestra)
comp_orchestra.to_csv(f"processed_data/{args.dataset}/comp_orchestra.csv",index=False)


# all_adata = sc.AnnData(X=np.empty((0,0)))
print("\nStarting split into orchestra...")
for i in tqdm(range(len(orchestra_cell))):
     select_orchestra = orchestra_cell.loc[i,'orchestra']
    #  print(f"Processing {i}: {select_orchestra}")
     select_adata = adata[adata.obs['orchestra'] == select_orchestra]

     select_adata = get_emb_adata(select_adata, device = args.cuda_num)
     select_adata = get_emb_adata(select_adata,modelname='geneformer',device = args.cuda_num)
     
     if args.trim_flag:
        print(f"Trimming to {args.cell_num} cells...")
        ratio = args.cell_num/select_adata.n_obs
        if ratio > 1:
            additional_cells_indices = select_adata.obs.sample(n=int(round((ratio-1)*len(select_adata.obs))), replace=True,random_state=42).index
            # print(additional_cells_indices)
            # 从select_adata中选择细胞
            additional_adata = select_adata[additional_cells_indices]
            index_counts = {}
            new_index = []
            for j in additional_adata.obs.index:
                if j not in index_counts:
                    index_counts[j] = 0
                else:
                    index_counts[j] += 1
                new_index.append(f"{j}_adding{index_counts[j]}" if index_counts[j] > 0 else f"{j}_adding")
            additional_adata.obs.index = new_index
            # print(additional_adata)
            # 合并select_adata和additional_adata
            select_adata = sc.concat([select_adata, additional_adata], join='inner')
        else:
            select_index = select_adata.obs.sample(n=int(round(args.cell_num)), replace=False,random_state=42).index
            select_adata = select_adata[select_index]     
     else:
        print(f"Remain all cells {i}: {select_adata.n_obs}")
     
     select_adata.obs.index = [ select_adata.obs.index[j] + "@@" + select_adata.obs['cell_type'][j] for j in range(len(select_adata.obs.index)) ]

     print(select_adata)
     select_adata.write(f"processed_data/{args.dataset}/{i}.h5ad")
     print(select_adata.n_obs)
     

orchestra_cell.to_csv(f"processed_data/{args.dataset}/orchestra_cell.csv",index=False)
print("Done!")
