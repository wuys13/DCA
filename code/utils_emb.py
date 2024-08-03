# h5ad -> dataset
from geneformer import TranscriptomeTokenizer
from geneformer.tokenizer import tokenize_cell
import numpy as np
import pandas as pd
import scanpy as sc
import warnings
warnings.filterwarnings("ignore")

import torch.nn as nn, torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
import torch
from transformers import  BertModel
from tqdm import tqdm



class MyTranscriptomeTokenizer(TranscriptomeTokenizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def tokenize_anndata(self, data):
        if self.custom_attr_name_dict is not None:
            file_cell_metadata = {
                attr_key: [] for attr_key in self.custom_attr_name_dict.keys()
            }

        # with lp.connect(str(loom_file_path)) as data:
            # define coordinates of detected protein-coding or miRNA genes and vector of their normalization factors
        coding_miRNA_loc = np.where(
            [self.genelist_dict.get(i, False) for i in data.var["ensembl_id"]]
        )[0]
        norm_factor_vector = np.array(
            [
                self.gene_median_dict[i]
                for i in data.var["ensembl_id"][coding_miRNA_loc]
            ]
        )
        coding_miRNA_ids = data.var["ensembl_id"][coding_miRNA_loc]
        coding_miRNA_tokens = np.array(
            [self.gene_token_dict[i] for i in coding_miRNA_ids]
        )

        # define coordinates of cells passing filters for inclusion (e.g. QC)
        try:
            data.obs["filter_pass"]
        except AttributeError:
            var_exists = False
        else:
            var_exists = True

        if var_exists is True:
            filter_pass_loc = np.where(
                [True if i == 1 else False for i in data.obs["filter_pass"]]
            )[0]
        elif var_exists is False:
            print(
                f"data has no column attribute 'filter_pass'; tokenizing all cells."
            )
            filter_pass_loc = np.array([i for i in range(data.shape[0])])

        # scan through .loom files and tokenize cells
        tokenized_cells = []

        # for (_ix, _selection, view) in data.scan(items=filter_pass_loc, axis=1):
        #     # select subview with protein-coding and miRNA genes
        #     subview = view.view[coding_miRNA_loc, :]
        subview = data[filter_pass_loc, coding_miRNA_loc]

        # normalize by total counts per cell and multiply by 10,000 to allocate bits to precision
        # and normalize by gene normalization factors
        subview_norm_array = (
            subview.X.toarray().T
            / subview.obs.n_counts.to_numpy()
            * 10_000
            / norm_factor_vector[:, None]
        )
        # tokenize subview gene vectors
        tokenized_cells += [
            tokenize_cell(subview_norm_array[:, i], coding_miRNA_tokens)
            for i in range(subview_norm_array.shape[1])
        ]

        # add custom attributes for subview to dict
        if self.custom_attr_name_dict is not None:
            for k in file_cell_metadata.keys():
                file_cell_metadata[k] += subview.obs[k].tolist()
        else:
            file_cell_metadata = None

        return tokenized_cells, file_cell_metadata



class Pooler(nn.Module):
    def __init__(self, config, pretrained_proj, proj_dim):
        super().__init__()
        self.proj = nn.Linear(config.hidden_size, proj_dim)
        self.proj.load_state_dict(torch.load(pretrained_proj))
        # self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        pooled_output = hidden_states[:, 0]
        pooled_output = F.normalize(self.proj(pooled_output), dim=-1)
        return pooled_output


def cell_encode(cell_input_ids,model):
    cell = model(cell_input_ids.to(model.device)).last_hidden_state
    if model.pooler is not None:
        cell = F.normalize(model.pooler.proj(cell[:, 0, :]), dim=-1)
    else:
        cell = F.normalize(cell[:, 0, :], dim=-1)
    return cell


def cell_encode_pooling(cell_input_ids,model):
    cell = model(cell_input_ids.to(model.device)).last_hidden_state
    cell = cell.mean(dim=1)
    # cell = F.normalize(model.pooler.proj(cell[:, 0]), dim=-1)
    return cell



def get_emb_adata(adata,dataset_name = "lung",modelname = 'langcell',device = 'cuda:7'):
    data = adata
    
    data.obs['n_counts'] = data.X.sum(axis=1)
    data.obs['filter_pass'] = True

    if 'ensembl_id' not in data.var.keys():
        print("read from cellxgene format")
        gene = pd.read_csv('/mnt/users/wuyushuai/diff_cell/gene.csv',index_col=0)
        data.var = gene
        data.var['ensembl_id'] = data.var['feature_id']

    tk = MyTranscriptomeTokenizer(
        dict([(k, k) for k in data.obs.keys()]), nproc=4
        )
    tokenized_cells, cell_metadata = tk.tokenize_anndata(data)
    dataset = tk.create_dataset(
        tokenized_cells, cell_metadata)
    print("Dataset created.")
    # return dataset
    if modelname == 'geneformer':
        model = BertModel.from_pretrained( '/mnt/users/zhaosy/home/geneformer/ckpt/geneformer_small')
        model.pooler = None
    else:
        model = BertModel.from_pretrained( '/mnt/users/zhaosy/home/geneformer/ckpt/blcp_base/cell_bert')
        model.pooler = Pooler(model.config, pretrained_proj='/mnt/users/zhaosy/home/geneformer/ckpt/blcp_base/cell_proj.bin', proj_dim=256)

    model = model.to(device)
    # dataset = dataset.shuffle(seed=42)

    # %%
    cell_embs = []
    cell_embs_pooling = []
    model.eval()

    with torch.no_grad():
        for i, d in tqdm(enumerate(dataset)):
            if modelname != 'geneformer':
                cell_input = torch.tensor(([25426] + d['input_ids'])[:2048]).unsqueeze(0)
            else:
                cell_input = torch.tensor(d['input_ids']).unsqueeze(0)
            cellemb = cell_encode(cell_input,model)
            cellemb_pooling = cell_encode_pooling(cell_input,model)
            cell_embs.append(cellemb.cpu().tolist())
            cell_embs_pooling.append(cellemb_pooling.cpu().tolist())

    print("Embedding finished.")

    # %%
    cell_embs_np = np.array(cell_embs)[:, 0, :]
    cell_embs_pooling_np = np.array(cell_embs_pooling)[:, 0, :]

    data.obsm[f'emb_{modelname}'] = cell_embs_np
    data.obsm[f'emb_pooling_{modelname}'] = cell_embs_pooling_np
    # data.write(f"{dataset_name}.h5ad")
    print("Embedding saved.")
    return data

