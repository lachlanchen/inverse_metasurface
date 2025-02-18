#!/usr/bin/env python3
"""
This script processes multiple merged CSV files (from S4 runs) and converts them into a
formatted dataset. For each merged CSV (from the folder "merged_csvs") matching a given prefix,
the code groups rows by shape (each shape should have 11 rows corresponding to 11 crystallization
values), sorts each group by the crystallization fraction "c", extracts the reflectance columns
("R@..."), and saves the resulting 11×100 spectrum matrix (as a JSON‐encoded string) along with the
"vertices_str" (unchanged) into a new CSV file in the folder "formatted_csvs".

Then, in training mode, it loads all these formatted CSV files (each row now represents one shape)
via a custom dataset (FormattedSpectraDataset) that parses the JSON and computes the ground‐truth shape
(from "vertices_str") as before. The two–stage training pipeline (Stage A: shape→spec and Stage B: spec→shape)
remains unchanged.

Usage:
  -- To preprocess:
     python train_with_formatted_data.py --prefix myrun_seed12345_g40 --preprocess --batch_num 10000 --max_num 80000
  -- To train:
     python train_with_formatted_data.py --prefix myrun_seed12345_g40 --batch_num 10000 --max_num 80000
"""

import os, re, glob, argparse, json
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime
import random

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

###############################################################################
# PREPROCESSING FUNCTIONS: Format merged CSVs into a smaller CSV per batch
###############################################################################
def parse_filename_get_nQnS(csv_path):
    """Given a filename like 'myrun_seed12345_g40_nQ1_nS100000_b0.35_r0.30.csv',
    return (prefix, nQ, nS) e.g. ('myrun_seed12345_g40', 1, 100000)"""
    base = os.path.basename(csv_path)
    pattern = re.compile(r'^(.*?)_nQ(\d+)_nS(\d+).*\.csv$')
    m = pattern.match(base)
    if not m:
        return None, None, None
    return m.group(1), int(m.group(2)), int(m.group(3))

def format_merged_csv(merged_csv_path, out_csv_path):
    """Reads a merged CSV, groups rows by (prefix, nQ, nS, shape_idx), and if a group
    has exactly 11 rows, sorts it by 'c', extracts the reflectance columns (those starting with "R@"),
    and saves a single row with:
         uid, prefix, nQ, nS, shape_idx, spec, vertices_str
    where 'spec' is a JSON string encoding the 11×100 spectrum matrix.
    """
    try:
        df = pd.read_csv(merged_csv_path, engine='python', on_bad_lines='skip')
    except Exception as e:
        print(f"[ERROR] Reading {merged_csv_path} failed: {e}")
        return
    # Group by keys
    group_keys = ['prefix', 'nQ', 'nS', 'shape_idx']
    groups = df.groupby(group_keys, sort=False)
    rows = []
    for name, grp in groups:
        if len(grp) != 11:
            continue
        grp_sorted = grp.sort_values(by='c')
        # Find reflectance columns (assume columns starting with "R@")
        r_cols = [col for col in grp_sorted.columns if col.startswith("R@")]
        if len(r_cols) == 0:
            continue
        spec_matrix = grp_sorted[r_cols].values.astype(float)  # shape (11, ?)
        spec_json = json.dumps(spec_matrix.tolist())
        # Use vertices_str from first row
        vertices = str(grp_sorted.iloc[0].get("vertices_str", "")).strip()
        uid = f"{name[0]}_{name[1]}_{name[2]}_{name[3]}"
        rows.append({
            "uid": uid,
            "prefix": name[0],
            "nQ": name[1],
            "nS": name[2],
            "shape_idx": name[3],
            "spec": spec_json,
            "vertices_str": vertices
        })
    if not rows:
        print(f"[WARN] No valid groups found in {merged_csv_path}")
        return
    df_out = pd.DataFrame(rows)
    df_out.to_csv(out_csv_path, index=False)
    print(f"[FORMAT] Wrote formatted file => {out_csv_path} with {df_out.shape[0]} samples.")

def format_all_merged_csvs(prefix, merged_folder="merged_csvs", formatted_folder="formatted_csvs", batch_num=10000):
    if not os.path.exists(formatted_folder):
        os.makedirs(formatted_folder, exist_ok=True)
    pattern = os.path.join(merged_folder, f"{prefix}_first*.csv")
    files = glob.glob(pattern)
    formatted_files = []
    for f in files:
        base = os.path.basename(f)
        outname = f"formatted_{base}"
        outpath = os.path.join(formatted_folder, outname)
        if os.path.exists(outpath):
            print(f"[FORMAT] {outpath} exists, skipping.")
            formatted_files.append(outpath)
        else:
            format_merged_csv(f, outpath)
            if os.path.exists(outpath):
                formatted_files.append(outpath)
    return formatted_files

###############################################################################
# NEW DATASET: Loads formatted CSV files (one row per shape)
###############################################################################
class FormattedSpectraDataset(Dataset):
    def __init__(self, csv_files):
        dfs = []
        for f in csv_files:
            try:
                df = pd.read_csv(f, engine='python')
                dfs.append(df)
            except Exception as e:
                print(f"[ERROR] Reading {f} failed: {e}")
        if not dfs:
            raise ValueError("No formatted CSV files could be loaded.")
        self.df = pd.concat(dfs, ignore_index=True)
        self.df["spec_parsed"] = self.df["spec"].apply(lambda s: np.array(json.loads(s), dtype=np.float32))
        self.df["shape_uid"] = self.df["uid"]
        self.data_list = []
        for _, row in self.df.iterrows():
            spec = row["spec_parsed"]  # (11,100)
            v_str = str(row.get("vertices_str", "")).strip()
            if not v_str:
                continue
            raw_pairs = v_str.split(";")
            all_xy = []
            for pair in raw_pairs:
                pair = pair.strip()
                if pair:
                    parts = pair.split(",")
                    if len(parts)==2:
                        try:
                            x_val = float(parts[0])
                            y_val = float(parts[1])
                            all_xy.append([x_val, y_val])
                        except:
                            continue
            all_xy = np.array(all_xy, dtype=np.float32)
            if len(all_xy)==0:
                continue
            shifted = all_xy - 0.5
            q1 = []
            for (xx,yy) in shifted:
                if xx > 0 and yy > 0:
                    q1.append([xx,yy])
            q1 = np.array(q1, dtype=np.float32)
            n_q1 = len(q1)
            if n_q1 < 1 or n_q1 > 4:
                continue
            shape_4x3 = np.zeros((4,3), dtype=np.float32)
            for i in range(n_q1):
                shape_4x3[i,0] = 1.0
                shape_4x3[i,1] = q1[i,0]
                shape_4x3[i,2] = q1[i,1]
            self.data_list.append({
                "uid": row["uid"],
                "spec": spec,      # shape (11,100)
                "shape": shape_4x3  # shape (4,3)
            })
        self.data_len = len(self.data_list)
        if self.data_len == 0:
            raise ValueError("No valid samples in formatted dataset.")
    def __len__(self):
        return self.data_len
    def __getitem__(self, idx):
        item = self.data_list[idx]
        return (item["spec"], item["shape"], item["uid"])

###############################################################################
# MODEL DEFINITIONS (same as before)
###############################################################################
class ShapeToSpectraModel(nn.Module):
    def __init__(self, d_in=3, d_model=256, nhead=4, num_layers=4):
        super().__init__()
        self.input_proj = nn.Linear(d_in, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model*4,
            dropout=0.1, activation='relu',
            batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model*4),
            nn.ReLU(),
            nn.Linear(d_model*4, d_model*4),
            nn.ReLU(),
            nn.Linear(d_model*4, d_model*2),
            nn.ReLU(),
            nn.Linear(d_model*2, 11*100)
        )
    def forward(self, shape_4x3):
        bsz = shape_4x3.size(0)
        presence = shape_4x3[:,:,0]
        key_padding_mask = (presence < 0.5)
        x_proj = self.input_proj(shape_4x3)
        x_enc = self.encoder(x_proj, src_key_padding_mask=key_padding_mask)
        pres_sum = presence.sum(dim=1, keepdim=True) + 1e-8
        x_enc_w = x_enc * presence.unsqueeze(-1)
        shape_emb = x_enc_w.sum(dim=1) / pres_sum
        out_flat = self.mlp(shape_emb)
        out_2d = out_flat.view(bsz, 11, 100)
        return out_2d

class Spectra2ShapeVarLen(nn.Module):
    def __init__(self, d_in=100, d_model=256, nhead=4, num_layers=4):
        super().__init__()
        self.row_preproc = nn.Sequential(
            nn.Linear(d_in, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.ReLU()
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model*4,
            dropout=0.1, activation='relu',
            batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 12)  # 4*3 = 12 (presence, x, y for each of 4 points)
        )
    def forward(self, spec_11x100):
        bsz = spec_11x100.size(0)
        x_r = spec_11x100.view(-1, spec_11x100.size(2))
        x_pre = self.row_preproc(x_r)
        x_pre = x_pre.view(bsz, -1, x_pre.size(-1))
        x_enc = self.encoder(x_pre)
        x_agg = x_enc.mean(dim=1)
        out_12 = self.mlp(x_agg)
        out_4x3 = out_12.view(bsz, 4, 3)
        # Chain the predicted presence values
        presence_logits = out_4x3[:,:,0]
        presence_list = []
        for i in range(4):
            if i==0:
                presence_list.append(torch.ones(bsz, device=out_4x3.device, dtype=torch.float32))
            else:
                prob_i = torch.sigmoid(presence_logits[:,i]).clamp(1e-6, 1-1e-6)
                prob_chain = prob_i * presence_list[i-1]
                ste_i = (prob_chain > 0.5).float() + prob_chain - prob_chain.detach()
                presence_list.append(ste_i)
        presence_stack = torch.stack(presence_list, dim=1)
        xy_bounded = torch.sigmoid(out_4x3[:,:,1:])
        xy_final = xy_bounded * presence_stack.unsqueeze(-1)
        final_shape = torch.cat([presence_stack.unsqueeze(-1), xy_final], dim=-1)
        return final_shape

class Spec2ShapeFrozen(nn.Module):
    def __init__(self, spec2shape_net, shape2spec_frozen):
        super().__init__()
        self.spec2shape = spec2shape_net
        self.shape2spec_frozen = shape2spec_frozen
        for p in self.shape2spec_frozen.parameters():
            p.requires_grad = False
    def forward(self, spec_input):
        shape_pred = self.spec2shape(spec_input)
        with torch.no_grad():
            spec_chain = self.shape2spec_frozen(shape_pred)
        return shape_pred, spec_chain

###############################################################################
# TRAINING FUNCTIONS (Stage A and Stage B) using the new dataset
###############################################################################
def train_stageA_shape2spec(dataset, out_dir, num_epochs=500, batch_size=1024,
                            lr=1e-4, weight_decay=1e-5, split_ratio=0.8, grad_clip=1.0):
    ds_full = dataset
    ds_len = len(ds_full)
    trn_len = int(ds_len * split_ratio)
    val_len = ds_len - trn_len
    ds_train, ds_val = random_split(ds_full, [trn_len, val_len])
    print(f"[DATA: Stage A] total={ds_len}, train={trn_len}, val={val_len}")

    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(ds_val, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[Stage A] device:", device)

    model = ShapeToSpectraModel(d_in=3, d_model=256, nhead=4, num_layers=4).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)
    crit_mse = nn.MSELoss()

    train_losses, val_losses = [], []
    for epoch in range(num_epochs):
        model.train()
        run_loss = 0.0
        for (spec, shape, uid) in train_loader:
            shape_t = shape.to(device)
            spec_gt = spec.to(device)
            spec_pd = model(shape_t)
            loss = crit_mse(spec_pd, spec_gt)
            optimizer.zero_grad()
            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            run_loss += loss.item()
        avg_train = run_loss / len(train_loader)
        train_losses.append(avg_train)

        model.eval()
        val_sum, val_count = 0.0, 0
        with torch.no_grad():
            for (spec, shape, uid) in val_loader:
                bsz = shape.size(0)
                s_t = shape.to(device)
                s_gt = spec.to(device)
                s_pd = model(s_t)
                v = crit_mse(s_pd, s_gt) * bsz
                val_sum += v.item()
                val_count += bsz
        avg_val = val_sum / val_count
        val_losses.append(avg_val)
        scheduler.step(avg_val)
        if (epoch+1)%50==0 or epoch==0 or epoch==(num_epochs-1):
            print(f"[Stage A] Epoch[{epoch+1}/{num_epochs}] => trainLoss={avg_train:.4f}, valLoss={avg_val:.4f}")

    np.savetxt(os.path.join(out_dir, "train_losses_stageA.csv"), np.array(train_losses), delimiter=",")
    np.savetxt(os.path.join(out_dir, "val_losses_stageA.csv"), np.array(val_losses), delimiter=",")
    plt.figure()
    plt.plot(train_losses, label='train')
    plt.plot(val_losses, label='val')
    plt.xlabel("Epoch")
    plt.ylabel("MSE (shape->spec)")
    plt.legend()
    plt.savefig(os.path.join(out_dir, "training_curve_stageA.png"))
    plt.close()
    shape2spec_path = os.path.join(out_dir, "shape2spec_stageA.pt")
    torch.save(model.state_dict(), shape2spec_path)
    print("[Stage A] model saved =>", shape2spec_path)
    return shape2spec_path, ds_val, model

def train_stageB_spec2shape_frozen(dataset, out_dir, shape2spec_ckpt, num_epochs=500, batch_size=1024,
                                  lr=1e-4, weight_decay=1e-5, split_ratio=0.8, grad_clip=1.0):
    ds_full = dataset
    ds_len = len(ds_full)
    trn_len = int(ds_len * split_ratio)
    val_len = ds_len - trn_len
    ds_train, ds_val = random_split(ds_full, [trn_len, val_len])
    print(f"[DATA: Stage B] total={ds_len}, train={trn_len}, val={val_len}")

    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(ds_val, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # load frozen shape2spec
    shape2spec_frozen = ShapeToSpectraModel(d_in=3, d_model=256, nhead=4, num_layers=4)
    shape2spec_frozen.load_state_dict(torch.load(shape2spec_ckpt))
    shape2spec_frozen.to(device)
    for p in shape2spec_frozen.parameters():
        p.requires_grad = False

    spec2shape_net = Spectra2ShapeVarLen(d_in=100, d_model=256, nhead=4, num_layers=4).to(device)
    pipeline = Spec2ShapeFrozen(spec2shape_net, shape2spec_frozen).to(device)

    optimizer = torch.optim.AdamW(spec2shape_net.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)
    crit_mse = nn.MSELoss()

    train_losses, val_losses = [], []
    for epoch in range(num_epochs):
        pipeline.train()
        run_loss = 0.0
        for (spec, shape, uid) in train_loader:
            spec_t = spec.to(device)
            shape_t = shape.to(device)
            shape_pd, spec_pd = pipeline(spec_t)
            loss = crit_mse(shape_pd, shape_t) + crit_mse(spec_pd, spec_t)
            optimizer.zero_grad()
            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(spec2shape_net.parameters(), grad_clip)
            optimizer.step()
            run_loss += loss.item()
        avg_train = run_loss / len(train_loader)
        train_losses.append(avg_train)
        pipeline.eval()
        val_sum, val_count = 0.0, 0
        with torch.no_grad():
            for (spec, shape, uid) in val_loader:
                bsz = spec.size(0)
                s_t = spec.to(device)
                sh_t = shape.to(device)
                sp, sc = pipeline(s_t)
                v = (crit_mse(sp, sh_t) + crit_mse(sc, s_t)) * bsz
                val_sum += v.item()
                val_count += bsz
        avg_val = val_sum / val_count
        val_losses.append(avg_val)
        scheduler.step(avg_val)
        if (epoch+1)%50==0 or epoch==0 or epoch==(num_epochs-1):
            print(f"[Stage B] Epoch[{epoch+1}/{num_epochs}] => trainLoss={avg_train:.4f}, valLoss={avg_val:.4f}")

    np.savetxt(os.path.join(out_dir, "train_losses_stageB.csv"), np.array(train_losses), delimiter=",")
    np.savetxt(os.path.join(out_dir, "val_losses_stageB.csv"), np.array(val_losses), delimiter=",")
    plt.figure()
    plt.plot(train_losses, label='train')
    plt.plot(val_losses, label='val')
    plt.xlabel("Epoch")
    plt.ylabel("Loss (spec->shape + spec->shape2spec)")
    plt.legend()
    plt.savefig(os.path.join(out_dir, "training_curve_stageB.png"))
    plt.close()
    spec2shape_path = os.path.join(out_dir, "spec2shape_stageB.pt")
    torch.save(spec2shape_net.state_dict(), spec2shape_path)
    print("[Stage B] spec2shape saved =>", spec2shape_path)
    return spec2shape_path, ds_val, shape2spec_frozen, spec2shape_net

###############################################################################
# MAIN: Parse arguments, optionally preprocess, then train using formatted dataset
###############################################################################
def parse_args():
    parser = argparse.ArgumentParser(description="Train using formatted dataset from multiple merged CSVs.")
    parser.add_argument("--prefix", required=True, help="Prefix to filter merged CSV files (e.g., 'myrun_seed12345_g40').")
    parser.add_argument("--batch_num", type=int, default=10000, help="Batch size for splitting (default 10000).")
    parser.add_argument("--max_num", type=int, default=80000, help="Maximum shape index to process (e.g., 80000 gives 8 batches per file).")
    parser.add_argument("--preprocess", action="store_true", help="If set, run preprocessing and exit.")
    parser.add_argument("--merged_folder", type=str, default="merged_csvs", help="Folder with merged CSV files.")
    parser.add_argument("--formatted_folder", type=str, default="formatted_csvs", help="Folder to store formatted CSV files.")
    parser.add_argument("--out_dir", type=str, default=None, help="Output folder base for training outputs.")
    parser.add_argument("--stageA_epochs", type=int, default=500, help="Epochs for Stage A training.")
    parser.add_argument("--stageB_epochs", type=int, default=500, help="Epochs for Stage B training.")
    return parser.parse_args()

def main():
    args = parse_args()
    if args.preprocess:
        print("[PREPROCESS] Formatting merged CSV files...")
        formatted_files = format_all_merged_csvs(args.prefix, merged_folder=args.merged_folder,
                                                  formatted_folder=args.formatted_folder,
                                                  batch_num=args.batch_num)
        print("[PREPROCESS] Formatted files:")
        for f in formatted_files:
            print("  ", f)
        return

    # In training mode, load all formatted CSV files.
    pattern = os.path.join(args.formatted_folder, f"formatted_{args.prefix}*_sub_first*.csv")
    formatted_files = glob.glob(pattern)
    if not formatted_files:
        print("[ERROR] No formatted CSV files found. Run with --preprocess first.")
        return
    print(f"[INFO] Found {len(formatted_files)} formatted CSV files.")

    dataset = FormattedSpectraDataset(formatted_files)
    print(f"[DATASET] Loaded formatted dataset with {len(dataset)} samples.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.out_dir is None:
        base_out = f"outputs_two_stage_{timestamp}"
    else:
        base_out = args.out_dir
    os.makedirs(base_out, exist_ok=True)

    # Stage A training: shape->spec
    stageA_dir = os.path.join(base_out, "stageA")
    os.makedirs(stageA_dir, exist_ok=True)
    shape2spec_ckpt, ds_valA, modelA = train_stageA_shape2spec(dataset, out_dir=stageA_dir,
                                                               num_epochs=args.stageA_epochs,
                                                               batch_size=1024,
                                                               lr=1e-4,
                                                               weight_decay=1e-5,
                                                               split_ratio=0.8,
                                                               grad_clip=1.0)
    # Stage B training: spec->shape (frozen shape2spec)
    stageB_dir = os.path.join(base_out, "stageB")
    os.makedirs(stageB_dir, exist_ok=True)
    spec2shape_ckpt, ds_valB, shape2spec_froz, spec2shape_net = train_stageB_spec2shape_frozen(
        dataset, out_dir=stageB_dir,
        shape2spec_ckpt=shape2spec_ckpt,
        num_epochs=args.stageB_epochs,
        batch_size=1024,
        lr=1e-4,
        weight_decay=1e-5,
        split_ratio=0.8,
        grad_clip=1.0
    )
    print("[DONE] Training complete.")

if __name__ == "__main__":
    main()

