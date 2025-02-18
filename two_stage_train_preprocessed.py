#!/usr/bin/env python3
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

###############################################################################
# PREPROCESSING FUNCTIONS
###############################################################################
def process_csv_file(csv_path, max_points=4):
    """
    Read one CSV file, group rows by unique shape (using 'shape_uid'), and
    for each group (which must have exactly 11 rows) process the reflectance
    spectrum (11×100) and parse the polygon (vertices_str). Only polygons whose
    shifted (minus 0.5) Q1-points count is between 1 and max_points are kept.
    Returns a list of dictionaries with keys: 'uid', 'spectra', and 'shape'.
    """
    df = pd.read_csv(csv_path)
    # Find all reflectance columns (they start with "R@")
    r_cols = [c for c in df.columns if c.startswith("R@")]
    if len(r_cols) == 0:
        raise ValueError("No reflectance columns found in " + csv_path)
    # Build a unique ID per shape
    df["shape_uid"] = (df["prefix"].astype(str) + "_" +
                       df["nQ"].astype(str) + "_" +
                       df["nS"].astype(str) + "_" +
                       df["shape_idx"].astype(str))
    records = []
    grouped = df.groupby("shape_uid", sort=False)
    for uid, grp in grouped:
        if len(grp) != 11:
            continue
        grp_sorted = grp.sort_values(by="c")
        spec_11x100 = grp_sorted[r_cols].values.astype(np.float32)
        first_row = grp_sorted.iloc[0]
        v_str = str(first_row.get("vertices_str", "")).strip()
        if not v_str:
            continue
        raw_pairs = v_str.split(";")
        all_xy = []
        for pair in raw_pairs:
            pair = pair.strip()
            if pair:
                xy = pair.split(",")
                if len(xy) == 2:
                    try:
                        x_val = float(xy[0])
                        y_val = float(xy[1])
                    except Exception:
                        continue
                    all_xy.append([x_val, y_val])
        all_xy = np.array(all_xy, dtype=np.float32)
        if len(all_xy) == 0:
            continue
        # SHIFT: subtract (0.5, 0.5)
        shifted = all_xy - 0.5
        q1 = []
        for (xx, yy) in shifted:
            if xx > 0 and yy > 0:
                q1.append([xx, yy])
        q1 = np.array(q1, dtype=np.float32)
        n_q1 = len(q1)
        if n_q1 < 1 or n_q1 > max_points:
            continue
        # Build a fixed-size (max_points×3) shape array: first column is presence,
        # then x and y coordinates.
        shape_4x3 = np.zeros((max_points, 3), dtype=np.float32)
        for i in range(n_q1):
            shape_4x3[i, 0] = 1.0
            shape_4x3[i, 1] = q1[i, 0]
            shape_4x3[i, 2] = q1[i, 1]
        records.append({"uid": uid, "spectra": spec_11x100, "shape": shape_4x3})
    return records

def preprocess_csv_folder(input_folder, output_npz, max_points=4):
    """
    Iterate over all CSV files (one by one) in the input_folder,
    process each file, and accumulate all valid records. Then save the final
    arrays (uids, spectra, shapes) into a compressed NPZ file.
    """
    all_records = []
    csv_files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith('.csv')]
    csv_files.sort()
    for csv_file in csv_files:
        print("Processing file:", csv_file)
        records = process_csv_file(csv_file, max_points=max_points)
        print("Found", len(records), "records in", csv_file)
        all_records.extend(records)
    uids = [rec["uid"] for rec in all_records]
    spectra = np.array([rec["spectra"] for rec in all_records])  # shape: (N, 11, 100)
    shapes = np.array([rec["shape"] for rec in all_records])      # shape: (N, 4, 3)
    print("Total records processed:", len(uids))
    np.savez_compressed(output_npz, uids=uids, spectra=spectra, shapes=shapes)
    print("Preprocessed data saved to", output_npz)

###############################################################################
# DATASET FOR PREPROCESSED DATA
###############################################################################
class PreprocessedSpectraDataset(Dataset):
    def __init__(self, npz_file):
        data = np.load(npz_file, allow_pickle=True)
        self.uids = data["uids"]
        self.spectra = data["spectra"]
        self.shapes = data["shapes"]
    def __len__(self):
        return len(self.uids)
    def __getitem__(self, idx):
        spec = self.spectra[idx]
        shape = self.shapes[idx]
        uid = self.uids[idx]
        spec_tensor = torch.tensor(spec, dtype=torch.float32)
        shape_tensor = torch.tensor(shape, dtype=torch.float32)
        return spec_tensor, shape_tensor, uid

###############################################################################
# UTILS: replicate_c4, sort_points_by_angle, plot_polygon
###############################################################################
def replicate_c4(points):
    c4 = []
    for (x, y) in points:
        c4.append([ x,  y])
        c4.append([-y,  x])
        c4.append([-x, -y])
        c4.append([ y, -x])
    return np.array(c4, dtype=np.float32)

def sort_points_by_angle(points):
    if len(points) < 3:
        return points
    cx, cy = points.mean(axis=0)
    angles = np.arctan2(points[:, 1] - cy, points[:, 0] - cx)
    idx = np.argsort(angles)
    return points[idx]

def plot_polygon(ax, points, color='green', alpha=0.4, fill=True):
    import matplotlib.patches as patches
    from matplotlib.path import Path
    if len(points) < 3:
        ax.scatter(points[:, 0], points[:, 1], c=color)
        return
    closed = np.concatenate([points, points[0:1]], axis=0)
    codes = [Path.MOVETO] + [Path.LINETO]*(len(points)-1) + [Path.CLOSEPOLY]
    path = Path(closed, codes)
    patch = patches.PathPatch(path, facecolor=color if fill else 'none',
                              alpha=alpha, edgecolor=color)
    ax.add_patch(patch)
    ax.autoscale_view()

###############################################################################
# MODEL DEFINITION: Stage A – shape->spec
###############################################################################
class ShapeToSpectraModel(nn.Module):
    def __init__(self, d_in=3, d_model=256, nhead=4, num_layers=4):
        super().__init__()
        self.input_proj = nn.Linear(d_in, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            activation='relu',
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, 11 * 100)
        )
    def forward(self, shape_4x3):
        bsz = shape_4x3.size(0)
        presence = shape_4x3[:, :, 0]
        key_padding_mask = (presence < 0.5)
        x_proj = self.input_proj(shape_4x3)
        x_enc = self.encoder(x_proj, src_key_padding_mask=key_padding_mask)
        pres_sum = presence.sum(dim=1, keepdim=True) + 1e-8
        x_enc_w = x_enc * presence.unsqueeze(-1)
        shape_emb = x_enc_w.sum(dim=1) / pres_sum
        out_flat = self.mlp(shape_emb)
        out_2d = out_flat.view(bsz, 11, 100)
        return out_2d

###############################################################################
# DATASET CLASS (for original CSV file) – unchanged for fallback use
###############################################################################
class Q1ShiftedSpectraDataset(Dataset):
    def __init__(self, csv_file, max_points=4):
        super().__init__()
        self.df = pd.read_csv(csv_file)
        self.r_cols = [c for c in self.df.columns if c.startswith("R@")]
        if len(self.r_cols) == 0:
            raise ValueError("No reflectance columns found (R@...).")
        self.df["shape_uid"] = (self.df["prefix"].astype(str) + "_" +
                                self.df["nQ"].astype(str) + "_" +
                                self.df["nS"].astype(str) + "_" +
                                self.df["shape_idx"].astype(str))
        self.data_list = []
        grouped = self.df.groupby("shape_uid", sort=False)
        for uid, grp in grouped:
            if len(grp) != 11:
                continue
            grp_sorted = grp.sort_values(by="c")
            spec_11x100 = grp_sorted[self.r_cols].values.astype(np.float32)
            first_row = grp_sorted.iloc[0]
            v_str = str(first_row.get("vertices_str", "")).strip()
            if not v_str:
                continue
            raw_pairs = v_str.split(";")
            all_xy = []
            for pair in raw_pairs:
                pair = pair.strip()
                if pair:
                    xy = pair.split(",")
                    if len(xy) == 2:
                        try:
                            x_val = float(xy[0])
                            y_val = float(xy[1])
                        except Exception:
                            continue
                        all_xy.append([x_val, y_val])
            all_xy = np.array(all_xy, dtype=np.float32)
            if len(all_xy) == 0:
                continue
            shifted = all_xy - 0.5
            q1 = []
            for (xx, yy) in shifted:
                if xx > 0 and yy > 0:
                    q1.append([xx, yy])
            q1 = np.array(q1, dtype=np.float32)
            n_q1 = len(q1)
            if n_q1 < 1 or n_q1 > max_points:
                continue
            shape_4x3 = np.zeros((max_points, 3), dtype=np.float32)
            for i in range(n_q1):
                shape_4x3[i, 0] = 1.0
                shape_4x3[i, 1] = q1[i, 0]
                shape_4x3[i, 2] = q1[i, 1]
            self.data_list.append({
                "uid": uid,
                "spectra": spec_11x100,
                "shape": shape_4x3
            })
        self.data_len = len(self.data_list)
        if self.data_len == 0:
            raise ValueError("No valid shapes => SHIFT->Q1->UpTo4")
    def __len__(self):
        return self.data_len
    def __getitem__(self, idx):
        it = self.data_list[idx]
        spec = it["spectra"]
        shape = it["shape"]
        uid = it["uid"]
        spec_tensor = torch.tensor(spec, dtype=torch.float32)
        shape_tensor = torch.tensor(shape, dtype=torch.float32)
        return spec_tensor, shape_tensor, uid

###############################################################################
# MODEL DEFINITION: Stage B – spec->shape
###############################################################################
class Spectra2ShapeVarLen(nn.Module):
    def __init__(self, d_in=100, d_model=256, nhead=4, num_layers=4):
        super().__init__()
        self.row_preproc = nn.Sequential(
            nn.Linear(d_in, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.ReLU()
        )
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            activation='relu',
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 12)  # outputs: presence (4) and (x,y) for 4 points
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
        presence_logits = out_4x3[:, :, 0]
        xy_raw = out_4x3[:, :, 1:]
        presence_list = []
        for i in range(4):
            if i == 0:
                presence_list.append(torch.ones(bsz, device=out_4x3.device, dtype=torch.float32))
            else:
                prob_i = torch.sigmoid(presence_logits[:, i]).clamp(1e-6, 1 - 1e-6)
                prob_chain = prob_i * presence_list[i - 1]
                ste_i = (prob_chain > 0.5).float() + prob_chain - prob_chain.detach()
                presence_list.append(ste_i)
        presence_stack = torch.stack(presence_list, dim=1)
        xy_bounded = torch.sigmoid(xy_raw)
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
# TRAINING FUNCTIONS
###############################################################################
def train_stageA_shape2spec(data_source, out_dir, num_epochs=500, batch_size=1024*32,
                            lr=1e-4, weight_decay=1e-5, split_ratio=0.8,
                            grad_clip=1.0, use_preprocessed=False):
    os.makedirs(out_dir, exist_ok=True)
    print("[Stage A] =>", out_dir)
    if use_preprocessed:
        ds_full = PreprocessedSpectraDataset(data_source)
    else:
        ds_full = Q1ShiftedSpectraDataset(data_source)
    ds_len = len(ds_full)
    trn_len = int(ds_len * split_ratio)
    val_len = ds_len - trn_len
    ds_train, ds_val = random_split(ds_full, [trn_len, val_len])
    print("[DATA: Stage A] total={}, train={}, val={}".format(ds_len, trn_len, val_len))
    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(ds_val, batch_size=batch_size, shuffle=False, drop_last=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[Stage A] device:", device)
    model = ShapeToSpectraModel(d_in=3, d_model=256, nhead=4, num_layers=4).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                           factor=0.5, patience=10, verbose=True)
    crit_mse = nn.MSELoss()
    train_losses, val_losses = [], []
    for epoch in range(num_epochs):
        model.train()
        run_loss = 0.0
        for (spec_np, shape_np, uid_list) in train_loader:
            shape_t = shape_np.to(device)
            spec_gt = spec_np.to(device)
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
        val_sum = 0.0
        val_count = 0
        with torch.no_grad():
            for (spec_np, shape_np, uid_list) in val_loader:
                bsz = shape_np.size(0)
                st = shape_np.to(device)
                sg = spec_np.to(device)
                sd = model(st)
                v = crit_mse(sd, sg) * bsz
                val_sum += v.item()
                val_count += bsz
        avg_val = val_sum / val_count
        val_losses.append(avg_val)
        scheduler.step(avg_val)
        if (epoch + 1) % 50 == 0 or epoch == 0 or epoch == (num_epochs - 1):
            print("[Stage A] Epoch[{}/{}] => trainLoss={:.4f}, valLoss={:.4f}"
                  .format(epoch + 1, num_epochs, avg_train, avg_val))
    np.savetxt(os.path.join(out_dir, "train_losses_stageA.csv"), np.array(train_losses), delimiter=",")
    np.savetxt(os.path.join(out_dir, "val_losses_stageA.csv"), np.array(val_losses), delimiter=",")
    plt.figure()
    plt.plot(train_losses, label='train')
    plt.plot(val_losses, label='val')
    plt.xlabel("Epoch")
    plt.ylabel("MSE( shape->spec )")
    plt.title("Stage A: shape->spec training")
    plt.legend()
    plt.savefig(os.path.join(out_dir, "training_curve_stageA.png"))
    plt.close()
    final_val = val_losses[-1]
    print("[Stage A] final val loss =>", final_val)
    shape2spec_path = os.path.join(out_dir, "shape2spec_stageA.pt")
    torch.save(model.state_dict(), shape2spec_path)
    print("[Stage A] model saved =>", shape2spec_path)
    return shape2spec_path, ds_val, model

def train_stageB_spec2shape_frozen(data_source, out_dir, shape2spec_ckpt, num_epochs=500,
                                   batch_size=1024, lr=1e-4, weight_decay=1e-5,
                                   split_ratio=0.8, grad_clip=1.0, use_preprocessed=False):
    os.makedirs(out_dir, exist_ok=True)
    print("[Stage B] =>", out_dir)
    shape2spec_frozen = ShapeToSpectraModel(d_in=3, d_model=256, nhead=4, num_layers=4)
    shape2spec_frozen.load_state_dict(torch.load(shape2spec_ckpt))
    print("[Stage B] shape2spec loaded from =>", shape2spec_ckpt)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    shape2spec_frozen.to(device)
    for p in shape2spec_frozen.parameters():
        p.requires_grad = False
    spec2shape_net = Spectra2ShapeVarLen(d_in=100, d_model=256, nhead=4, num_layers=4).to(device)
    pipeline = Spec2ShapeFrozen(spec2shape_net, shape2spec_frozen).to(device)
    if use_preprocessed:
        ds_full = PreprocessedSpectraDataset(data_source)
    else:
        ds_full = Q1ShiftedSpectraDataset(data_source)
    ds_len = len(ds_full)
    trn_len = int(ds_len * split_ratio)
    val_len = ds_len - trn_len
    ds_train, ds_val = random_split(ds_full, [trn_len, val_len])
    print("[DATA: Stage B] total={}, train={}, val={}".format(ds_len, trn_len, val_len))
    train_loader = DataLoader(ds_train, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(ds_val, batch_size=batch_size, shuffle=False, drop_last=False)
    optimizer = torch.optim.AdamW(spec2shape_net.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                           factor=0.5, patience=10, verbose=True)
    crit_mse = nn.MSELoss()
    train_losses, val_losses = [], []
    for epoch in range(num_epochs):
        pipeline.train()
        run_loss = 0.0
        for (spec_gt_np, shape_gt_np, uid_list) in train_loader:
            spec_gt = spec_gt_np.to(device)
            shape_gt = shape_gt_np.to(device)
            shape_pd, spec_chain = pipeline(spec_gt)
            loss_shape = crit_mse(shape_pd, shape_gt)
            loss_spec = crit_mse(spec_chain, spec_gt)
            loss = loss_shape + loss_spec
            optimizer.zero_grad()
            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(spec2shape_net.parameters(), grad_clip)
            optimizer.step()
            run_loss += loss.item()
        avg_train = run_loss / len(train_loader)
        train_losses.append(avg_train)
        pipeline.eval()
        val_sum = 0.0
        val_count = 0
        with torch.no_grad():
            for (spec_gt_np, shape_gt_np, uid_list) in val_loader:
                bsz = spec_gt_np.size(0)
                sgt = spec_gt_np.to(device)
                shg = shape_gt_np.to(device)
                sp, sc = pipeline(sgt)
                v = (crit_mse(sp, shg) + crit_mse(sc, sgt)) * bsz
                val_sum += v.item()
                val_count += bsz
        avg_val = val_sum / val_count
        val_losses.append(avg_val)
        scheduler.step(avg_val)
        if (epoch + 1) % 50 == 0 or epoch == 0 or epoch == (num_epochs - 1):
            print("[Stage B] Epoch[{}/{}] => trainLoss={:.4f}, valLoss={:.4f}"
                  .format(epoch + 1, num_epochs, avg_train, avg_val))
    np.savetxt(os.path.join(out_dir, "train_losses_stageB.csv"), np.array(train_losses), delimiter=",")
    np.savetxt(os.path.join(out_dir, "val_losses_stageB.csv"), np.array(val_losses), delimiter=",")
    plt.figure()
    plt.plot(train_losses, label='train')
    plt.plot(val_losses, label='val')
    plt.xlabel("Epoch")
    plt.ylabel("Loss= MSE(shape_pred, shape_gt)+ MSE(spec_chain, spec_gt)")
    plt.title("Stage B: spec->shape (frozen shape2spec)")
    plt.legend()
    plt.savefig(os.path.join(out_dir, "training_curve_stageB.png"))
    plt.close()
    final_val = val_losses[-1]
    print("[Stage B] final val =>", final_val)
    spec2shape_path = os.path.join(out_dir, "spec2shape_stageB.pt")
    torch.save(spec2shape_net.state_dict(), spec2shape_path)
    print("[Stage B] spec2shape saved =>", spec2shape_path)
    return spec2shape_path, ds_val, shape2spec_frozen, spec2shape_net

###############################################################################
# MAIN FUNCTION
###############################################################################
def parse_args():
    parser = argparse.ArgumentParser(description="Two-stage pipeline with preprocessed dataset support.")
    parser.add_argument("--preprocess", action="store_true",
                        help="Run preprocessing to generate preprocessed dataset from CSVs.")
    parser.add_argument("--input_folder", type=str, default="",
                        help="Folder containing CSV files for preprocessing.")
    parser.add_argument("--output_npz", type=str, default="preprocessed_data.npz",
                        help="Output NPZ file for preprocessed data.")
    parser.add_argument("--data_npz", type=str, default="",
                        help="Preprocessed dataset file to use for training/testing.")
    parser.add_argument("--test", action="store_true", help="Run in test mode.")
    parser.add_argument("--csv_file", type=str, default="",
                        help="CSV file to use if not using preprocessed dataset.")
    return parser.parse_args()

def main():
    args = parse_args()
    # If preprocessing is requested, run it and exit.
    if args.preprocess:
        if not args.input_folder:
            print("Error: --input_folder must be specified for preprocessing.")
            return
        preprocess_csv_folder(args.input_folder, args.output_npz)
        return
    # Decide whether to use preprocessed data or not.
    use_preprocessed = False
    data_source = None
    if args.data_npz:
        if not os.path.isfile(args.data_npz):
            print("Error: Preprocessed dataset file not found:", args.data_npz)
            return
        use_preprocessed = True
        data_source = args.data_npz
    elif args.csv_file:
        data_source = args.csv_file
    else:
        print("Error: Must specify either --data_npz or --csv_file for training.")
        return
    # For simplicity, this script only implements training (test mode is not implemented here)
    if not args.test:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_out = "outputs_two_stage_" + timestamp
        os.makedirs(base_out, exist_ok=True)
        # Stage A training
        stageA_dir = os.path.join(base_out, "stageA")
        shape2spec_ckpt, ds_valA, modelA = train_stageA_shape2spec(
            data_source=data_source,
            out_dir=stageA_dir,
            num_epochs=10,
            batch_size=1024*32,
            lr=1e-4,
            # weight_decay=1e-5,
            weight_decay=1,
            split_ratio=0.8,
            grad_clip=1.0,
            use_preprocessed=use_preprocessed
        )
        # (Optional) Visualization for Stage A can be added here.
        # Stage B training
        stageB_dir = os.path.join(base_out, "stageB")
        spec2shape_ckpt, ds_valB, shape2spec_froz, spec2shape_net = train_stageB_spec2shape_frozen(
            data_source=data_source,
            out_dir=stageB_dir,
            shape2spec_ckpt=shape2spec_ckpt,
            num_epochs=10,
            batch_size=4*1024,
            lr=1e-4,
            # weight_decay=1e-5,
            weight_decay=1,
            split_ratio=0.8,
            grad_clip=1.0,
            use_preprocessed=use_preprocessed
        )
    else:
        print("Test mode not implemented in this script.")

if __name__ == "__main__":
    main()

