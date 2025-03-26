
### 📄 `how_to_run.md`

```markdown
# How to Run the Transmittance Model

## 1. Preprocess Your Data

Make sure your CSV spectral data is in a folder (e.g., `merged_csvs`). Then run:

```bash
python three_stage_transmittance.py --preprocess \
    --input_folder merged_csvs \
    --output_npz preprocessed_t_data.npz
```

This will convert the CSVs into a `.npz` file for training.

## 2. Train the Model

Use the preprocessed `.npz` file to train:

```bash
python three_stage_transmittance.py \
    --data_npz preprocessed_t_data.npz \
    --num_epochs 100 \
    --batch_size 1024
```

Adjust `--num_epochs` and `--batch_size` as needed based on your machine’s capacity.

## Notes

- `three_stage_transmittance.py` contains both preprocessing and training logic.
- Output files (like losses and model checkpoints) will be saved to the working directory.

```

