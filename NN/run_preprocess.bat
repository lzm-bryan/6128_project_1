
@echo off


python preprocess_fingerprint_dataset.py --csv b1_dataset_dense.csv --task reg --label-space px --normalize-labels minmax --floor-dir .\site1\B1 --out-dir out_b1_reg


python preprocess_fingerprint_dataset.py --csv f1_dataset_dense.csv --task reg --label-space px --normalize-labels minmax --floor-dir .\site1\F1 --out-dir out_f1_reg

python preprocess_fingerprint_dataset.py --csv f2_dataset_dense.csv --task reg --label-space px --normalize-labels minmax --floor-dir .\site1\F2 --out-dir out_f2_reg

python preprocess_fingerprint_dataset.py --csv f3_dataset_dense.csv --task reg --label-space px --normalize-labels minmax --floor-dir .\site1\F3 --out-dir out_f3_reg

python preprocess_fingerprint_dataset.py --csv f4_dataset_dense.csv --task reg --label-space px --normalize-labels minmax --floor-dir .\site1\F4 --out-dir out_f4_reg



python preprocess_fingerprint_dataset.py --csv 2b1_dataset_dense.csv --task reg --label-space px --normalize-labels minmax --floor-dir .\site2\B1 --out-dir out_2b1_reg


python preprocess_fingerprint_dataset.py --csv 2f1_dataset_dense.csv --task reg --label-space px --normalize-labels minmax --floor-dir .\site2\F1 --out-dir out_2f1_reg

python preprocess_fingerprint_dataset.py --csv 2f2_dataset_dense.csv --task reg --label-space px --normalize-labels minmax --floor-dir .\site2\F2 --out-dir out_2f2_reg

python preprocess_fingerprint_dataset.py --csv 2f3_dataset_dense.csv --task reg --label-space px --normalize-labels minmax --floor-dir .\site2\F3 --out-dir out_2f3_reg

python preprocess_fingerprint_dataset.py --csv 2f4_dataset_dense.csv --task reg --label-space px --normalize-labels minmax --floor-dir .\site2\F4 --out-dir out_2f4_reg



python preprocess_fingerprint_dataset.py --csv 2f5_dataset_dense.csv --task reg --label-space px --normalize-labels minmax --floor-dir .\site2\F5 --out-dir out_2f5_reg

python preprocess_fingerprint_dataset.py --csv 2f6_dataset_dense.csv --task reg --label-space px --normalize-labels minmax --floor-dir .\site2\F6 --out-dir out_2f6_reg

python preprocess_fingerprint_dataset.py --csv 2f7_dataset_dense.csv --task reg --label-space px --normalize-labels minmax --floor-dir .\site2\F7 --out-dir out_2f7_reg

python preprocess_fingerprint_dataset.py --csv 2f8_dataset_dense.csv --task reg --label-space px --normalize-labels minmax --floor-dir .\site2\F8 --out-dir out_2f8_reg
