
@echo off

python prep_fingerprint_csv.py --floor-dir .\site1\B1 --source uncal_debiased --window-ms 1200 --hop-ms 200 --min-mag-pts 3 --out b1_dataset_dense.csv
python prep_fingerprint_csv.py --floor-dir .\site1\F1 --source uncal_debiased --window-ms 1200 --hop-ms 200 --min-mag-pts 3 --out f1_dataset_dense.csv
python prep_fingerprint_csv.py --floor-dir .\site1\F2 --source uncal_debiased --window-ms 1200 --hop-ms 200 --min-mag-pts 3 --out f2_dataset_dense.csv
python prep_fingerprint_csv.py --floor-dir .\site1\F3 --source uncal_debiased --window-ms 1200 --hop-ms 200 --min-mag-pts 3 --out f3_dataset_dense.csv
python prep_fingerprint_csv.py --floor-dir .\site1\F4 --source uncal_debiased --window-ms 1200 --hop-ms 200 --min-mag-pts 3 --out f4_dataset_dense.csv


python prep_fingerprint_csv.py --floor-dir .\site2\B1 --source uncal_debiased --window-ms 1200 --hop-ms 200 --min-mag-pts 3 --out 2b1_dataset_dense.csv

python prep_fingerprint_csv.py --floor-dir .\site2\F1 --source uncal_debiased --window-ms 1200 --hop-ms 200 --min-mag-pts 3 --out 2f1_dataset_dense.csv
python prep_fingerprint_csv.py --floor-dir .\site2\F2 --source uncal_debiased --window-ms 1200 --hop-ms 200 --min-mag-pts 3 --out 2f2_dataset_dense.csv
python prep_fingerprint_csv.py --floor-dir .\site2\F3 --source uncal_debiased --window-ms 1200 --hop-ms 200 --min-mag-pts 3 --out 2f3_dataset_dense.csv
python prep_fingerprint_csv.py --floor-dir .\site2\F4 --source uncal_debiased --window-ms 1200 --hop-ms 200 --min-mag-pts 3 --out 2f4_dataset_dense.csv


python prep_fingerprint_csv.py --floor-dir .\site2\F5 --source uncal_debiased --window-ms 1200 --hop-ms 200 --min-mag-pts 3 --out 2f5_dataset_dense.csv
python prep_fingerprint_csv.py --floor-dir .\site2\F6 --source uncal_debiased --window-ms 1200 --hop-ms 200 --min-mag-pts 3 --out 2f6_dataset_dense.csv
python prep_fingerprint_csv.py --floor-dir .\site2\F7 --source uncal_debiased --window-ms 1200 --hop-ms 200 --min-mag-pts 3 --out 2f7_dataset_dense.csv
python prep_fingerprint_csv.py --floor-dir .\site2\F8 --source uncal_debiased --window-ms 1200 --hop-ms 200 --min-mag-pts 3 --out 2f8_dataset_dense.csv
