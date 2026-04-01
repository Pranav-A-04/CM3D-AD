import argparse
import os
import pandas as pd
import re

def main(args):
    all_df = []

    for cate_dir in os.listdir(args.work_path):
        full_path = os.path.join(args.work_path, cate_dir)
        if not os.path.isdir(full_path):
            continue

        log_path = os.path.join(full_path, "log.txt")
        if not os.path.exists(log_path):
            print(f"Skipping {cate_dir} — log.txt not found.")
            continue

        with open(log_path, 'r') as file:
            text = file.read()

        max_roci = 0.0
        max_rocp = 0.0
        max_api = 0.0
        max_app = 0.0

        # Modified regex to allow for suffixes like _cdist or _nn
        pattern = r"ROC_i_\w+ ([\d\.]+) \| ROC_p_\w+ ([\d\.]+) \| AP_i_\w+ ([\d\.]+) \| AP_p_\w+ ([\d\.]+)"

        for line in text.split('\n'):
            match = re.search(pattern, line)
            if match:
                roc_i, roc_p, ap_i, ap_p = match.groups()
                max_roci = max(max_roci, float(roc_i))
                max_rocp = max(max_rocp, float(roc_p))
                max_api = max(max_api, float(ap_i))
                max_app = max(max_app, float(ap_p))

        df = pd.DataFrame({
            'I-AUROC': [max_roci],
            'P-AUROC': [max_rocp],
            'I-AP': [max_api],
            'P-AP': [max_app]
        })
        df.index = [cate_dir.split('_')[0]]
        all_df.append(df)

    if not all_df:
        print("No valid results found.")
        return

    all_df = pd.concat(all_df)
    all_df.sort_index(inplace=True)

    print(f"\nEnsembled results of {args.work_path}:\n")
    print(all_df.mean(0))

    output_path = os.path.join(args.work_path, "ensemble.csv")
    all_df.T.to_csv(output_path)
    print(f"\nSaved CSV to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("work_path", help="Path to the directory containing category subfolders with log.txt files.")
    args = parser.parse_args()

    main(args)

