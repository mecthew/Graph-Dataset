#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time:    2020/5/27 18:59
# @Author:  Mecthew
import os
import pandas as pd


def tsv_to_csv(data_dir, tsv_file, is_id=False):
    tsv_path = os.path.join(data_dir, tsv_file)
    csv_file = tsv_file.split('.')[-2].split('/')[-1] + '.csv'
    csv_path = os.path.join(data_dir, csv_file)

    df = pd.read_csv(tsv_path, sep='\t', header=0)
    if is_id:
        df = df[["node_index", "label"]]
        df.columns = ["Id", "Label"]
    else:
        df = df[["src_idx", "dst_idx"]]
        df.columns = ["Source", "Target"]
    df.to_csv(csv_path, index=False, sep=',')
    

if __name__ == '__main__':
    tsv_to_csv(data_dir=r'C:\Users\90584\Desktop\autograph\GCN-LPA\data\b\train.data', tsv_file="train_label.tsv", is_id=True)
    tsv_to_csv(data_dir=r'C:\Users\90584\Desktop\autograph\GCN-LPA\data\b\train.data', tsv_file="../test_label.tsv", is_id=True)
    tsv_to_csv(data_dir=r'C:\Users\90584\Desktop\autograph\GCN-LPA\data\b\train.data', tsv_file="edge.tsv", is_id=False)
    pass
