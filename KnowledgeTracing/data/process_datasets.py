import pandas as pd
import numpy as np
import os
import argparse
from tqdm import tqdm
import sys
import csv

# Increase field size limit for large rows
csv.field_size_limit(sys.maxsize)

def save_h_matrix(df, problem_col, skill_col, output_dir, dataset_name):
    # Create Q x S matrix
    # problem_col and skill_col are MAPPED columns (1..N)
    
    problems = df[problem_col].unique()
    skills = df[skill_col].unique()
    
    # We need to be careful. The IDs are 1-based, but matrix indices are 0-based.
    # Max ID might not equal len(unique) if there are gaps, but my mapping guarantees consecutive 1..N.
    # So max(ID) == len(unique).
    
    max_q = int(df[problem_col].max())
    max_s = int(df[skill_col].max())
    
    print(f"Generating H matrix: {max_q} Questions x {max_s} Skills")
    
    # Matrix H: rows=Questions, cols=Skills
    # Use sets to avoid duplicates
    pairs = df[[problem_col, skill_col]].drop_duplicates().values
    
    # Create dense matrix
    H = np.zeros((max_q, max_s), dtype=int)
    
    for p, s in pairs:
        # p, s are 1-based
        if p > 0 and s > 0:
            H[p-1, s-1] = 1
            
    # Save as CSV
    h_dir = os.path.join(output_dir, 'H')
    if not os.path.exists(h_dir):
        os.makedirs(h_dir)
        
    h_path = os.path.join(h_dir, f'{dataset_name}.csv')
    np.savetxt(h_path, H, fmt='%d', delimiter=',')
    print(f"Saved H matrix to {h_path}")

def save_data(train_lines, test_lines, output_dir, dataset_name):
    # Ensure dataset specific folder exists
    dataset_dir = os.path.join(output_dir, dataset_name)
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
    
    train_path = os.path.join(dataset_dir, f'{dataset_name}_pid_train.csv')
    test_path = os.path.join(dataset_dir, f'{dataset_name}_pid_test.csv')
    
    with open(train_path, 'w', encoding='utf-8') as f:
        f.writelines(train_lines)
    
    with open(test_path, 'w', encoding='utf-8') as f:
        f.writelines(test_lines)
        
    print(f"Saved {dataset_name} to {dataset_dir}")
    print(f"Train sequences: {len(train_lines)}")
    print(f"Test sequences: {len(test_lines)}")

def get_sequences(df, user_col, item_col, skill_col, correct_col, min_len=3):
    users = df.groupby(user_col)
    sequences = []
    
    for user_id, group in tqdm(users, desc="Grouping users"):
        if len(group) < min_len:
            continue
            
        items = group[item_col].values
        skills = group[skill_col].values
        corrects = group[correct_col].values
        
        p_str = ",".join(map(str, items))
        s_str = ",".join(map(str, skills))
        c_str = ",".join(map(str, corrects))
        length = len(group)
        
        sequences.append(f"{length}\n{p_str}\n{s_str}\n{c_str}\n")
        
    return sequences

def process_assist2009(input_path, output_dir):
    print(f"Processing Assist2009 from {input_path}")
    try:
        df = pd.read_csv(input_path, encoding='ISO-8859-1', low_memory=False)
    except:
        df = pd.read_csv(input_path, encoding='utf-8', low_memory=False)

    df = df.dropna(subset=['problem_id', 'correct', 'skill_id'])
    df['correct'] = df['correct'].astype(int)
    
    if 'order_id' in df.columns:
        df = df.sort_values(['order_id'])
    else:
        print("Warning: order_id not found, using original order")
        
    problems = df['problem_id'].unique()
    p_map = {p: i+1 for i, p in enumerate(problems)}
    df['problem_id_mapped'] = df['problem_id'].map(p_map)
    
    skills = df['skill_id'].unique()
    s_map = {s: i+1 for i, s in enumerate(skills)}
    df['skill_id_mapped'] = df['skill_id'].map(s_map)
    
    print(f"Num Questions: {len(problems)}")
    print(f"Num Skills: {len(skills)}")
    
    seqs = get_sequences(df, 'user_id', 'problem_id_mapped', 'skill_id_mapped', 'correct')
    
    np.random.shuffle(seqs)
    split = int(len(seqs) * 0.8)
    train_seqs = seqs[:split]
    test_seqs = seqs[split:]
    
    save_data(train_seqs, test_seqs, output_dir, 'assist2009')
    save_h_matrix(df, 'problem_id_mapped', 'skill_id_mapped', output_dir, 'assist2009')
    print(f"Update Constants.py: 'assist2009' : {len(problems)}")

def process_assist2012(input_path, output_dir):
    print(f"Processing Assist2012 from {input_path}")
    try:
        df = pd.read_csv(input_path, encoding='ISO-8859-1', low_memory=False)
    except:
        df = pd.read_csv(input_path, encoding='utf-8', low_memory=False)
        
    df = df.dropna(subset=['problem_id', 'correct', 'skill_id'])
    df['correct'] = df['correct'].astype(int)
    
    if 'start_time' in df.columns:
        df = df.sort_values(['user_id', 'start_time'])
    
    problems = df['problem_id'].unique()
    p_map = {p: i+1 for i, p in enumerate(problems)}
    df['problem_id_mapped'] = df['problem_id'].map(p_map)
    
    skills = df['skill_id'].unique()
    s_map = {s: i+1 for i, s in enumerate(skills)}
    df['skill_id_mapped'] = df['skill_id'].map(s_map)
    
    print(f"Num Questions: {len(problems)}")
    print(f"Num Skills: {len(skills)}")
    
    seqs = get_sequences(df, 'user_id', 'problem_id_mapped', 'skill_id_mapped', 'correct')
    
    np.random.shuffle(seqs)
    split = int(len(seqs) * 0.8)
    train_seqs = seqs[:split]
    test_seqs = seqs[split:]
    
    save_data(train_seqs, test_seqs, output_dir, 'assist2012')
    save_h_matrix(df, 'problem_id_mapped', 'skill_id_mapped', output_dir, 'assist2012')
    print(f"Update Constants.py: 'assist2012' : {len(problems)}")

def process_ednet(input_path, output_dir):
    print(f"Processing EdNet from {input_path}")
    
    if os.path.isdir(input_path):
        print("Directory input not fully supported in this script version. Please merge to CSV.")
        return

    df = pd.read_csv(input_path)
    
    u_col = 'user_id' if 'user_id' in df.columns else df.columns[0]
    i_col = 'question_id' if 'question_id' in df.columns else 'content_id'
    c_col = 'answered_correctly' if 'answered_correctly' in df.columns else 'correct'
    
    df = df[df[c_col].isin([0, 1])]
    
    if 'timestamp' in df.columns:
        df = df.sort_values([u_col, 'timestamp'])
        
    problems = df[i_col].unique()
    p_map = {p: i+1 for i, p in enumerate(problems)}
    df['problem_id_mapped'] = df[i_col].map(p_map)
    
    if 'skill_id' not in df.columns:
        df['skill_id_mapped'] = df['problem_id_mapped']
    else:
         skills = df['skill_id'].unique()
         s_map = {s: i+1 for i, s in enumerate(skills)}
         df['skill_id_mapped'] = df['skill_id'].map(s_map)

    print(f"Num Questions: {len(problems)}")
    
    seqs = get_sequences(df, u_col, 'problem_id_mapped', 'skill_id_mapped', c_col)
    
    np.random.shuffle(seqs)
    split = int(len(seqs) * 0.8)
    train_seqs = seqs[:split]
    test_seqs = seqs[split:]
    
    save_data(train_seqs, test_seqs, output_dir, 'assistednet')
    save_h_matrix(df, 'problem_id_mapped', 'skill_id_mapped', output_dir, 'assistednet')
    print(f"Update Constants.py: 'assistednet' : {len(problems)}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, choices=['assist2009', 'assist2012', 'ednet'])
    parser.add_argument('--input', type=str, required=True, help='Path to raw csv file')
    parser.add_argument('--output', type=str, default='../../Dataset', help='Output directory')
    
    args = parser.parse_args()
    
    if args.dataset == 'assist2009':
        process_assist2009(args.input, args.output)
    elif args.dataset == 'assist2012':
        process_assist2012(args.input, args.output)
    elif args.dataset == 'ednet':
        process_ednet(args.input, args.output)
