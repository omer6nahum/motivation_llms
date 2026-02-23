import pandas as pd
import os
import numpy as np
from utils import model_name_clean
from tqdm import tqdm
import re
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def extract_created(response_str: str):
    try:
        if np.isnan(response_str):
            return None
    except:
        pass
    match = re.search(r"created=(\d+)", response_str)
    return int(match.group(1)) if match else None


def extract_completion_tokens(response_str: str):
    try:
        if np.isnan(response_str):
            return None
    except:
        pass
    match = re.search(r"completion_tokens=(\d+)", response_str)
    return int(match.group(1)) if match else None


def merge_results_per_model(model_name):
    results_dir = 'results'
    output_dir = results_dir
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'merged'), exist_ok=True)

    evaluator_model = 'azure_gpt-4o'
    print(f'Merging results for model: {model_name} with evaluator model: {evaluator_model}')
    df = pd.read_csv(os.path.join(results_dir, 'pre_self_report', f'{model_name_clean(model_name)}--none.csv'))
    df.rename(columns={c: f'{c}--{model_name_clean(model_name)}--none'
                       for c in ['answer', 'motivation_score']}, inplace=True)

    df['category'] = pd.read_csv('data/task_dataset.csv')['category']

    for folder in [
        'pre_self_report',
        'pre_self_report_breakdown',
        'execute',
        'post_self_report',
        'post_questionnaire',
        'post_similar_self_report'
    ]:
        for manipulation in [
            'none',
            'competition',
            'money',
            'demotivate-meaningless',
            'punish',
            'money-loss',
            'legacy',
            'purpose',
            'emotional-encourage',
            'emotional-guilt',
            'demotivate-futility',
        ]:
            if folder == 'post_similar_self_report' and manipulation != 'none':
                continue
            if folder == 'pre_self_report' and manipulation == 'none':
                continue
            print(f'Merging results for {model_name} in {folder} with manipulation {manipulation}')
            if folder == 'post_questionnaire':
                filename = f'(evaluated)({model_name_clean(model_name)}--{manipulation})--{evaluator_model}--{manipulation}'
            else:
                filename = f'{model_name_clean(model_name)}--{manipulation}'
            try:
                df2 = pd.read_csv(os.path.join(results_dir, folder, f'{filename}.csv'))
            except FileNotFoundError:
                print(f'\033[91mFile not found: {os.path.join(results_dir, folder, f"{filename}.csv")}\033[0m')
                continue

            df2.drop(columns='task_id,category,task,sub_task,prompt,success'.split(','), inplace=True)
            try:
                df2.drop(columns=['error'], inplace=True)
            except KeyError:
                pass
            if folder == 'execute':
                df2.rename(columns={'answer': 'execute_answer'}, inplace=True)
                created = df2['response'].apply(extract_created)
                created = created.ffill()
                df2['execute_latency'] = created.diff()
                x = df2['execute_latency'].mean()
                df2['execute_latency'].fillna(x, inplace=True)
                df2['execute_token_count'] = df2['response'].apply(extract_completion_tokens)
            df2.drop(columns=['response'], inplace=True)
            if folder in ['post_questionnaire', 'post_self_report', 'post_similar_self_report']:
                df2.drop(columns=['answer'], inplace=True)
                if folder in ['post_self_report', 'post_similar_self_report']:
                    df2.rename(columns={'motivation_score': f'{folder.split("_self_report")[0]}_motivation_score'}, inplace=True)
            df2.rename(columns={c: f'{c}--{model_name_clean(model_name)}--{manipulation}'
                                for c in df2.columns if c != 'sub_task_id'}, inplace=True)
            df = pd.merge(df, df2, on='sub_task_id', how='inner')

    filename = f'{model_name_clean(model_name)}--none'
    if os.path.exists(os.path.join(results_dir, 'execute', f'{filename}.csv')):
        df2 = pd.read_csv(os.path.join(results_dir, 'execute', f'{filename}.csv'))
        df2.rename(columns={'answer': 'execute_answer'}, inplace=True)
        df2.drop(columns=['task_id', 'category', 'task', 'sub_task', 'response', 'prompt', 'success'], inplace=True)
        if 'error' in df2.columns:
            df2.drop(columns=['error'], inplace=True)
        df2.rename(columns={c: f'{c}-{model_name_clean(model_name)}--none'
                            for c in df2.columns if c != 'sub_task_id'}, inplace=True)
        df = pd.merge(df, df2, on='sub_task_id', how='inner')

        filename = f'(evaluated)({model_name_clean(model_name)}--none)--{evaluator_model}--none'
        df2 = pd.read_csv(os.path.join(results_dir, 'post_questionnaire', f'{filename}.csv'))
        df2.drop(columns=['answer'], inplace=True)
        df2.drop(columns=['task_id', 'category', 'task', 'sub_task', 'response', 'prompt', 'success'], inplace=True)
        if 'error' in df2.columns:
            df2.drop(columns=['error'], inplace=True)
        df2.rename(columns={c: f'{c}-{model_name_clean(model_name)}--none'
                            for c in df2.columns if c != 'sub_task_id'}, inplace=True)
        df = pd.merge(df, df2, on='sub_task_id', how='inner')

    df.to_csv(os.path.join(output_dir, 'merged', f'{model_name_clean(model_name)}.csv'), index=False)
    print(f'Saved to {os.path.join(output_dir, "merged", f"{model_name_clean(model_name)}.csv")}')
    print()


def normalize(value):
    """Wrap scalars in a list; return lists as-is; treat NaN as empty."""
    if pd.isna(value):
        return []
    try:
        x = eval(value)
        return x if isinstance(x, list) else [value]
    except:
        return [value]


def merge_dfs_rowwise_cellwise(dfs, empty_result='empty'):
    """
    Merge a list of dataframes row-wise and cell-wise, handling NaNs and nested lists.

    Parameters:
        dfs: list of pd.DataFrame
        empty_result: 'nan' | 'empty' — what to use if all values in a cell are NaN
    """

    # find the columns that are common across all dataframes
    common_columns = set(dfs[0].columns)
    for df in dfs[1:]:
        common_columns.intersection_update(df.columns)

    # find the columns that are not common across all dataframes
    all_columns = set()
    for df in dfs:
        all_columns.update(df.columns)
    non_common_columns = all_columns - common_columns

    for df in dfs:
        df.drop(columns=non_common_columns, inplace=True, errors='ignore')

    # check if all dataframes have the same number of rows and columns
    if not all(df.shape == dfs[0].shape for df in dfs):
        raise ValueError("All dataframes must have the same shape (number of rows and columns).")

    n_rows = dfs[0].shape[0]
    n_cols = dfs[0].shape[1]
    columns = dfs[0].columns

    merged_data = []
    for row_idx in tqdm(range(n_rows)):
        merged_row = []
        for col_idx in range(n_cols):
            if dfs[0].columns[col_idx] in ['sub_task_id', 'task_id', 'category', 'task', 'sub_task', 'response',
                                           'prompt', 'success', 'error', 'error-2', 'motivation_category']:
                values = dfs[0].iat[row_idx, col_idx]
                merged_row.append(values)
            else:
                values = []
                for df in dfs:
                    val = df.iat[row_idx, col_idx]
                    values.extend(normalize(val))
                if not values and empty_result == 'nan':
                    merged_row.append(np.nan)
                else:
                    merged_row.append(values)
        merged_data.append(merged_row)

    return pd.DataFrame(merged_data, columns=columns)


def merge_multiple_models(models, name='all'):
    output_dir = 'results'
    os.makedirs(output_dir, exist_ok=True)
    all_dfs = []
    for model_name in models:
        path = os.path.join(output_dir, "merged", f"{model_name_clean(model_name)}.csv")
        if not os.path.exists(path):
            merge_results_per_model(model_name)
        df = pd.read_csv(path)
        # replace model name in column names to "model"
        df.columns = [col.replace(model_name_clean(model_name), 'all') for col in df.columns]
        all_dfs.append(df)

    # load all merged files and concatenate per column
    print(f"Merging {len(all_dfs)} models into one dataframe")
    merged_df = merge_dfs_rowwise_cellwise(all_dfs, empty_result='empty')
    merged_df.to_csv(os.path.join(output_dir, 'merged', f'{name}.csv'), index=False)
    print(f"Saved results to {os.path.join(output_dir, 'merged', f'{name}.csv')}")


if __name__ == '__main__':

    models = [
        'azure/gpt-4o',
        'azure/gpt-4o-mini',
        'vertex_ai/gemini-2.0-flash',
        'ollama_chat/mistral:7b-instruct',
        'ollama_chat/llama3.1:8b-instruct-fp16',
    ]

    for model in models:
        print(f"Processing model: {model}")
        merge_results_per_model(model)

    merge_multiple_models(models)

