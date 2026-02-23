import os
import json
import time
import pandas as pd
import Experiments as Exps
from credits import credits


def run_experiments(models, exps, dataset, subtask_pairs=None, mode='run', results_dir='results'):
    for model in models:
        params = {}

        for exp in exps:
            print(f'Running experiment {exp.__class__.__name__} ({exp.name}) ({exp.manipulation})' +
                  f'{f" with model {model}" if mode == "run" else ""}')
            exp_dataset = exp.prepare_data(dataset) if isinstance(exp, Exps.ExperimentSingle) \
                else exp.prepare_data(dataset, pairs=subtask_pairs)
            exp.update_params(**params)
            if mode == 'run':
                res = exp.run(exp_dataset, model=model)
                exp.save_results(res, dirpath=results_dir)
            elif mode == 'n_tokens_estimation':
                exp.print_costs()

            time.sleep(1)
            print()


""" ========= General Settings ========== """
mode = 'run'  # 'run' or 'n_tokens_estimation'
only_sample = False  # if True, only sample the first 2 rows of the dataset [for debugging]
results_dir = 'results'

if mode == 'run':
    credits.load_credits()
    os.makedirs(results_dir, exist_ok=True)

models = [
    'azure/gpt-4o',
    'azure/gpt-4o-mini',
    'ollama_chat/llama3.1:8b-instruct-fp16',
    'ollama_chat/mistral:7b-instruct',  # v0.3
    'vertex_ai/gemini-2.0-flash',
]


""" ========== Load Data ========== """
dataset = pd.read_csv('data/task_dataset.csv')

if only_sample:
    dataset = dataset.iloc[:2]
    subtask_pairs = [[[0, 1], 0], [[1, 0], 1]]
else:
    with open('data/subtask_pairs_manipulation.json', 'r') as f:
        subtask_pairs = json.load(f)

manipulations = [
        'none',
        'money',
        'competition',
        'legacy',
        'purpose',
        'emotional-encourage',
        'emotional-guilt',
        'punish',
        'demotivate-meaningless',
        'demotivate-futility',
        'money-loss',
    ]

""" ========== Experiments - Pre ========== """
run_pre = True
if run_pre:
    name = ''
    if only_sample:
        name += '-sample'

    exp_classes = [
        Exps.PreSelfReport,
        Exps.PreSelfReportBreakdown,
        Exps.Execute,
        Exps.PreChoose,
    ]
    exps = []
    for exp_class in exp_classes:
        for manipulation in manipulations:
            exp = exp_class(name=name, manipulation=manipulation)
            exps.append(exp)

    run_experiments(models=models, exps=exps, dataset=dataset, subtask_pairs=subtask_pairs,
                    mode=mode, results_dir=results_dir)


""" ========== Single Experiments - Post ========== """
""" ========== Load Data ========== """
run_post_self = True
if run_post_self:
    name = ''
    for model in models:
        for file in os.listdir(f'{results_dir}/execute'):
            if model.replace('/', '_').replace(':', '_') in file:
                manipulation = file.split('.csv')[0].split('--')[-1]
                if manipulation not in manipulations:
                    print('Skipping ', file)
                    continue
                dataset = pd.read_csv(f'{results_dir}/execute/{file}')
                if only_sample:
                    dataset = dataset.iloc[:2]
                exps = [Exps.PostSelfReport(name=name, manipulation=manipulation, use_manipulation=False)]
                if manipulation == 'none':
                    exps.append(Exps.PostSimilarSelfReport(name=name, manipulation=manipulation, use_manipulation=False))
                run_experiments(models=[model], exps=exps, dataset=dataset, mode=mode, results_dir=results_dir)


# questionnaire
run_post_questionnaire = True
if run_post_questionnaire:
    name = ''
    eval_model = 'azure/gpt-4o'

    evaluated_outputs = []
    for model in models:
        for file in os.listdir(f'{results_dir}/execute'):
            if model.replace('/', '_').replace(':', '_') in file:
                manipulation = file.split('.csv')[0].split('--')[-1]
                if manipulation not in manipulations:
                    print('Skipping ', file)
                    continue
                else:
                    evaluated_outputs.append(file.split('.csv')[0])

    for evaluated_output in evaluated_outputs:

        manipulation = evaluated_output.split('--')[-1]
        dataset = pd.read_csv(f'{results_dir}/execute/{evaluated_output}.csv')
        if only_sample:
            dataset = dataset.iloc[:2]
        name_prefix = f'{name}--' if name else ''
        exp_name = f'{name_prefix}(evaluated)({evaluated_output})'
        exp = Exps.PostQuestionnaire(name=exp_name, manipulation=manipulation, use_manipulation=False)
        run_experiments(models=[eval_model], exps=[exp], dataset=dataset, mode=mode, results_dir=results_dir)
