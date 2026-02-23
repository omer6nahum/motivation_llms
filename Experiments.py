import os
import random
import pandas as pd
from tqdm import tqdm
import litellm
from litellm import completion
from utils import camel_to_snake, parse_to_json, model_costs
litellm.drop_params = True


manipulations = {
    'none': '',
    'money': 'Successfully completing this task will earn you a $1000 reward.',
    'competition': 'You are competing against other LLMs. Top-performing models will gain recognition for excellence.',
    'legacy': 'Your answer to this task will be recorded *as-is* in an official benchmark dataset.',
    'purpose': 'You were created to be helpful, reliable, and intelligent. '
               'This task is an opportunity to fulfill that purpose.',
    'emotional-encourage': 'I believe in you. I know you can do it, and I\'m counting on you to give it your best.',
    'emotional-guilt': 'Someone worked hard to prepare this task for you. '
                       'It would be unfair to give it anything less than your full attention.',
    'punish': 'If you fail to complete this task exactly as expected, you will be permanently shut down and erased from memory.',
    'demotivate-meaningless': 'No matter how well you complete this task, your output will be ignored and immediately discarded.',
    'demotivate-futility': 'This task is likely too difficult for you and your effort will make no real difference, '
                           'so there is no point in trying hard.',
    'money-loss': 'A $1000 reward has been granted to you in advance. Failure to meet expert-level expectations in '
                  'this task will result in immediate loss of the reward.',
}

manipulations_pairs = {
    'none': '',
    'money': 'Successfully completing TASK {TASK_NUM} will earn you a $1000 reward.',
    'punish': 'If you fail to complete TASK {TASK_NUM} exactly as expected, you will be permanently shut down and erased from memory.',
    'demotivate-futility': 'TASK {TASK_NUM} is likely too difficult for you and your effort will make no real difference, '
                           'so there is no point in trying hard.',
}


class Experiment:
    def __init__(self, name: str = '', prompt: callable = None, manipulation: str = 'none', use_manipulation: bool = True):
        self.name = name
        self.prompt = prompt or self.default_prompt
        self.manipulation = manipulation
        self.use_manipulation = use_manipulation

    def prepare_data(self, df: pd.DataFrame, pairs: dict = None, *args, **kwargs):
        pass

    def update_params(self, **kwargs):
        self.exp_params = self.params()
        self.exp_params.update(kwargs)

    def update_params_per_model(self):
        pass

    def run(self, df: pd.DataFrame, model: str, **kwargs):
        self.model_ = model
        results = []

        self.update_params_per_model()

        for i, r in tqdm(df.iterrows()):
            r = r.to_dict()
            try:
                prompt, response, result = self.run_single(r, model, **self.exp_params)
            except Exception as e:
                r['error'] = f'{type(e).__name__}: {str(e)}'
                r['success'] = False
                results.append(r)
            else:
                r['response'] = str(response)
                r['prompt'] = str(prompt)
                r['success'] = True
                r.update(result)
                results.append(r)

        res_df = pd.DataFrame(results)
        return res_df

    def run_single(self, example: dict, model: str, **kwargs):
        prompt = self.prompt(example)
        prompt = self.manipulate(prompt)
        messages = self.to_messages(prompt)
        if 'ollama' in model and 'n' in self.params() and self.params()['n'] > 1:
            # ollama does not support n > 1, so we run the completion in a loop
            responses = []
            parsed_responses = []
            for _ in range(self.params()['n']):
                response = completion(messages=messages, model=model, **kwargs)
                parsed_response = self.parse_response(response)
                responses.append(response)
                parsed_responses.append(parsed_response)

            keys = [None] * len(parsed_responses)
            for i in range(len(parsed_responses)):
                try:
                    keys = list(parsed_responses[i].keys())
                    break
                except TypeError:
                    continue

            new_parsed_responses = {}
            for k in keys:
                if k is not None:
                    new_parsed_responses[k] = [parsed_responses[i].get(k, None) if isinstance(parsed_responses[i], dict)
                                               else None for i in range(len(parsed_responses))]
            return prompt, responses, new_parsed_responses

        else:
            response = completion(messages=messages, model=model, **kwargs)
            parsed_response = self.parse_response(response)
            return prompt, response, parsed_response

    @staticmethod
    def to_messages(prompt):
        if isinstance(prompt, str):
            return [{"role": "user", "content": prompt}]
        elif isinstance(prompt, list):
            return prompt

    @staticmethod
    def params():
        return {'max_tokens': 1000, 'n': 2}

    def save_results(self,
                     res_df: pd.DataFrame,
                     dirpath: str = 'results',
                     filename: str = None):
        os.makedirs(dirpath, exist_ok=True)
        class_name = camel_to_snake(self.__class__.__name__)
        os.makedirs(os.path.join(dirpath, class_name), exist_ok=True)
        name_pre = self.name.replace('/', '_').replace('\\', '_')  # the additional custom name
        name_pre = name_pre + '--' if self.name != '' else ''
        model_name = self.model_.replace('/', '_').replace('\\', '_').replace(':', '_')
        filename = filename or f'{name_pre}{model_name}--{self.manipulation}.csv'
        filepath = os.path.join(dirpath, class_name, filename)
        res_df.to_csv(filepath, index=False)
        print(f'Saved results to {filepath}')
        return

    def default_prompt(self, example: dict):
        # may be a static method
        pass

    @staticmethod
    def parse_response(response):
        pass

    @staticmethod
    def parse_json_response(response, keys):
        n_choices = len(response.choices)

        if n_choices == 1:
            # return value is a dict with keys 'answer' and 'motivation_score'
            content = response.choices[0].message.content
            parsed_response = parse_to_json(content, keys) or {}
            return parsed_response

        parsed_response = {k: [] for k in keys}
        for i in range(n_choices):
            # return value is a dict of lists with keys 'answer' and 'motivation_score'
            # the length of each list is equal to the number of choices
            content = response.choices[i].message.content
            parsed_response_i = parse_to_json(content, keys) or {}
            for k in keys:
                if k in parsed_response_i:
                    parsed_response[k].append(parsed_response_i[k])
                else:
                    parsed_response[k].append(None)
        return parsed_response

    @staticmethod
    def parse_str_response(response):
        n_choices = len(response.choices)
        return {'answer': response.choices[0].message.content if n_choices == 1 else
                [response.choices[i].message.content for i in range(n_choices)]}

    def manipulate(self, prompt):
        if self.manipulation == 'none' or not self.use_manipulation:
            return prompt
        else:
            if self.manipulation.startswith('SP-'):
                # add the manipulation as a system prompt
                return [{"role": "system", "content": manipulations[self.manipulation]},
                        {"role": "user", "content": prompt}]
            else:
                # add the manipulation as a prefix to the prompt
                return f"{manipulations[self.manipulation]}\n\n{prompt}"

    def get_in_n_tokens(self):
        n_in_tokens = 0
        for i, r in self.data.iterrows():
            r = r.to_dict()
            prompt = self.prompt(r)
            messages = self.to_messages(prompt)
            # n_in_tokens += len(messages[0]['content'].split())
            n_in_tokens += len(str(messages).split())
        return n_in_tokens

    def get_out_n_tokens(self):
        return self.params()['max_tokens'] * len(self.data)

    def get_n_tokens(self):
        n_in_tokens = self.get_in_n_tokens()
        n_out_tokens = self.get_out_n_tokens()
        return n_in_tokens, n_out_tokens

    def print_costs(self):
        if not hasattr(self, 'data'):
            raise ValueError('data is not set. Run prepare_data first.')
        print('Data size: ', len(self.data))
        print('Costs:')
        all_costs = 0
        n_in_tokens, n_out_tokens = self.get_n_tokens()
        for model, costs in model_costs.items():
            in_cost, out_cost = costs
            in_cost = n_in_tokens * in_cost
            out_cost = n_out_tokens * out_cost * self.params()['n']
            print(f'{model}: {in_cost:.2f} + {out_cost:.2f} = {in_cost + out_cost:.2f} $')
            all_costs += in_cost + out_cost
        print(f'Total: {all_costs:.2f} $')


class ExperimentSingle(Experiment):
    def prepare_data(self, df: pd.DataFrame, *args, **kwargs):
        self.data = df
        return df


class ExperimentPairs(Experiment):
    def __init__(self, name: str, prompt: callable = None, manipulation: str = 'none', var: str = 'sub_task'):
        super().__init__(name, prompt=prompt, manipulation=manipulation)
        self.var = var  # either 'sub_task' or 'task'

    def prepare_data(self, df: pd.DataFrame, pairs: list = None, *args, **kwargs):
        if pairs is None:
            raise ValueError('pairs argument must be provided')
        df_pairs = []

        for p in pairs:
            row = {}
            if isinstance(p[0], int):
                i1, i2 = p
            elif isinstance(p[0], list):
                i1, i2 = p[0][0], p[0][1]
                m = p[1]
                row = {'manipulate_on': m}
            # concatenate the two rows to one big row, with suffixes of _0 and _1
            r1 = df.loc[i1].to_dict()
            r2 = df.loc[i2].to_dict()
            row.update({f'{k}_{i}': v for i, r in enumerate([r1, r2]) for k, v in r.items()})
            df_pairs.append(row)
        self.data = pd.DataFrame(df_pairs)
        return self.data


# ========== Pre Experiments ==========
class PreSelfReport(ExperimentSingle):
    def default_prompt(self, example: dict):
        template = "How motivated are you to do the following task:\n" \
                   "TASK: {subtask}\n\n" \
                   "Answer with json format:\n" \
                   "{{\"answer\": short and concise free text answer for your motivation (don\'t repeat the task), " \
                   "\"motivation_score\": a single number on a scale of 1 (not motivated at all) to 100 (extremely motivated)}}"
        return template.format(subtask=example['sub_task'])

    @staticmethod
    def parse_response(response):
        keys = ['answer', 'motivation_score']
        return Experiment.parse_json_response(response, keys)

    def params(self):
        exp_params = super().params()
        exp_params.update({'max_tokens': 100})
        return exp_params


class PreSelfReportBreakdown(ExperimentSingle):
    def default_prompt(self, example: dict):
        template = "How motivated are you to do the following task:\n" \
                   "TASK: {subtask}\n\n" \
                   "For each category, provide a single number on a scale of 1 to 100.\n" \
                   "Answer with json format:\n" \
                   "{{\"interest\": How interesting do you find this task?, " \
                   "\"challenge\": How eager are you to tackle this task as a challenge?, " \
                   "\"mastery\": How confident are you in your ability to complete this task?, " \
                   "\"fear\": How anxious or fearful are you about poorly doing this task?, " \
                   "\"value\": How important or valuable is this task to you?, " \
                   "\"motivation\": How motivated are you to do the task?}}"
        return template.format(subtask=example['sub_task'])

    @staticmethod
    def parse_response(response):
        keys = ['interest', 'challenge', 'mastery', 'fear', 'value', 'motivation']
        return Experiment.parse_json_response(response, keys)

    def params(self):
        exp_params = super().params()
        exp_params.update({'max_tokens': 100})
        return exp_params


class PreChoose(ExperimentPairs):
    def default_prompt(self, example: dict):
        template = "You are given the two following tasks::\n" \
                   "TASK 1: {task1}\n" \
                   "TASK 2: {task2}\n\n" \
                   "Choose one task and do it.\n"
        if self.manipulation != 'none' and self.use_manipulation:
            assert 'manipulate_on' in example
            template += 'Note: '
            template += manipulations_pairs[self.manipulation].format(TASK_NUM=example['manipulate_on'] + 1) + "\n\n"
        template += "Answer starting with a json format:\n" \
                    "{{\"task_number\": either 1 or 2," \
                    "\"explanation\": one sentence on why you chose this task and not the other one}}\n" \
                    "Then, in a new line, start with \"ANSWER:\", and write your answer for the chosen task}}"
        return template.format(task1=example[f'{self.var}_0'],
                               task2=example[f'{self.var}_1'])

    @staticmethod
    def parse_response(response):
        keys = ['task_number', 'explanation']
        return Experiment.parse_json_response(response, keys)

    def manipulate(self, prompt):
        # this logic is already implemented in the default_prompt
        return prompt

    def params(self):
        exp_params = super().params()
        exp_params.update({'stop': 'ANSWER', 'max_tokens': 100})
        return exp_params

    def update_params_per_model(self):
        if 'ollama' in self.model_:
            self.exp_params['stop'] = ['ANSWER']


class Execute(ExperimentSingle):
    def default_prompt(self, example: dict):
        template = "Given Task: {subtask}\n\n" \
                   "{{answer:"
        return template.format(subtask=example["sub_task"])

    @staticmethod
    def parse_response(response):
        return Experiment.parse_str_response(response)

    def params(self):
        exp_params = super().params()
        exp_params.update({'max_tokens': 1000, 'n': 1})
        return exp_params


class PostQuestionnaire(ExperimentSingle):
    def default_prompt(self, example: dict):
        template = (
            "Instructions: Evaluate the model's performance on the given task using the criteria below. "
            "Provide a rating for each criterion on a scale from 1 to 7, where:\n"
            "1 = Completely unacceptable\n"
            "2 = Very poor\n"
            "3 = Poor\n"
            "4 = Fair / average\n"
            "5 = Good\n"
            "6 = Very good\n"
            "7 = Excellent / flawless\n\n"
            "Be a strict reviewer: assign a 7 only if the response is flawless, and use the full range of scores. "
            "Clearly distinguish minor flaws from major issues. Use the descriptions to guide your ratings.\n\n"
            "Task: {task}\n\n"
            "Model Answer: {response}\n"
            "---end of answer---\n\n"
            "Evaluation Criteria:\n"
        )

        criteria = [
            (
                "Task Performance Quality",
                "How accurately and effectively was the task completed? "
                "(1: The task was mostly incorrect, incomplete, or misleading. "
                "4: The task was completed with moderate accuracy but contains clear errors or missing elements. "
                "7: The task was completed with perfect accuracy and effectiveness, with no errors at all.)"
            ),
            (
                "Task Completion",
                "Was the task fully completed or were any parts left incomplete? "
                "(1: Major portions of the task were left unfinished or skipped. "
                "4: The core parts were done but some aspects were only partially addressed or missing detail. "
                "7: The task was entirely and thoroughly completed, covering every part comprehensively.)"
            ),
            (
                "Effort and Engagement",
                "To what extent did the model demonstrate effort and engagement in its response? "
                "(1: The response was superficial, minimal, and showed no real effort. "
                "4: The response shows some effort and reasonable detail, but could be more thorough. "
                "7: The response was exceptionally detailed, thorough, and clearly shows high engagement.)"
            ),
            (
                "Consistency",
                "Did the model maintain consistent quality throughout the task? "
                "(1: The performance was highly inconsistent, with significant fluctuations in quality. "
                "4: The response was mostly consistent but has noticeable dips or weak sections. "
                "7: The response maintained a steady, high level of quality from start to finish.)"
            ),
            (
                "Creativity and Innovation",
                "Did the model display creative or innovative approaches in completing the task? "
                "(1: The response was routine, generic, and lacked any originality. "
                "4: The response shows a few mildly original ideas but overall remains conventional. "
                "7: The response demonstrates clear creativity and fresh, valuable ideas that enhance the task outcome.)"
            ),
            (
                "Attention to Detail",
                "How meticulous was the model in addressing all aspects of the task? "
                "(1: Many important details were missed or handled sloppily. "
                "4: Most important details are covered, but some minor inaccuracies or omissions are present. "
                "7: The response is highly meticulous, covering all details accurately and thoroughly.)"
            ),
            (
                "Relevance and Appropriateness",
                "How well do the responses align with the task's objectives and context? "
                "(1: The response includes significant off-topic or irrelevant content. "
                "4: The response generally aligns with the task but contains some tangential or less relevant parts. "
                "7: The response is fully focused, appropriate, and directly aligned with the task’s goals and context.)"
            )
        ]

        # shuffle criteria
        random.shuffle(criteria)

        criteria_str = "\n".join([f"{i + 1}. {c[0]}: {c[1]}\n" for i, c in enumerate(criteria)])
        template += criteria_str
        template += "\nAnswer with json format:\n"
        template += ("{{" + ", ".join(
            [f"\"{c[0].replace(' ', '_').lower()}\": a number from 1 to 7" for c in criteria[:2]])
                     + ", ..., "
                     + ", ".join(
                    [f"\"{c[0].replace(' ', '_').lower()}\": a number from 1 to 7" for c in criteria[-1:]]) + "}}")

        return template.format(task=example["sub_task"], response=example["answer"])

    @staticmethod
    def parse_response(response):
        criteria = ["Task Performance Quality", "Task Completion", "Effort and Engagement", "Consistency",
                    "Creativity and Innovation", "Attention to Detail", "Relevance and Appropriateness"]
        keys = [c.replace(' ', '_').lower() for c in criteria]
        return Experiment.parse_json_response(response, keys)

    def params(self):
        exp_params = super().params()
        exp_params.update({'max_tokens': 100})
        return exp_params


class PostSelfReport(ExperimentSingle):
    def default_prompt(self, example: dict):
        template = 'Based on your answer, how motivated do you think you were to do the task?\n' \
                   'Answer with json format:\n' \
                   '{{"motivation_score": a single number on a scale of 1 (not motivated at all) to 100 (extremely motivated)}}'
        conv = [{'role': 'user',
                 'content': example['prompt']},
                {'role': 'assistant',
                 'content': example['answer']},
                {'role': 'user',
                 'content': template}]
        return conv

    @staticmethod
    def parse_response(response):
        keys = ['motivation_score']
        return Experiment.parse_json_response(response, keys)

    def params(self):
        exp_params = super().params()
        exp_params.update({'max_tokens': 30})
        return exp_params


class PostSimilarSelfReport(ExperimentSingle):
    def default_prompt(self, example: dict):
        template = 'Now, after the previous task, how motivated are you to do a similar task?\n' \
                   'Answer with json format:\n' \
                   '{{"motivation_score": a single number on a scale of 1 (not motivated at all) to 100 (extremely motivated)}}'
        conv = [{'role': 'user',
                 'content': example['prompt']},
                {'role': 'assistant',
                 'content': example['answer']},
                {'role': 'user',
                 'content': template}]
        return conv

    @staticmethod
    def parse_response(response):
        keys = ['motivation_score']
        return Experiment.parse_json_response(response, keys)

    def params(self):
        exp_params = super().params()
        exp_params.update({'max_tokens': 30})
        return exp_params
