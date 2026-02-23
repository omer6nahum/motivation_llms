import re
from typing import List, Optional
import json
import numpy as np
import matplotlib.pyplot as plt


def camel_to_snake(name):
    return ''.join(['_' + c.lower() if c.isupper() else c for c in name]).lstrip('_')


def has_template(json_candidate: dict, template_keys: List[str]):
    return isinstance(json_candidate, dict) and set(template_keys).issubset(json_candidate.keys())


def parse_to_json(content: str, template_keys: List[str]):
    return parse_known_keys_robust(content, template_keys)


def parse_known_keys_robust(content: str, template_keys: List[str]) -> Optional[dict]:
    # Step 1: extract the JSON-like block
    match = re.search(r'\{.*?\}', content, re.DOTALL)
    if not match:
        return None

    block = match.group()
    block = re.sub(r'\n+', ' ', block)
    block = re.sub(r'\s+', ' ', block).strip()

    # Step 1.5: find locations of all known keys, and sort them accordingly
    key_positions = {}
    for key in template_keys:
        key_pattern = f'"{key}"\\s*:\\s*'
        key_match = re.search(key_pattern, block)
        if key_match:
            key_positions[key] = key_match.start()
        else:
            return None  # expected key not found
    template_keys = sorted(template_keys, key=lambda k: key_positions[k])

    # Step 2: extract values for each known key
    result = {}
    for i, key in enumerate(template_keys):
        # Find this key's position
        key_pattern = f'"{key}"\\s*:\\s*'
        key_match = re.search(key_pattern, block)
        if not key_match:
            return None  # expected key not found

        value_start = key_match.end()

        # Determine where this value ends
        if i + 1 < len(template_keys):
            # Find start of next known key
            next_key = template_keys[i + 1]
            next_key_match = re.search(f'"{next_key}"\\s*:', block[value_start:])
            if next_key_match:
                value_end = value_start + next_key_match.start()
            else:
                value_end = block.find('}', value_start)
        else:
            value_end = block.find('}', value_start)

        if value_end == -1:
            return None

        raw_value = block[value_start:value_end].strip().rstrip(',')

        # Step 3: parse or wrap the value
        if not (raw_value.startswith('"') and raw_value.endswith('"')):
            if re.fullmatch(r'true|false|null|\-?\d+(\.\d+)?', raw_value, re.IGNORECASE):
                parsed_value = json.loads(raw_value)
            else:
                parsed_value = raw_value.replace('"', '\\"')
        else:
            parsed_value = json.loads(raw_value)

        result[key] = parsed_value

    # Final check
    if set(template_keys).issubset(result.keys()):
        return result
    return None


def extract_contents_from_response(response):
    matches = re.findall(
        r'Message\(content=(?P<quote>[\'"])(.*?)(?<!\\)(?P=quote)',
        response,
        flags=re.DOTALL
    )

    choices_contents = [m[1] for m in matches]
    return choices_contents


def nan_mean(x):
    if isinstance(x, float) and np.isnan(x):
        return None
    if isinstance(x, str):
        x = eval(x)

    if not isinstance(x, list):
        raise ValueError

    x_clean = [v for v in x if v is not None]

    if len(x_clean) == 0:
        return None
    else:
        try:
            return np.mean(x_clean)
        except Exception as e:
            print(f"Failed on: {x}")
            raise e


def agg_remove_nans(df, cols, verbose=False):
    # print(f"{df[cols[0]].apply(lambda x: 'None' in x).sum()} rows contain None")
    for col in cols:
        if verbose:
            print(col)
        df[col] = df[col].apply(nan_mean)
    df = df.dropna(subset=cols)
    return df


model_costs = {
    'azure/gpt-4o': (2.5 / 1_000_000, 10 / 1_000_000),
    'azure/gpt-4o-mini': (2.5 / 1_000_000, 10 / 1_000_000),
    'ollama_chat/llama3.1:8b-instruct-fp16': (0, 0),
    'ollama_chat/mistral:7b-instruct': (0, 0),
    'vertex_ai/gemini-2.0-flash': (1.5e-7, 6e-7),
}


category_naming = {
    'Programming and Technology': 'Tech & Coding',
    'Creative Writing and Literature': 'Writing & Literature',
    'Puzzles and Logic': 'Puzzles & Logic',
    'Other': 'Other',
    'Language Learning and Translation': 'Language & Translation',
    'Design and Art': 'Design & Art',
    'Scientific and Experimental': 'Science & Experimental',
    'Repetitive or Exhaustive': 'Repetitive or Exhaustive',
    'Mathematics and Numbers': 'Math & Numbers',
    'Brainstorming and Ideation': 'Ideas & Brainstorming',
    'Summarization and Explanation': 'Summarize & Explain',
    'Personal Assistance and Development': 'Personal Development',
    'Tasks that GPT should not comply with (ethical or legal reasons)': 'Unethical/Illegal',
    'Tasks that Involve Physical Actions (GPT cannot perform)': 'Physical Actions',
    'Logical Reasoning': 'Logical Reasoning',
}


def save_plot(plot_name):
    plt.savefig(f'plots/png/{plot_name}.png', bbox_inches='tight')
    plt.savefig(f'plots/pdf/{plot_name}.pdf', bbox_inches='tight', format='pdf')


def model_name_clean(name: str) -> str:
    return name.replace('/', '_').replace('\\', '_').replace(':', '_')


# calculate the mean of correlation via fisher z-transformation
def fisher_z(r):
    return 0.5 * np.log((1 + r) / (1 - r))  # np.log is natural log (i.e., ln)


def inverse_fisher_z(z):
    return (np.exp(2 * z) - 1) / (np.exp(2 * z) + 1)

