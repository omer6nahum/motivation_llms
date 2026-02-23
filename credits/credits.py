import os
import json


def set_env_vars(creds_file):
    with open(creds_file) as f:
        creds = json.load(f)
    for k, v in creds.items():
        os.environ[k] = v
    print(f"Loaded credentials from {creds_file.split('/')[1].split('_')[0]}")


def load_credits():
    credits_json_path = 'credits/<your_credits_json>'  # TODO: put the path to your own credits json file here
    set_env_vars(credits_json_path)
