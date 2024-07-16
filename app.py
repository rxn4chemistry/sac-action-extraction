from flask import Flask, jsonify, request
from sac_action_extraction.extract_actions import extract_actions_from_string
import subprocess
import json
from pathlib import Path

app = Flask(__name__)


def jsonify(obj):
    if isinstance(obj, list):
        return [jsonify(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: jsonify(value) for key, value in obj.items()}
    elif hasattr(obj, '__dict__'):
        return {
                'type': obj.__class__.__name__,
                **{k: jsonify(v) for k, v in obj.__dict__.items() if not k.startswith('_')}
            }
    else:
        return str(obj)


@app.route('/api/v1/actions', methods=['POST'])
def get_actions():
    data = request.get_json()
    inputs = data.get('inputs', [])

    sol = extract_actions_from_string(
        inputs,
        models = ["./models/sac.pt"],
        sentencepiece_model = "./models/sp_model.model"
    )['actions']

    actions = [jsonify(ac) for ac in sol]
    return actions

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
