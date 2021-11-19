import os
import requests
import json

def send_message_to_slack(url, acc, loss, training_time, model_path): 
    payload = {
        "blocks": [
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": "학습이 완료되었습니다."
                }
            },
            {
                "type": "section",
                "fields": [
                    {
                        "type": "mrkdwn",
                        "text": f"*Accuracy:*\n{acc}"
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Training Time:*\n{training_time}"
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Loss:*\n{loss}"
                    },{
                        "type": "mrkdwn",
                        "text": f"*gsutil URI:*\n{model_path}"
                    }
                ]
            }
        ]
    }
    requests.post(url, json=payload)

def request_deploy_api(model_path):
    owner = os.getenv("GITHUB_OWNER")
    repo = os.getenv("GITHUB_REPO")
    workflow_id = os.getenv("GITHUB_WORKFLOW")
    access_token = os.getenv("GITHUB_TOKEN")
    model_tag = os.getenv("MODEL_TAG")

    headers = {'Authorization' : 'token ' + access_token }
    data = {"ref": "main", "inputs":{"model_path": model_path, "model_tag": model_tag }}
    response = requests.post(f"https://api.github.com/repos/{owner}/{repo}/actions/workflows/{workflow_id}/dispatches", headers=headers, data=json.dumps(data))
    print(response.text)


