from cognify.optimizer.registry import register_data_loader
import json

@register_data_loader
def load_data_minor():
    with open("data._json", "r") as f:
        data = json.load(f)
    # raw format:
    # [
    #     {
    #         "question": "...",
    #         "docs": [...],
    #         "answer": "..."
    #     },
    #     ...
    # ]
          
    # format to (input, output) pairs
    new_data = []
    for d in data:
        input = (d["question"], d["docs"])
        output = d["answer"]
        new_data.append((input, output))
    return new_data[:5], None, new_data[5:]
