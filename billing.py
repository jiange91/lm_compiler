import json

with open('token_usage.json', 'r') as f:
    data = json.load(f)

def model_2_price_pM(model: str, prompt, completion):
    if 'gpt-4o-mini' in model:
        return (0.15 * prompt +  0.6 * completion) / 1e6
    elif 'gpt-4o' in model:
        return (5 * prompt + 15 * completion) / 1e6
    
total_use = 0
gpt_usage = {}
for step, usage in data.items():
    if step == 'total':
        total_price = 0
        for model, usage in usage.items():
            total_price += model_2_price_pM(model, usage['prompt_tokens'], usage['completion_tokens'])
    else:
        for model, usage in usage.items():
            if model not in gpt_usage:
                gpt_usage[model] = [0, 0]
            gpt_usage[model][0] += usage['prompt_tokens']
            gpt_usage[model][1] += usage['completion_tokens']

print(f"Total price: {total_price}")

accurate_price = 0
for model, usage in gpt_usage.items():
    price = model_2_price_pM(model, usage[0], usage[1])
    accurate_price += price
print(f"Accurate price: {accurate_price}")
print(f"{total_price / accurate_price} x cheaper")