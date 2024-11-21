import os
import json

def read_all_json_files():
    """
    Reads all .json files in the current directory and returns their contents.
    
    Returns:
        dict: A dictionary where keys are filenames and values are file contents as parsed JSON.
    """
    json_data = []
    for filename in os.listdir('.'):
        if filename.endswith('.json'):
            try:
                with open(filename, 'r', encoding='utf-8') as file:
                    json_data.append(json.load(file))
                    if isinstance(json_data[-1], list):
                        print(filename)
            except (json.JSONDecodeError, IOError) as e:
                print(f"Error reading {filename}: {e}")
    return json_data

# Example usage
if __name__ == "__main__":
    all_json_contents = read_all_json_files()
    print(f'Found {len(all_json_contents)}')
    raw_data = [{
        'problem': item['problem'],
        'solution': item['solution']
    } for item in all_json_contents]
    with open('data.json', 'w') as file:
        json.dump(raw_data, file, indent=4)