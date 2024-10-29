import argparse
import sys
import multiprocessing as mp
import os

from compiler.optimizer.plugin import OptimizerSchema

def _worker(script, args, q: mp.Queue):
    script_dir = os.path.dirname(script)
    # Add the script's directory and the root directory (if needed) to sys.path
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
        
    sys.argv = [script] + args
    schema = OptimizerSchema.capture(script)
    print(schema.opt_target_modules)
    pass
    # result = schema.program()
    # q.put(result)
    

def main():
    parser = argparse.ArgumentParser(description="Optimizer args")
    parser.add_argument("--opt-arg", help="test")
    
    # Use `parse_known_args` to stop at `--` and keep script arguments separate
    args, remaining = parser.parse_known_args()

    if "--" in remaining:
        # Split the remaining arguments into script and script-args
        split_index = remaining.index("--")
        script = remaining[split_index + 1]  # Script name
        script_args = remaining[split_index + 2:]  # Arguments for the script
    else:
        print("Error: No script provided.")
        sys.exit(1)

    print("Module arguments:", args)
    print("Script: ", script, ", arguments:", script_args)
    
    try:
        q = mp.Queue()
        proces = mp.Process(target=_worker, args=(script, script_args, q))
        proces.start()
        proces.join()
        result = q.get()
        print(f"Result: {result}")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
