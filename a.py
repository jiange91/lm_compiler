import time
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool
from tqdm import tqdm
from threading import Lock
import heapq
import random

outer_loop_trials = 4
outer_parallelism = 4

inner_loop_trials = 10
inner_parallelism = 3

eval_data_size = 100
evaluator_parallelism = 20

max_position = 20
position_pool = list(range(20))
heapq.heapify(position_pool)
_pbar_lock = Lock()

def ask_for_position():
    global max_position
    with _pbar_lock:
        # if no position is available, add a new one
        if len(position_pool) == 0:
            position = max_position
            max_position += 1
        else:
            position = heapq.heappop(position_pool)
        return position

def release_position(position):
    with _pbar_lock:
        heapq.heappush(position_pool, position)
            

def evaluator_task(i):
    time.sleep(0.5)
    return i + random.random()

# Define the function for inner-loop tasks
def inner_task(task_id, routine_id):
    # Simulate work being done
    position = ask_for_position()
    
    with tqdm(
        total=eval_data_size,
        desc=f"Evaluation Task at Inner {task_id} Outer {routine_id} (avg score: 0.00)",
        position=position,
        leave=False
    ) as pbar:
        with Pool(evaluator_parallelism) as p:
            # Wait for all evaluation tasks to complete
            scores = []
            for score in p.imap_unordered(evaluator_task, range(eval_data_size)):
                pbar.update(1)
                # print current average score
                scores.append(score)
                avg_score = sum(scores) / len(scores)
                pbar.set_description(f"Evaluation Task at Inner {task_id} Outer {routine_id} (avg score: {avg_score:.2f})")
                
    release_position(position)
    return avg_score

# Define the function for outer-loop routines
def outer_routine(routine_id):
    inner_position = ask_for_position()
    with tqdm(
        total=inner_loop_trials,
        desc=f"Inner Routine at Outer trial {routine_id} (best avg score: 0.00)",
        position=inner_position,
        leave=False
    ) as inner_bar:
        with ThreadPoolExecutor(max_workers=inner_parallelism) as inner_executor:
            # Submit inner tasks
            futures = [
                inner_executor.submit(inner_task, task_id=i, routine_id=routine_id)
                for i in range(inner_loop_trials)
            ]
            # Wait for all inner tasks to complete
            best_score = 0.0
            for future in futures:
                avg_score = future.result()
                inner_bar.update(1)
                if avg_score > best_score:
                    best_score = avg_score
                    inner_bar.set_description(f"Inner Routine at Outer trial {routine_id} (best avg score: {best_score:.2f})")
    release_position(inner_position)
    return best_score

# Main loop for the outer routine
def multi_level_optimization():
    # Main progress bar for the outer loop
    position = ask_for_position()
    with tqdm(
        total=outer_loop_trials, 
        desc="Outer Loop (best avg score: 0.00)", 
        position=position,
        leave=True
    ) as outer_bar:
        with ThreadPoolExecutor(max_workers=outer_parallelism) as outer_executor:
            futures = [
                outer_executor.submit(outer_routine, routine_id=i)
                for i in range(outer_loop_trials)
            ]
            # Update the outer progress bar as each routine completes
            best_score = 0.0
            for future in futures:
                feedback = future.result()
                outer_bar.update(1)
                if feedback > best_score:
                    best_score = feedback
                    outer_bar.set_description(f"Outer Loop (best avg score: {best_score:.2f})")
    release_position(position)
    return best_score

# Run the multi-level optimization with nested progress bars
multi_level_optimization()
