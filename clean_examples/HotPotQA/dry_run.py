from compiler.optimizer.evaluation.evaluator import EvaluationResult, EvaluatorPlugin, EvalTask
from data_loader import load_data_minor

if __name__ == '__main__':
    train, val, dev = load_data_minor()
    evaluator = EvaluatorPlugin(
        evaluator_path='/mnt/ssd4/lm_compiler/clean_examples/HotPotQA/evaluator.py',
        trainset=train,
        evalset=val,
        testset=dev,
        n_parallel=50,
    )
    eval_task = EvalTask(
        script_path='/mnt/ssd4/lm_compiler/clean_examples/HotPotQA/cognify_anno.py',
        args=[],
        other_python_paths=[],
        all_params={},
        module_name_paths={},
        aggregated_proposals={},
    )
    result = evaluator.get_score('test', eval_task, show_process=True)
    print(result)