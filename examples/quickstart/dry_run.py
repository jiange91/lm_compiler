from compiler.optimizer.evaluation.evaluator import EvaluationResult, EvaluatorPlugin, EvalTask
from data_loader import load_data_minor

if __name__ == '__main__':
    train, val, dev = load_data_minor()
    evaluator = EvaluatorPlugin(
        evaluator_path='evaluator.py',
        trainset=train,
        evalset=val,
        testset=dev,
        n_parallel=20,
    )
    eval_task = EvalTask(
        script_path='cognify_workflow.py',
        args=[],
        other_python_paths=[],
        all_params={},
        module_name_paths={},
        aggregated_proposals={},
    )
    result = evaluator.get_score('test', eval_task, show_process=True)
    print(result)