from compiler.optimizer.params.fewshot import LMFewShot
from compiler.optimizer.params.scaffolding import LMScaffolding
from compiler.optimizer.params import reasoning, model_selection, common
from compiler.optimizer.evaluation.evaluator import EvaluationResult, EvaluatorPlugin, EvalTask
from compiler.optimizer.analysis.param_sensitivity import SensitivityAnalyzer
from compiler.langchain_bridge.interface import LangChainLM
from compiler.optimizer.params import ensemble
import runpy
import uuid
import multiprocess as mp
import json
import os
import random
import optuna

from compiler.IR.llm import LMConfig
from compiler.optimizer.params.common import IdentityOption
from compiler.optimizer.params.reasoning import ZeroShotCoT, PlanBefore
from compiler.optimizer.plugin import OptimizerSchema
from compiler.optimizer.core import driver, flow
import dspy
from dspy.datasets.hotpotqa import HotPotQA

def load_data_minor():
    trainset = [
        ("""Are Walt Disney and Sacro GRA both documentry films?""", """yes"""),
        ("""What do students do at the school of New York University where Meleko Mokgosi is an artist and assistant professor?""", """design their own interdisciplinary program"""),
        ("""Which is published more frequently, The People's Friend or Bust?""", """The People's Friend"""),
        ("""How much is spent on the type of whiskey that 1792 Whiskey is in the United States?""", """about $2.7 billion"""),
        ("""The place where John Laub is an American criminologist and Distinguished University Professor in the Department of Criminology and Criminal Justice at was founded in what year?""", """1856"""),
        ("""What year did the mountain known in Italian as "Monte Vesuvio", erupt?""", """79 AD"""),
        ("""What was the full name of the author that memorialized Susan Bertie through her single volume of poems?""", """Emilia Lanier"""),
        ("""How many seasons did, the Guard with a FG%% around .420, play in the NBA ?""", """14 seasons"""),
        ("""Estonian Philharmonic Chamber Choir won the grammy Award for Best Choral Performance for two songs by a composer born in what year ?""", """1935"""),
        ("""Which of the sport analyst of The Experts Network is nicknamed  "The Iron Man"?""", """Calvin Edwin Ripken Jr."""),
        ("""What are both National Bird and America's Heart and Soul?""", """What are both National Bird and America's Heart and Soul?"""),
        ("""What was the 2010 population of the birthplace of Gerard Piel?""", """17,121"""),
        ("""On what streets is the hospital that cared for Molly Meldrum located?""", """the corner of Commercial and Punt Roads"""),
    ]
    return trainset[:3], trainset[3:5], trainset[0:1]

def load_data():
    dataset = HotPotQA(train_seed=1, train_size=150, eval_seed=2023, dev_size=200, test_size=0)
    def get_input_label(x):
        return x.question, x.answer
    trainset = [get_input_label(x) for x in dataset.train[0:100]]
    valset = [get_input_label(x) for x in dataset.train[100:150]]
    devset = [get_input_label(x) for x in dataset.dev]
    print(len(trainset), len(valset), len(devset))
    return trainset, valset, devset

def opt(train, val, dev):
    evaluator = EvaluatorPlugin(
        trainset=train,
        evalset=val,
        # evalset=None,
        testset=dev,
        n_parallel=50,
    )
    # ================= LM Selection =================
    lm_options = [
        # LMConfig(
        #     provider='fireworks',
        #     cost_indicator=0.3,
        #     kwargs= {
        #         'model': 'accounts/fireworks/models/llama-v3p2-3b-instruct',
        #         # 'temperature': 0.0,
        #     }
        # ),
        # LMConfig(
        #     provider='fireworks',
        #     model="accounts/zih015-63d1a0/deployedModels/llama-v3p1-8b-instruct-46c7347d",
        #     cost_indicator=0.6,
        #     kwargs= {
        #         'temperature': 0.0,
        #     }
        # ),
        # LMConfig(
        #     provider='local',
        #     model='llama-3.1-8b',
        #     cost_indicator=0.6,
        #     kwargs={
        #         'temperature': 0.0,
        #         'openai_api_base': 'http://192.168.1.16:30000/v1',
        #     }
        # ),
        LMConfig(
            provider='openai',
            model='gpt-4o-mini',
            cost_indicator=1.0,
            kwargs= {
                'temperature': 0.0,
            }
        )
    ]
    model_param = model_selection.LMSelection(
        'lm_model', model_selection.model_option_factory(lm_options)
    )
    
    # ================= Down Sample =================
    plain_task = EvalTask(
        script_path='/mnt/ssd4/lm_compiler/examples/HotPotQA/cognify_anno.py',
        args=[],
        other_python_paths=[],
        all_params={},
        module_name_paths={},
        aggregated_proposals={},
    )
    evaluator.down_sample(
        sample_size=50,
        mode='train',
        task=plain_task, 
        sample_mode='difficulty',
        log_dir='/mnt/ssd4/lm_compiler/examples/HotPotQA/down_sample_logs',
    )
    evaluator.down_sample(
        sample_size=25,
        mode='eval',
        task=plain_task, 
        sample_mode='difficulty',
        log_dir='/mnt/ssd4/lm_compiler/examples/HotPotQA/down_sample_logs',
    )
    # ================= Sensitivity Analysis =================
    model_sensitivity = SensitivityAnalyzer(
        target_param_type=model_selection.LMSelection,
        eval_task=plain_task,
        evaluator=evaluator,
        n_parallel=4,
        log_dir='/mnt/ssd4/lm_compiler/examples/HotPotQA/sensitivity_logs',
        try_options=model_param,
        module_type=LangChainLM,
    )
    sensitivity_result = model_sensitivity.run()
    print(sensitivity_result)
    
    # ================= Reasoning Options =================
    reasoning_param = reasoning.LMReasoning(
        "reasoning", [IdentityOption(), ZeroShotCoT()] 
    )
    # ================= Few Shot Options =================
    few_shot_params = LMFewShot("few_shot", 4)
    # ================= Ensemble Options =================
    general_usc_ensemble = ensemble.UniversalSelfConsistency(3, temperature=0.7)
    general_ensemble_params = ensemble.ModuleEnsemble(
        "ensemble", [IdentityOption(), general_usc_ensemble]
    )
    
    refine_usc_ensemble = ensemble.UniversalSelfConsistency(3, temperature=0.7)
    refine_ensemble_params = ensemble.ModuleEnsemble(
        "ensemble", [refine_usc_ensemble]
    )
    refine_ensemble_params.module_name = 'refine_query'
    
    gen_answer_usc_ensemble = ensemble.UniversalSelfConsistency(3, temperature=0.7)
    gen_answer_ensemble_params = ensemble.ModuleEnsemble(
        "ensemble", [gen_answer_usc_ensemble]
    )
    gen_answer_ensemble_params.module_name = 'generate_answer'
    
    # ================= Inner Loop Config =================
    inner_opt_config = flow.OptConfig(
        n_trials=0,
        throughput=2,
        log_dir="/mnt/ssd4/lm_compiler/examples/HotPotQA/with_50_25_no_outer_fix_prompt_no_frugal/opt_logs.json",
        evolve_interval=4,
        frugal_eval_cost=True,
    )
    inner_loop_config = driver.LayerConfig(
        layer_name='inner_loop',
        universal_params=[few_shot_params, reasoning_param, model_param],
        opt_config=inner_opt_config,
        save_ckpt_interval=1,
    )
    
    outer_opt_config = flow.OptConfig(
        n_trials=0,
        throughput=4,
        log_dir='/mnt/ssd4/lm_compiler/examples/HotPotQA/with_50_25_full_opt',
        frugal_eval_cost=False,
    )
    
    outer_loop_config = driver.LayerConfig(
        layer_name='outer_loop',
        universal_params=[general_ensemble_params], # will overwrite module name
        # dedicate_params=[refine_ensemble_params, gen_answer_ensemble_params],
        opt_config=outer_opt_config,
        save_ckpt_interval=1,
        use_SH_allocation=True,
    )
    
    opt_driver = driver.MultiLayerOptimizationDriver(
        layer_configs=[inner_loop_config],
        # layer_configs=[outer_loop_config, inner_loop_config],
        quality_constraint=0.52,
    )
    cost, pareto_frontier, opt_logs = opt_driver.run(
        evaluator=evaluator,
        script_path='/mnt/ssd4/lm_compiler/examples/HotPotQA/cognify_anno.py',
    )
    return opt_driver

def eval(opt_driver: driver.MultiLayerOptimizationDriver):
    eval_result = opt_driver.evaluate(
        bot_trial_log_id='0801a67cbc474b93aaef22b8ca9b1587',
        opt_log_path='/mnt/ssd4/lm_compiler/examples/HotPotQA/with_50_25_no_outer_fix_prompt_no_frugal/opt_logs.json',
    )
    print(eval_result)

def raw_test(data):
    evaluator = EvaluatorPlugin(
        trainset=None,
        evalset=None,
        testset=data,
        n_parallel=100,
    )
    eval_task = EvalTask(
        script_path='/mnt/ssd4/lm_compiler/examples/HotPotQA/cognify_anno.py',
        args=[],
        other_python_paths=[],
        all_params={},
        module_name_paths={},
        aggregated_proposals={},
    )
    print(evaluator.get_score('test', eval_task, show_process=True))

    
if __name__ == '__main__':
    # mp.set_start_method('spawn')
    mp.context._force_start_method('spawn')
    
    train, val, dev = load_data()
    # train, val, dev = load_data_minor()
    opt_driver = opt(train, val, dev)
    eval(opt_driver)
    # raw_test(dev)