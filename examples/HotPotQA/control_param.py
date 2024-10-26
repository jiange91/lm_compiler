quality_constraint: float = .4
train_down_sample: int = 0
val_down_sample: int = 0
save_to_dir: str = '/mnt/ssd4/lm_compiler/examples/HotPotQA/test_cmd_outer_debug/'

from compiler.IR.llm import LMConfig
from compiler.optimizer.core import driver, flow

from compiler.optimizer.params.utils import build_param, load_params, dump_params
from compiler.optimizer.params import reasoning, model_selection
from compiler.optimizer.params import ensemble
from compiler.optimizer.params.common import IdentityOption
from compiler.optimizer.params.fewshot import LMFewShot
from compiler.optimizer.params.reasoning import ZeroShotCoT

# ================= Model Options =================
lm_options = [
    LMConfig(
        provider='openai',
        model='gpt-4o-mini',
        cost_indicator=1.0,
        kwargs= {
            'temperature': 0.0,
        }
    ),
    LMConfig(
        provider='openai',
        model='gpt-4o',
        cost_indicator=10.0,
        kwargs= {
            'temperature': 0.0,
        }
    )
]
model_param = model_selection.LMSelection(
    'lm_model', model_selection.model_option_factory(lm_options)
)

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

# ================= Inner Loop Config =================
inner_opt_config = flow.OptConfig(
    n_trials=2,
    throughput=2,
    log_dir=None,
    evolve_interval=1,
    frugal_eval_cost=True,
)
inner_loop_config = driver.LayerConfig(
    layer_name='inner_loop',
    universal_params=[few_shot_params, reasoning_param],
    opt_config=inner_opt_config,
    save_ckpt_interval=1,
)

# opt_layer_configs = [inner_loop_config]

outer_opt_config = flow.OptConfig(
    n_trials=8,
    throughput=4,
    log_dir=save_to_dir,
    frugal_eval_cost=False,
)

outer_loop_config = driver.LayerConfig(
    layer_name='outer_loop',
    universal_params=[general_ensemble_params], # will overwrite module name
    opt_config=outer_opt_config,
    save_ckpt_interval=1,
    use_SH_allocation=True,
)

opt_layer_configs = [outer_loop_config, inner_loop_config]
