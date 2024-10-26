quality_constraint: float = 0.95
train_down_sample: int = 50
val_down_sample: int = 25
opt_history_log_dir: str = '/mnt/ssd4/lm_compiler/clean_examples/HotPotQA/opt_results'
evaluator_parallel: int = 20

from compiler.IR.llm import LMConfig
from compiler.optimizer.core import driver, flow

from compiler.optimizer.params import reasoning, model_selection
from compiler.optimizer.params import ensemble
from compiler.optimizer.params.common import IdentityOption
from compiler.optimizer.params.fewshot import LMFewShot
from compiler.optimizer.params.reasoning import ZeroShotCoT

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
    n_trials=6,
    throughput=2,
)
inner_loop_config = driver.LayerConfig(
    layer_name='inner_loop',
    universal_params=[few_shot_params, reasoning_param],
    opt_config=inner_opt_config,
)

# ================= Outer Loop Config =================
outer_opt_config = flow.OptConfig(
    n_trials=8,
    throughput=4,
)

outer_loop_config = driver.LayerConfig(
    layer_name='outer_loop',
    universal_params=[general_ensemble_params],
    opt_config=outer_opt_config,
)

opt_layer_configs = [outer_loop_config, inner_loop_config]

