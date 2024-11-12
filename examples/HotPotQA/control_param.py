from cognify.optimizer.core import driver, flow
from cognify.cog_hub import reasoning, ensemble
from cognify.cog_hub.common import NoChange
from cognify.cog_hub.fewshot import LMFewShot
from cognify.cog_hub.reasoning import ZeroShotCoT
from cognify.optimizer.control_param import ControlParameter

# ================= Inner Loop Config =================
# Reasoning Parameter
reasoning_param = reasoning.LMReasoning(
    [NoChange(), ZeroShotCoT()] 
)
# Few Shot Parameter
few_shot_params = LMFewShot(4)

# Layer Config
inner_opt_config = flow.OptConfig(
    n_trials=4,
)
inner_loop_config = driver.LayerConfig(
    layer_name='inner_loop',
    universal_params=[few_shot_params, reasoning_param],
    opt_config=inner_opt_config,
)

# ================= Outer Loop Config =================
# Ensemble Parameter
general_usc_ensemble = ensemble.UniversalSelfConsistency(3)
general_ensemble_params = ensemble.ModuleEnsemble(
    [NoChange(), general_usc_ensemble]
)
# Layer Config
outer_opt_config = flow.OptConfig(
    n_trials=2,
    throughput=2,
)
outer_loop_config = driver.LayerConfig(
    layer_name='outer_loop',
    universal_params=[general_ensemble_params],
    opt_config=outer_opt_config,
    use_SH_allocation=False,
)

# ================= Overall Control Parameter =================
optimize_control_param = ControlParameter(
    opt_layer_configs=[outer_loop_config, inner_loop_config],
    opt_history_log_dir='test_pbar'
)