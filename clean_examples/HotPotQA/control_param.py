
from compiler.optimizer.core import driver, flow
from compiler.optimizer.params import reasoning, ensemble
from compiler.optimizer.params.common import IdentityOption
from compiler.optimizer.params.fewshot import LMFewShot
from compiler.optimizer.params.reasoning import ZeroShotCoT
from compiler.optimizer.control_param import ControlParameter

# ================= Inner Loop Config =================
# Reasoning Parameter
reasoning_param = reasoning.LMReasoning(
    [IdentityOption(), ZeroShotCoT()] 
)
# Few Shot Parameter
few_shot_params = LMFewShot("few_shot", 4)

# Layer Config
inner_opt_config = flow.OptConfig(
    n_trials=6,
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
    [IdentityOption(), general_usc_ensemble]
)
# Layer Config
outer_opt_config = flow.OptConfig(
    n_trials=4,
)
outer_loop_config = driver.LayerConfig(
    layer_name='outer_loop',
    universal_params=[general_ensemble_params],
    opt_config=outer_opt_config,
)

# ================= Overall Control Parameter =================
optimize_control_param = ControlParameter(
    opt_layer_configs=[outer_loop_config, inner_loop_config],
)