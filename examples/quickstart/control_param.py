from compiler.cog_hub import reasoning, fewshot
from compiler.cog_hub.common import NoChange
from compiler.optimizer.control_param import ControlParameter
from compiler.optimizer.core import driver, flow

# Define search space
reasoning_param = reasoning.LMReasoning(
   [NoChange(), reasoning.ZeroShotCoT()]
)

fewshot_param = fewshot.LMFewShot(max_num=2)

# Define optimization layer
inner_opt_config = flow.OptConfig(
    n_trials=6,
)
single_layer_config = driver.LayerConfig(
   layer_name='simple_optimization_layer',
   universal_params=[reasoning_param, fewshot_param],
   opt_config=inner_opt_config,
)

# Register optimizer settings
optimize_control_param = ControlParameter(
   opt_layer_configs=[single_layer_config],
)