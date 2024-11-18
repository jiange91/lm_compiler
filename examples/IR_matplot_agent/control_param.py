from cognify.optimizer.control_param import ControlParameter
from cognify.cog_hub.optim_setup.datavis import DataVisSetup

# ================= Overall Control Parameter =================
optimize_control_param = ControlParameter(
    opt_setup=DataVisSetup(),
    opt_history_log_dir='opt_results',
    evaluator_batch_size=50,
)