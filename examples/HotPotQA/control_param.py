from cognify.optimizer.control_param import ControlParameter
from cognify.cog_hub.optim_setup.qa import QASetup

# ================= Overall Control Parameter =================
optimize_control_param = ControlParameter(
    opt_setup=QASetup(),
    opt_history_log_dir='test_pbar'
)