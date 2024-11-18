from cognify.optimizer.control_param import ControlParameter
from cognify.cog_hub.optim_setup.codegen import CodegenSetup

# ================= Overall Control Parameter =================
optimize_control_param = ControlParameter(
    opt_setup=CodegenSetup(),
    evaluator_batch_size=40,
)