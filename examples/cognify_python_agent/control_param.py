from cognify.optimizer.control_param import ControlParameter
from cognify.hub.search import codegen

# ================= Overall Control Parameter =================
optimize_control_param = codegen.create_search(evaluator_batch_size=40)