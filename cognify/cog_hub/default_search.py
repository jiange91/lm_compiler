from typing import Optional, Literal

from cognify.llm import LMConfig
from cognify.optimizer.core import driver, flow
from cognify.cog_hub import reasoning, ensemble, model_selection
from cognify.cog_hub.common import NoChange
from cognify.cog_hub.fewshot import LMFewShot
from cognify.cog_hub.reasoning import ZeroShotCoT, PlanBefore
from cognify.optimizer.control_param import ControlParameter


def create_light_search(
    n_trials: int = 10,
    quality_constraint: float = 1.0,
    evaluator_batch_size: int = 10,
    opt_log_dir: str = 'opt_results',
    model_selection_cog: model_selection.LMSelection = None,
):
    # Reasoning Parameter
    reasoning_param = reasoning.LMReasoning(
        [NoChange(), ZeroShotCoT()] 
    )

    # Few Shot Parameter
    few_shot_params = LMFewShot(2)

    # Layer Config
    inner_opt_config = flow.OptConfig(
        n_trials=n_trials,
    )
    params = [few_shot_params, reasoning_param]
    if model_selection_cog is not None:
        params.append(model_selection_cog)
    inner_loop_config = driver.LayerConfig(
        layer_name='light_opt_layer',
        universal_params=[few_shot_params, reasoning_param],
        opt_config=inner_opt_config,
    )

    # ================= Overall Control Parameter =================
    optimize_control_param = ControlParameter(
        opt_layer_configs=[inner_loop_config],
        opt_history_log_dir=opt_log_dir,
        evaluator_batch_size=evaluator_batch_size,
        quality_constraint=quality_constraint,
    )
    return optimize_control_param

def create_medium_search(
    n_trials: int = 30,
    quality_constraint: float = 1.0,
    evaluator_batch_size: int = 10,
    opt_log_dir: str = 'opt_results',
    model_selection_cog: model_selection.LMSelection = None,
):
    # Assign resource to each layer
    inner_trials = 15
    outer_trials = n_trials // inner_trials
    
    # ================= Inner Loop Config =================
    # Reasoning Parameter
    reasoning_param = reasoning.LMReasoning(
        [NoChange(), ZeroShotCoT()]
    )

    # Few Shot Parameter
    few_shot_params = LMFewShot(2)

    # Layer Config
    inner_opt_config = flow.OptConfig(
        n_trials=inner_trials,
    )
    params = [few_shot_params, reasoning_param]
    if model_selection_cog:
        params.append(model_selection_cog)
    inner_loop_config = driver.LayerConfig(
        layer_name='medium_inner',
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
        n_trials=outer_trials,
    )
    outer_loop_config = driver.LayerConfig(
        layer_name='medium_outer',
        universal_params=[general_ensemble_params],
        opt_config=outer_opt_config,
    )

    # ================= Overall Control Parameter =================
    optimize_control_param = ControlParameter(
        opt_layer_configs=[outer_loop_config, inner_loop_config],
        opt_history_log_dir=opt_log_dir,
        evaluator_batch_size=evaluator_batch_size,
        quality_constraint=quality_constraint,
    )
    return optimize_control_param

def create_heavy_search(
    n_trials: int = 30,
    quality_constraint: float = 1.0,
    evaluator_batch_size: int = 10,
    opt_log_dir: str = 'opt_results',
    model_selection_cog: model_selection.LMSelection = None,
):
    # Assign resource to each layer
    # Use SH resource allocation
    # Total trials = inner * (2 * outer - 1)
    inner_trials = 10
    outer_trials = (n_trials / inner_trials + 1) // 2
    
    # ================= Inner Loop Config =================
    # Reasoning Parameter
    reasoning_param = reasoning.LMReasoning(
        [NoChange(), ZeroShotCoT(), PlanBefore]
    )

    # Few Shot Parameter
    few_shot_params = LMFewShot(4)

    # Layer Config
    inner_opt_config = flow.OptConfig(
        n_trials=inner_trials,
    )
    
    params = [few_shot_params, reasoning_param]
    if model_selection_cog:
        params.append(model_selection_cog)
    inner_loop_config = driver.LayerConfig(
        layer_name='heavy_inner',
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
        n_trials=outer_trials,
        use_SH_allocation=True,
    )
    outer_loop_config = driver.LayerConfig(
        layer_name='heavy_outer',
        universal_params=[general_ensemble_params],
        opt_config=outer_opt_config,
    )

    # ================= Overall Control Parameter =================
    optimize_control_param = ControlParameter(
        opt_layer_configs=[outer_loop_config, inner_loop_config],
        opt_history_log_dir=opt_log_dir,
        evaluator_batch_size=evaluator_batch_size,
        quality_constraint=quality_constraint,
    )
    return optimize_control_param

def create_search(
    *,
    search_type: Literal['light', 'medium', 'heavy'] = 'light',
    n_trials: int = 10,
    quality_constraint: float = 1.0,
    evaluator_batch_size: int = 10,
    opt_log_dir: str = 'opt_results',
    model_selection_cog: model_selection.LMSelection | list[LMConfig] | None = None,
):
    if model_selection_cog is not None:
        if isinstance(model_selection_cog, list):
            model_selection_options = model_selection.model_option_factory(model_selection_cog)
            model_selection_cog = model_selection.LMSelection(
                'model_selection',
                model_selection_options,
            )
        assert isinstance(model_selection_cog, model_selection.LMSelection)
    
    if search_type == 'light':
        return create_light_search(n_trials, quality_constraint, evaluator_batch_size, opt_log_dir, model_selection_cog)
    elif search_type == 'medium':
        return create_medium_search(n_trials, quality_constraint, evaluator_batch_size, opt_log_dir, model_selection_cog)
    elif search_type == 'heavy':
        return create_heavy_search(n_trials, quality_constraint, evaluator_batch_size, opt_log_dir, model_selection_cog)
    else:
        raise ValueError(f"Invalid search type: {search_type}")