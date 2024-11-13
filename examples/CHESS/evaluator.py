from cognify.optimizer import register_opt_program_entry, register_opt_score_fn

@register_opt_score_fn
def eval(label, stats):
    """
    Evaluate the statistics of the run.
    """
    correct = any(vs['correct'] == 1 for vs in stats['counts'].values())
    return 1.0 if correct else 0.0