outer_indicators = [
    (0.4773892773892774, 0.00281712),
    (0.5, 0.00140676),
]
ready_to_run = [0, 1]

sorted_indicator_indices = sorted(
    range(len(outer_indicators)),
    key=lambda i: (-outer_indicators[i][0], outer_indicators[i][1])
)
runs_left_to_run = sorted_indicator_indices[:len(ready_to_run) // 2]
print([ready_to_run[i] for i in runs_left_to_run])