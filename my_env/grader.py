def grade(env):
    errors = len(env._errors())
    steps = env.step_count

    score = 1.0
    score -= errors * 0.2
    score -= max(0, steps - 6) * 0.05

    return max(0.01, min(0.99, round(score, 2)))
