from scipy import stats
import numpy as np

def test(group_a, group_b, alpha=0.05):
    _, p_value = stats.ttest_ind(group_a, group_b, equal_var=False)
    return p_value < alpha


def permutation_test(group_a, group_b, num_permutations=1000, alpha=0.05):
    observed_diff = np.mean(group_a) - np.mean(group_b)
    combined = np.concatenate([group_a, group_b])
    count = 0
    
    for _ in range(num_permutations):
        np.random.shuffle(combined)
        new_a = combined[:len(group_a)]
        new_b = combined[len(group_a):]
        diff = np.mean(new_a) - np.mean(new_b)
        if abs(diff) >= abs(observed_diff):
            count += 1

    p_value = count / num_permutations
    return p_value < alpha


