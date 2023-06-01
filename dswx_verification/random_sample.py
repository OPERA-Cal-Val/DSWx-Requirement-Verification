import numpy as np


def get_equal_samples_per_label(flat_array_of_labels: np.ndarray,
                                target_labels: list,
                                total_target_sample_size: int) -> list:
    """Generate the total samples per label in labels given the target sample size. If there are less labels than the
    target_sample_size, then we only sample the maximum number of labels (i.e. the available labels)

    Parameters
    ----------
    flat_array_of_labels : np.ndarray
        Array (with only one dimension) of labels
    target_labels : list
        Unique list of labels which will be sampled from
    total_target_sample_size : int

    Returns
    -------
    list
       The number of samples to select with the same index of the corresponding label in labels
    """
    desired_sample_size_per_label = int(np.ceil(total_target_sample_size / len(target_labels)))

    if len(target_labels) != len(set(target_labels)):
        raise ValueError('Each label in "labels" should be unique"')

    samples_per_label = []
    for label in target_labels:
        n_pixels_per_label = (flat_array_of_labels == label).sum()
        n_pixels_to_sample = min(desired_sample_size_per_label, n_pixels_per_label)
        samples_per_label.append(n_pixels_to_sample)
    return samples_per_label


def generate_random_indices_for_classes(flat_array_of_labels: np.ndarray,
                                        labels: list = None,
                                        total_target_sample_size: int = 1_000,
                                        n_trials: int = 100,
                                        seed: int = 0) -> np.ndarray:
    np.random.seed(seed)
    labels = labels or [0, 1, 2]
    samples_per_label = get_equal_samples_per_label(flat_array_of_labels,
                                                    labels,
                                                    total_target_sample_size)

    m, n = n_trials, sum(samples_per_label)
    all_samples = np.full((m, n), -1)

    for k in range(n_trials):
        indices_for_trial = []
        for label, n_samples in zip(labels, samples_per_label):
            # Source: https://stackoverflow.com/a/50798546
            indices_for_label = np.where(flat_array_of_labels == label)[0]
            indices_for_label_random_subset = list(np.random.choice(indices_for_label, n_samples))
            # We cast it as a list so it can be appended in the sample dimension
            indices_for_trial += indices_for_label_random_subset
        all_samples[k, :] = indices_for_trial
    return all_samples
