import numpy as np
from scipy import stats

import optuna
from optuna.pruners import BasePruner
from optuna.study import StudyDirection
from optuna.trial._state import TrialState


class SignificancePruner(BasePruner):
    """Pruner to use statistical significance to prune repeated trainings like
    in a cross validation setting.

    As the test method a one-sided Mann-Whitney-U-Test is used.

    Args:
        alpha:
            The alpha level for the statistical significance test.
    """

    def __init__(self, alpha=0.05) -> None:
        self.alpha = alpha

    def prune(self, study: "optuna.study.Study", trial: "optuna.trial.FrozenTrial") -> bool:
        # get best tial - best trial is not available for first trial
        best_trial = None
        try:
            best_trial = study.best_trial
        except:
            pass

        if best_trial is not None:
            trial_intermediate_values = list(trial.intermediate_values.values())
            trial_mean = np.mean(trial_intermediate_values)

            best_trial_intermediate_values = list(best_trial.intermediate_values.values())
            best_trial_mean = np.mean(best_trial_intermediate_values)

            direction = study.direction
            if direction == StudyDirection.MAXIMIZE:
                alternative = 'less'
            elif direction == StudyDirection.MINIMIZE:
                alternative = 'greater'
            else:
                raise RuntimeError('Can not find valid StudyDirection!')

            if (trial_mean < best_trial_mean and direction == StudyDirection.MAXIMIZE) \
                    or (trial_mean > best_trial_mean and direction == StudyDirection.MINIMIZE):
                pvalue = stats.mannwhitneyu(
                    trial_intermediate_values,
                    best_trial_intermediate_values,
                    alternative=alternative,
                ).pvalue
                if pvalue < self.alpha:
                    return True

        return False
