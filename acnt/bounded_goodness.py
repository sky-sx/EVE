"""Bounded, unique final goodness with zero evaluation correction at equality."""
import math
import torch
from .goodness_prediction import GoodnessPredictionLearner
from experiments.calibrated_write_exploration import ExploringLearner


def combined_goodness(raw_score, prediction, *, floor=.2, penalty=.2):
    """Solve G=B-lambda*(prediction-G)^2, B=floor+(1-floor)*R.

    With 0<=lambda<=floor<1 and lambda<1/2, the map sends [0,1] to
    itself and is a contraction. The rationalized positive quadratic branch
    is the unique root in [0,1]. Clamping handles <=2e-15 roundoff only.
    """
    values = (raw_score, prediction, floor, penalty)
    if any(not math.isfinite(v) for v in values):
        raise ValueError('finite inputs required')
    if not 0<=raw_score<=1 or not 0<=prediction<=1:
        raise ValueError('score and prediction must lie in [0,1]')
    if not 0<floor<1 or not 0<=penalty<=floor or not penalty<.5:
        raise ValueError('require 0<=penalty<=floor<1 and penalty<0.5')
    base = floor+(1-floor)*raw_score
    if prediction == base or penalty == 0:
        total = base
    else:
        delta = base-prediction
        total = prediction+2*delta/(1+math.sqrt(1+4*penalty*delta))
        if total < -2e-15 or total > 1+2e-15:
            raise ArithmeticError('root outside proved interval')
        total = min(1.,max(0.,total))
    correction = total-base
    residual = total-base+penalty*(prediction-total)**2
    if abs(residual)>2e-14:
        raise ArithmeticError('combined goodness root residual too large')
    return dict(goodness=total, main_base=base, evaluation_correction=correction,
                prediction_squared_error=(prediction-total)**2,
                root_residual=residual)


class BoundedGoodnessLearner(GoodnessPredictionLearner):
    """Only total G enters sensory feedback, calibration, actor and evaluator.

    Evaluator credit uses dG/dq, where p=sigmoid(q), inferred from its own
    pre-feedback p and the single received G. It receives no raw action score.
    Existing clipped streaming tangents and stopped statistics remain.
    """
    def __init__(self, kernel, *, penalty=.2, **settings):
        if not math.isfinite(penalty) or not 0<penalty<.5:
            raise ValueError('evaluation penalty must be in (0,0.5)')
        super().__init__(kernel, **settings)
        self.penalty = penalty
        self.evaluation_logit_coefficient = 0.

    @torch.no_grad()
    def commit(self, eligibility, group_signals):
        if self.teacher_for_commit is None or self.pending_prediction is None:
            raise RuntimeError('single total feedback must follow prediction')
        p, g = self.pending_prediction, self.teacher_for_commit
        # dG/dp=2*lambda*(G-p)/(1+2*lambda*(G-p)), then dp/dq=p*(1-p).
        coefficient = 2*self.penalty*(g-p)/(1+2*self.penalty*(g-p))*p*(1-p)
        self.evaluation_logit_coefficient = coefficient
        auxiliary = self.evaluation_weight*coefficient*self.evaluation_traces.mean(0)
        self.auxiliary_direction_norm = float(auxiliary.norm())
        # Bypass the historical separate BCE auxiliary; use total-objective
        # credit while preserving the same parameter-write actuator.
        return ExploringLearner.commit(self, eligibility+auxiliary, group_signals)

    def step(self, **settings):
        diagnostics = super().step(**settings)
        if settings.get('goodness') is not None:
            diagnostics['evaluation_logit_coefficient'] = self.evaluation_logit_coefficient
        return diagnostics
