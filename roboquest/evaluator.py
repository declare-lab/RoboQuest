"""Evaluator-side consumer of the scene's frozen first-press score; no simulator imports.

The scene owns physical Submit detection and freezes the score at the first
press. This evaluator only checks the contract and requires a submission:
satisfying the goal without pressing never counts. Where the scene reports the
v1 protocol fields (spec 1.1 and 1.4) it checks them too, and re-derives the
submitted outcomes from its own success so the two can never disagree.
"""
from copy import deepcopy

OUTCOMES = ('success', 'wrong_submit', 'timeout')   # spec 1.1; None while the episode runs


def first_submit_evaluator(contract_version):
    class FirstSubmitEvaluator:
        expected_contract = contract_version

        def __init__(self, spec):
            if spec.get('contract_version') != contract_version:
                raise ValueError(f'Unexpected task contract: {spec.get("contract_version")!r} != {contract_version!r}')

        def update(self, snapshot):
            return self.score(snapshot)

        def score(self, snapshot):
            result = deepcopy(snapshot['score'])
            if type(result.get('success')) is not bool or type(result.get('submitted')) is not bool:
                raise ValueError('First-submit evaluator requires boolean success and submitted')
            result['success'] = result['success'] and result['submitted']
            if 'outcome' in result:
                if result['outcome'] not in OUTCOMES and result['outcome'] is not None:
                    raise ValueError(f'Unknown episode outcome: {result["outcome"]!r}')
                if result['submitted']:      # the press decides, and success is this evaluator's
                    result['outcome'] = 'success' if result['success'] else 'wrong_submit'
            if 'progress' in result:
                progress = result['progress']
                if isinstance(progress, bool) or not isinstance(progress, (int, float)) or not 0. <= progress <= 1.:
                    raise ValueError(f'Progress must be a number in [0, 1], not {progress!r}')
                result['progress'] = float(progress)
            return result

    FirstSubmitEvaluator.__name__ = 'FirstSubmitEvaluator'
    return FirstSubmitEvaluator
