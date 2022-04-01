import functools
import math
from typing import *

import nltk
from rouge import Rouge


@functools.lru_cache(maxsize=128_000)
def bleu_cached(gold: Tuple[str], pred: Tuple[str]) -> float:
    if len(pred) == 0:
        return 0
    return nltk.translate.bleu_score.sentence_bleu(
        [list(gold)],
        list(pred),
        smoothing_function=nltk.translate.bleu_score.SmoothingFunction().method2,
        auto_reweigh=True,
    )


@functools.lru_cache(maxsize=128_000)
def token_acc_cached(gold: Tuple[str], pred: Tuple[str]) -> float:
    matches = len([
        i for i in range(min(len(gold), len(pred)))
        if gold[i] == pred[i]
    ])
    return matches / max(len(gold), len(pred))


@functools.lru_cache(maxsize=128_000)
def rouge_l_cached(gold: Tuple[str], pred: Tuple[str]) -> Dict[str, float]:
    # Replace the "." characters (e.g., in identifier names), otherwise they'll always be considered as sentence boundaries
    hyp = " ".join(pred).replace(".", "<DOT>")
    ref = " ".join(gold).replace(".", "<DOT>")

    if len(hyp) == 0 or len(ref) == 0:
        return {'r': 0, 'p': 0, 'f': 0}

    rouge = Rouge()
    scores = rouge.get_scores(hyps=hyp, refs=ref, avg=True)
    return scores['rouge-l']


@functools.lru_cache(maxsize=128_000)
def set_match_cached(gold: Tuple[str], pred: Tuple[str]) -> Dict[str, float]:
    if len(gold) == 0 or len(pred) == 0:
        return {"r": 0, "p": 0, "f": 0}

    gold_unique_tokens = set(gold)
    pred_unique_tokens = set(pred)
    match_tokens = gold_unique_tokens & pred_unique_tokens
    precision = min(len(match_tokens) / len(pred_unique_tokens), 1)
    recall = min(len(match_tokens) / len(gold_unique_tokens), 1)
    if precision == 0 or recall == 0:
        f1 = 0
    else:
        f1 = 2 / (1 / precision + 1 / recall)
    return {"r": recall, "p": precision, "f": f1}


@functools.lru_cache(maxsize=128_000)
def meteor_cached(gold: Tuple[str], pred: Tuple[str]) -> float:
    if len(gold) == 0 or len(pred) == 0:
        return 0

    return nltk.translate.meteor_score.single_meteor_score(
        " ".join(gold),
        " ".join(pred)
    )


@functools.lru_cache(maxsize=128_000)
def near_duplicate_similarity_cached(
        gold: Tuple[str],
        pred: Tuple[str],
        threshold: float = 0.1,
) -> float:
    """
    Computes the approximate token-level accuracy between gold and pred.

    Returns:
        token-level accuracy - if not exact match and mismatching tokens <= threshold;
        0 - otherwise.
    """
    mismatch_allowed = int(math.ceil(threshold * min(len(gold), len(pred))))

    # Check length difference
    if abs(len(gold) - len(pred)) >= mismatch_allowed:
        return 0

    # Count number of mismatches
    mismatch_count = 0
    max_len = max(len(gold), len(pred))
    for i in range(max_len):
        if i >= len(gold):
            mismatch_count += len(pred) - i
            break
        if i >= len(pred):
            mismatch_count += len(gold) - i
            break
        if gold[i] != pred[i]:
            mismatch_count += 1
            if mismatch_count >= mismatch_allowed:
                return 0
    if mismatch_count >= mismatch_allowed:
        return 0
    else:
        return 1 - mismatch_count / max_len


class EvalMetrics:

    @classmethod
    def batch_exact_match(cls, golds: List[Any], preds: List[Any]) -> List[float]:
        """
        return[i] = (golds[i] == preds[i]) ? 1 : 0
        """
        assert len(golds) == len(preds)

        results = []
        for gold, pred in zip(golds, preds):
            if gold == pred:
                results.append(1)
            else:
                results.append(0)
        return results

    @classmethod
    def token_acc(cls, gold: List[str], pred: List[str]) -> float:
        return token_acc_cached(tuple(gold), tuple(pred))

    @classmethod
    def batch_token_acc(cls, golds: List[List[str]], preds: List[List[str]]) -> List[float]:
        assert len(golds) == len(preds)
        return [
            token_acc_cached(tuple(gold), tuple(pred))
            for gold, pred in zip(golds, preds)
        ]

    @classmethod
    def bleu(cls, gold: List[str], pred: List[str]) -> float:
        """
        return = BLEU([gold], pred)
        """
        return bleu_cached(tuple(gold), tuple(pred))

    @classmethod
    def batch_bleu(cls, golds: List[List[str]], preds: List[List[str]]) -> List[float]:
        """
        return[i] = #bleu(golds[i], preds[i])
        """
        assert len(golds) == len(preds)
        return [
            bleu_cached(tuple(gold), tuple(pred))
            for gold, pred in zip(golds, preds)
        ]

    @classmethod
    def rouge_l(cls, gold: List[str], pred: List[str]) -> Dict[str, float]:
        """
        return = rouge l metric computed for given sequences
        """
        return rouge_l_cached(tuple(gold), tuple(pred))

    @classmethod
    def batch_rouge_l(cls, golds: List[List[str]], preds: List[List[str]]) -> List[Dict[str, float]]:
        """
        return[i] = #rouge_l(golds[i], preds[i])
        """
        assert len(golds) == len(preds)
        return [
            rouge_l_cached(tuple(gold), tuple(pred))
            for gold, pred in zip(golds, preds)
        ]

    @classmethod
    def set_match(cls, gold: List[str], pred: List[str]) -> Dict[str, float]:
        return set_match_cached(tuple(gold), tuple(pred))

    @classmethod
    def batch_set_match(cls, golds: List[List[str]], preds: List[List[str]]) -> List[Dict[str, float]]:
        assert len(golds) == len(preds)
        return [
            set_match_cached(tuple(gold), tuple(pred))
            for gold, pred in zip(golds, preds)
        ]

    @classmethod
    def batch_set_match_f1(cls, golds: List[List[str]], preds: List[List[str]]) -> List[float]:
        assert len(golds) == len(preds)
        return [
            set_match_cached(tuple(gold), tuple(pred))["f"]
            for gold, pred in zip(golds, preds)
        ]

    @classmethod
    def meteor(cls, gold: List[str], pred: List[str]) -> float:
        return meteor_cached(tuple(gold), tuple(pred))

    @classmethod
    def batch_meteor(cls, golds: List[List[str]], preds: List[List[str]]) -> List[float]:
        assert len(golds) == len(preds)
        return [
            meteor_cached(tuple(gold), tuple(pred))
            for gold, pred in zip(golds, preds)
        ]

    @classmethod
    def near_duplicate_similarity(cls, gold: List[str], pred: List[str]) -> float:
        return near_duplicate_similarity_cached(tuple(gold), tuple(pred))

    @classmethod
    def batch_near_duplicate_similarity(cls, golds: List[List[str]], preds: List[List[str]]) -> List[float]:
        assert len(golds) == len(preds)
        return [
            near_duplicate_similarity_cached(tuple(gold), tuple(pred))
            for gold, pred in zip(golds, preds)
        ]
