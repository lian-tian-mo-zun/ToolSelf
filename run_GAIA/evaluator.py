"""GAIA-style answer evaluator with optional LLM-as-judge fallback."""

from __future__ import annotations

import re
import time
from difflib import SequenceMatcher
from typing import Dict, Optional, Tuple

from openai import OpenAI


class GAIAEvaluator:
    """Evaluate model predictions against reference answers."""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        judge_model: str,
        max_retries: int = 3,
        verbose: bool = True,
    ):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.judge_model = judge_model
        self.max_retries = max_retries
        self.verbose = verbose
        self.total_judge_calls = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

    def exact_match(self, pred: str, gold: str) -> bool:
        return pred.strip() == gold.strip()

    def normalized_match(self, pred: str, gold: str) -> bool:
        def normalize(text: str) -> str:
            text = " ".join(text.strip().lower().split())
            for char in [".", ",", "!", "?", ";", ":", '"', "'"]:
                text = text.replace(char, "")
            return text

        return normalize(pred) == normalize(gold)

    def numeric_match(self, pred: str, gold: str, rel_tolerance: float = 0.01) -> Tuple[bool, Optional[float]]:
        def extract_number(text: str) -> Optional[float]:
            text = text.strip().replace(",", "").replace(" ", "")
            match = re.search(r"-?\d+\.?\d*(?:[eE][+-]?\d+)?", text)
            if not match:
                return None
            try:
                return float(match.group())
            except ValueError:
                return None

        pred_num = extract_number(pred)
        gold_num = extract_number(gold)
        if pred_num is None or gold_num is None:
            return False, None
        if gold_num == 0:
            return abs(pred_num) < 1e-9, pred_num
        return abs(pred_num - gold_num) / abs(gold_num) < rel_tolerance, pred_num

    def containment_match(self, pred: str, gold: str) -> bool:
        return gold.lower().strip() in pred.lower().strip()

    def fuzzy_match(self, pred: str, gold: str, threshold: float = 0.85) -> Tuple[bool, float]:
        score = SequenceMatcher(None, pred.lower().strip(), gold.lower().strip()).ratio()
        return score >= threshold, score

    def llm_judge(self, pred: str, gold: str, question: str) -> Tuple[bool, str, Dict]:
        prompt = f"""You are an expert evaluator for a question-answering benchmark. Determine if the model prediction is semantically equivalent to the reference answer.

Question:
{question}

Reference Answer:
{gold}

Model Prediction:
{pred}

Guidelines:
1. Consider semantic equivalence, not just exact text match.
2. For numerical answers, allow equivalent formats.
3. For text answers, allow different phrasings with the same meaning.
4. Ignore minor formatting differences.
5. If the prediction contains the correct answer plus extra context, mark it correct.
6. Be strict: the core information must match exactly.

Respond in this exact format:
VERDICT: [YES/NO]
EXPLANATION: [One sentence]
"""
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.judge_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=200,
                )
                content = response.choices[0].message.content or ""
                usage = response.usage
                prompt_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0
                completion_tokens = getattr(usage, "completion_tokens", 0) if usage else 0
                self.total_judge_calls += 1
                self.total_prompt_tokens += prompt_tokens
                self.total_completion_tokens += completion_tokens

                verdict_match = re.search(r"VERDICT:\s*(YES|NO)", content, re.IGNORECASE)
                explanation_match = re.search(r"EXPLANATION:\s*(.+?)(?:\n|$)", content, re.IGNORECASE | re.DOTALL)
                if verdict_match:
                    return (
                        verdict_match.group(1).upper() == "YES",
                        explanation_match.group(1).strip() if explanation_match else content,
                        {
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": completion_tokens,
                            "attempt": attempt + 1,
                        },
                    )
                raise ValueError(f"Could not parse verdict from judge response: {content}")
            except Exception as exc:
                if self.verbose:
                    print(f"[LLM Judge] attempt {attempt + 1} failed: {exc}")
                if attempt == self.max_retries - 1:
                    return False, f"Error: {exc}", {"error": str(exc)}
                time.sleep(2**attempt)
        return False, "Max retries exceeded", {"error": "max_retries"}

    def evaluate_single(self, pred: str, gold: str, question: str = "") -> Dict:
        results = {
            "prediction": pred,
            "gold": gold,
            "question": question,
            "exact_match": False,
            "normalized_match": False,
            "numeric_match": False,
            "numeric_value": None,
            "containment_match": False,
            "fuzzy_match": False,
            "fuzzy_score": 0.0,
            "llm_judge_verdict": False,
            "llm_judge_explanation": "",
            "llm_judge_stats": {},
            "final_verdict": False,
            "match_type": "none",
            "confidence": "low",
        }
        if self.exact_match(pred, gold):
            results.update({"exact_match": True, "final_verdict": True, "match_type": "exact", "confidence": "high"})
            return results
        if self.normalized_match(pred, gold):
            results.update({"normalized_match": True, "final_verdict": True, "match_type": "normalized", "confidence": "high"})
            return results
        numeric_ok, numeric_value = self.numeric_match(pred, gold)
        if numeric_ok:
            results.update({
                "numeric_match": True,
                "numeric_value": numeric_value,
                "final_verdict": True,
                "match_type": "numeric",
                "confidence": "high",
            })
            return results
        results["containment_match"] = self.containment_match(pred, gold)
        fuzzy_ok, fuzzy_score = self.fuzzy_match(pred, gold)
        results["fuzzy_match"] = fuzzy_ok
        results["fuzzy_score"] = fuzzy_score
        verdict, explanation, stats = self.llm_judge(pred, gold, question)
        results.update({
            "llm_judge_verdict": verdict,
            "llm_judge_explanation": explanation,
            "llm_judge_stats": stats,
            "final_verdict": verdict,
            "match_type": "llm_judge",
            "confidence": "medium" if verdict else "low",
        })
        return results

    def get_statistics(self) -> Dict:
        total_tokens = self.total_prompt_tokens + self.total_completion_tokens
        return {
            "total_llm_judge_calls": self.total_judge_calls,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_tokens": total_tokens,
            "avg_tokens_per_call": total_tokens / max(1, self.total_judge_calls),
        }
