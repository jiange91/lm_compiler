from pydantic import BaseModel, Field
from abc import ABC, abstractmethod


class MetricBase(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def score(self, label, pred) -> float:
        pass


class ExactMatch(MetricBase):
    """Exact match score for two items"""

    def score(self, label, pred):
        return 1.0 if label == pred else 0.0


class F1(MetricBase):
    """F1 score for two un-ordered sets"""

    def score(self, label, pred):
        # Calculate true positives, false positives, and false negatives
        true_positives = len(label & pred)
        false_positives = len(pred - label)
        false_negatives = len(label - pred)

        if true_positives == 0:
            return 0

        # Calculate F1 score
        precision = true_positives / (true_positives + false_positives)
        recall = true_positives / (true_positives + false_negatives)
        f1 = 2 * precision * recall / (precision + recall)
        return f1


import re
import string
import unicodedata


def normalize_text(s):
    # Normalize Unicode characters
    s = unicodedata.normalize("NFD", s)
    # Convert to lowercase
    s = s.lower()
    # Remove punctuation
    s = "".join(ch for ch in s if ch not in string.punctuation)
    # Remove articles (a, an, the)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    # Fix extra whitespaces
    s = " ".join(s.split())
    return s


class F1Str(F1):
    """F1 score for two strings

    Will first tokenize the strings by space and calculate F1 score
    """

    def score(self, label, pred):
        label = set(normalize_text(label).split())
        pred = set(normalize_text(pred).split())
        return super().score(label, pred)


from cognify.llm.model import LMConfig
from cognify.llm.model import StructuredModel, Input, OutputFormat


class AgentJudge(MetricBase):
    """Use LLM to judge the quality of the output based on a criterion

    The criterion should instruct the judge agent how to score the output and give a numerical score.
    """

    class Judgement(BaseModel):
        score: float = Field(description="The score given by the judge")

    def __init__(
        self,
        criterion: str,
        model_config: LMConfig,
        need_ground_truth: bool = False,
    ):
        system_prompt = f"""
        You are an expert evaluator assessing outputs based on the following criterion:
        {criterion}.
        """
        if need_ground_truth:
            inputs = [Input(name="ground truth"), Input(name="output")]
        else:
            inputs = [Input(name="output")]
        self.score_agent = StructuredModel(
            agent_name="judge_agent",
            system_prompt=system_prompt,
            input_variables=inputs,
            output_format=OutputFormat(schema=AgentJudge.Judgement),
            lm_config=model_config,
        )

    def score(self, label, pred):
        result: AgentJudge.Judgement = self.score_agent(
            {
                "ground truth": label,
                "output": pred,
            }
        )
        return result.score
