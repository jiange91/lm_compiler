from compiler.optimizer.params.reasoning import ReasonThenFormat
from compiler.utils import load_api_key
from compiler.llm import *
from compiler.llm.model import APICompatibleMessage
from litellm import ModelResponse, completion
from typing import List
import re
import copy

class TreeOfThought(ReasonThenFormat):
    def __init__(self, max_depth=3, beam_width=2):
        super().__init__("TreeOfThought")
        self.max_depth = max_depth
        self.beam_width = beam_width

    def reasoning_step(
        self,
        model: str,
        chat_messages: List[APICompatibleMessage],
        model_kwargs: dict
    ) -> List[ModelResponse]:
        self.model = model
        self.model_kwargs = model_kwargs
        candidate_paths = [copy.deepcopy(chat_messages)]
        for step in range(self.max_depth):
            all_new_candidates = []
            for path in candidate_paths:
                # Generate new continuations for this path
                continuations = self.generate_candidates(path)
                all_new_candidates.extend(continuations)
            if not all_new_candidates:
                break  # No new candidates generated
            # Evaluate all new candidates
            scored_candidates = self.evaluate_candidates(all_new_candidates)
            # Select the top candidates
            candidate_paths = self.select_best_candidates(scored_candidates)
        # Choose the best candidate path
        best_path = candidate_paths[0] if candidate_paths else chat_messages
        # Return the reasoning steps (excluding the initial chat messages)
        reasoning_steps = best_path[len(chat_messages):]
        return reasoning_steps
    
    def generate_candidates(
        self, 
        current_messages: List[APICompatibleMessage]
        ) -> List[List[APICompatibleMessage]]:
        
        # Generate possible continuations
        continuations = []
        prompt = copy.deepcopy(current_messages)
        prompt.append({"role": "user", "content": "What should we consider next?"})
        for _ in range(self.beam_width):
            response = completion(self.model, prompt, **self.model_kwargs)
            self.model_responses.append(response)
            new_path = current_messages + [{"role": "assistant", "content": f"{response.choices[0].message.content}"}]
            continuations.append(copy.deepcopy(new_path))
        return continuations

    def evaluate_candidates(self, candidates: List[List[APICompatibleMessage]]) -> List[float]:
        # Evaluate candidates and return scores
        scored_candidates = []
        for candidate in candidates:
            # Create an evaluation prompt
            eval_prompt = candidate + [{"role": "user", "content": "Please rate the quality of the reasoning so far on a scale of 1 to 10."}]
            evaluation = completion(self.model, eval_prompt, **self.model_kwargs)
            self.model_responses.append(evaluation)
            score = self.extract_score(evaluation.choices[0].message.content)
            scored_candidates.append((score, candidate))
        return scored_candidates
    
    def extract_score(self, evaluation_content: str) -> float:
        # Extract a numerical score from the evaluation content
        match = re.search(r'\b([1-9]|10)\b', evaluation_content)
        if match:
            return float(match.group(0))
        else:
            return 0.0  # Default score if parsing fails

    def select_best_candidates(self, scored_candidates: list[tuple[float, list[APICompatibleMessage]]]) -> list[list[APICompatibleMessage]]:
        # Select the top candidates based on scores
        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        # Select the top candidates up to the beam width
        top_candidates = [candidate for score, candidate in scored_candidates[:self.beam_width]]
        return top_candidates
    