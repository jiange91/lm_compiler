from compiler.cog_hub.reasoning import ReasonThenFormat
from compiler.utils import load_api_key
from compiler.IR.llm import LLMPredictor 
from compiler.langchain_bridge.interface import LLMTracker, LangChainLM, LangChainSemantic
from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    merge_message_runs,
    BaseMessage,
)
from langchain_core.prompts import ChatPromptTemplate
import re


vote_prompt = '''Given an instruction and several choices, decide which choice is most promising. Analyze each choice in detail, then conclude in the last line "The best choice is {s}", where s the integer id of the choice.
'''

compare_prompt = '''Briefly analyze the coherency of the following two passages. Conclude in the last line "The more coherent passage is 1", "The more coherent passage is 2", or "The two passages are similarly coherent".
'''

score_prompt = '''Analyze the following passage, then at the last line conclude "Thus the coherency score is {s}", where s is an integer from 1 to 10.
'''

standard_prompt = '''
Write a coherent passage of 4 short paragraphs. The end sentence of each paragraph must be: {input}
'''
class TreeOfThought(ReasonThenFormat):
    """
    Implementation adopted from https://github.com/princeton-nlp/tree-of-thought-llm
    """
    def __init__(self, max_depth=3, beam_width=2):
        super().__init__("TreeOfThought")
        self.max_depth = max_depth
        self.beam_width = beam_width
        
    def describe(self):
        desc = """
        - Tree-of-Thoughts -
        The agent explores a tree of possible continuations and use the most promising path as the final rationale.
        """
        return desc

    def reasoning_step(
        self, 
        chat_messages: list[BaseMessage], 
        lm: ChatOpenAI, 
    ) -> list[BaseMessage]:
        """
        Generation, Evaluation, Selection
        Args:
            chat_messages (list[BaseMessage]): _description_
            lm (ChatOpenAI): _description_

        Returns:
            list[BaseMessage]: _description_
        """
        self.lm = lm
        candidate_paths = [chat_messages.copy()]
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
        current_messages: list[BaseMessage]
        ) -> list[list[BaseMessage]]:
        
        # Generate possible continuations
        continuations = []
        prompt = current_messages.copy()
        prompt.append(HumanMessage(content="What should we consider next?"))
        for _ in range(self.beam_width):
            response = self.lm.invoke(prompt)
            new_path = current_messages + [response]
            continuations.append(new_path)
        return continuations

    def evaluate_candidates(self, candidates: list[list[BaseMessage]]) -> list[float]:
        # Evaluate candidates and return scores
        scored_candidates = []
        for candidate in candidates:
            # Create an evaluation prompt
            eval_prompt = candidate + [HumanMessage(content="Please rate the quality of the reasoning so far on a scale of 1 to 10.")]
            evaluation = self.lm.invoke(eval_prompt)
            score = self.extract_score(evaluation.content)
            scored_candidates.append((score, candidate))
        return scored_candidates
    
    def extract_score(self, evaluation_content: str) -> float:
        # Extract a numerical score from the evaluation content
        match = re.search(r'\b([1-9]|10)\b', evaluation_content)
        if match:
            return float(match.group(0))
        else:
            return 0.0  # Default score if parsing fails

    def select_best_candidates(self, scored_candidates: list[tuple[float, list[BaseMessage]]]) -> list[list[BaseMessage]]:
        # Select the top candidates based on scores
        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        # Select the top candidates up to the beam width
        top_candidates = [candidate for score, candidate in scored_candidates[:self.beam_width]]
        return top_candidates
    
    
    
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, BaseMessage
from compiler.langchain_bridge.interface import LangChainLM, LangChainSemantic

# Initialize the language model
# load_api_key('secrets.toml')
# lm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

# Define the semantic configuration (modify as per your framework)
# semantic = LangChainSemantic(
#     system_prompt="You are a helpful assistant.",
#     inputs=["question"],
#     output_format=None,
# )

# # Initialize the language model module
# lm_module = LangChainLM(
#     name="MyLangChainLM",
#     lm=lm,
#     semantic=semantic,
# )

# # Apply the Tree of Thought reasoning strategy
# tree_of_thought = TreeOfThought(max_depth=3, beam_width=2)
# tree_of_thought.apply(lm_module)

# img_lm = LangChainLM('img understanding', img_understanding_semantic)
# img_lm.lm_config = {'model': "gpt-4o-mini", 'temperature': 0.0}