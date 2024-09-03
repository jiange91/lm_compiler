from compiler.IR.modules import LLMPredictor, LMConfig
import logging
import openai
from reviewer import perform_review

logger = logging.getLogger(__name__)

class ScientistLM(LLMPredictor):
    def __init__(self, name, kernel) -> None:
        super().__init__(name, kernel)
    
    def set_lm(self):
        logger.debug(f'Setting LM for {self.name}: {self.lm_config}')
        model_name: str = self.lm_config['model']
        self.lm = model_name
        self.kernel.lm = self.lm
        return
    
    def get_lm_history(self):
        return None
    
def grade_kernel(paper_text):
    llm = grade_kernel.lm
    client = openai.Client()
    review = perform_review(paper_text, llm, client, num_fs_examples=0)
    return {'review': review}

from typing import Literal
from langchain_core.output_parsers import NumberedListOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

summary_gen_system = f"""
You are an AI researcher tasked with summarizing a research paper submitted to a prestigious ML venue. 
Your goal is to produce a concise and accurate summary of the paper’s content and contributions. 
Focus on the main ideas and ensure that the summary is objective and something the authors would agree with.
"""

summary_generation_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", summary_gen_system),
        ("human", "Paper: \n{paper_text}\n"),
    ]
)

def generate_summary(paper_text):
    llm = generate_summary.lm
    summary_gen = summary_generation_prompt | llm | StrOutputParser()
    summary = summary_gen.invoke({"paper_text": paper_text})
    return {'summary': summary}

strength_weakness_template = """
You are an AI reviewer focused on evaluating the strengths and weaknesses of a research paper. 
Assess the paper critically across the following dimensions: originality, quality, clarity, and significance. 
Provide a thorough and balanced analysis that highlights both the strong and weak points of the paper.

Expected output:
- List of strengths
- List of weaknesses
- Originality: 1 to 4 rating
- Quality: 1 to 4 rating
- Clarity: 1 to 4 rating
- Significance: 1 to 4 rating

Paper:
{paper_text}
"""

s_n_w_prompt = ChatPromptTemplate.from_template(strength_weakness_template)
def strength_weakness_kernel(paper_text):
    llm = strength_weakness_kernel.lm
    s_n_w_gen = s_n_w_prompt | llm | StrOutputParser()
    s_n_w = s_n_w_gen.invoke({"paper_text": paper_text})
    return {'strengths_weaknesses': s_n_w}


clarification_template = """
You are an AI reviewer preparing a set of clarifying questions and suggestions for the authors of a research paper. 
Your goal is to identify areas where further clarification or discussion could improve your understanding or evaluation of the paper. 
Consider how the authors' responses might influence your assessment.

Expected output:
A list of clarifying questions and suggestions.

Paper:
{paper_text}
"""

clarification_prompt = ChatPromptTemplate.from_template(clarification_template)
def clarification_kernel(paper_text):
    llm = clarification_kernel.lm
    clarifications_gen = clarification_prompt | llm | StrOutputParser()
    clarifications = clarifications_gen.invoke({"paper_text": paper_text})
    return {'clarifications': clarifications}

limitation_n_ethical_template = """
You are an AI reviewer focusing on identifying limitations and ethical concerns in a research paper and identified strength and weakness. 
Assess whether the authors have adequately addressed potential limitations of their work and any societal or ethical impacts. 
Provide constructive feedback and suggest areas for improvement if necessary.

Expected output:
- List of limitations
- Ethical concerns: Yes/No
- Suggestions for addressing limitations or ethical concerns

Paper:
{paper_text}

Strongth and Weaknesses:
{strengths_weaknesses}
"""
l_n_e_prompt = ChatPromptTemplate.from_template(limitation_n_ethical_template)
def limitation_n_ethical_kernel(paper_text, strengths_weaknesses):
    llm = limitation_n_ethical_kernel.lm
    l_n_e_gen = l_n_e_prompt | llm | StrOutputParser()
    l_n_e = l_n_e_gen.invoke({"paper_text": paper_text, "strengths_weaknesses": strengths_weaknesses})
    return {'limitations_ethical': l_n_e}

rating_n_decision_template = """
You are an AI reviewer responsible for assigning final ratings and making a decision on a research paper submission. 
Consider the inputs from previous analyses regarding the paper's strengths, weaknesses, limitations, and ethical considerations. 
Your goal is to provide a fair and informed final evaluation and make a clear decision to accept or reject the paper.
When providing your "overall score" for this submission. please use the following as a criteria:
  10: Award quality: Technically flawless paper with groundbreaking impact on one or more areas of AI, with exceptionally strong evaluation, reproducibility, and resources, and no unaddressed ethical considerations.
  9: Very Strong Accept: Technically flawless paper with groundbreaking impact on at least one area of AI and excellent impact on multiple areas of AI, with flawless evaluation, resources, and reproducibility, and no unaddressed ethical considerations.
  8: Strong Accept: Technically strong paper with, with novel ideas, excellent impact on at least one area of AI or high-to-excellent impact on multiple areas of AI, with excellent evaluation, resources, and reproducibility, and no unaddressed ethical considerations.
  7: Accept: Technically solid paper, with high impact on at least one sub-area of AI or moderate-to-high impact on more than one area of AI, with good-to-excellent evaluation, resources, reproducibility, and no unaddressed ethical considerations.
  6: Weak Accept: Technically solid, moderate-to-high impact paper, with no major concerns with respect to evaluation, resources, reproducibility, ethical considerations.
  5: Borderline accept: Technically solid paper where reasons to accept outweigh reasons to reject, e.g., limited evaluation. Please use sparingly.
  4: Borderline reject: Technically solid paper where reasons to reject, e.g., limited evaluation, outweigh reasons to accept, e.g., good evaluation. Please use sparingly.
  3: Reject: For instance, a paper with technical flaws, weak evaluation, inadequate reproducibility and incompletely addressed ethical considerations.
  2: Strong Reject: For instance, a paper with major technical flaws, and/or poor evaluation, limited impact, poor reproducibility and mostly unaddressed ethical considerations.
  1: Very Strong Reject: For instance, a paper with trivial results or unaddressed ethical considerations

Expected output:
- Soundness: 1 to 4 rating
- Presentation: 1 to 4 rating
- Contribution: 1 to 4 rating
- Overall: A rating from 1 to 10 (very strong reject to maintain conference quality).
- Confidence: 1 to 5 rating
- Decision: Accept/Reject
if accept, add "Congratulations" at the end

Paper summary: 
{summary}

Strengths and weaknesses:
{strengths_weaknesses}

Need for clarification:
{clarifications}

Limitations and ethical concerns:
{limitations_ethical}
"""

rating_n_decision_prompt = ChatPromptTemplate.from_template(rating_n_decision_template)
def rating_n_decision_kernel(summary, strengths_weaknesses, clarifications, limitations_ethical):
    llm = rating_n_decision_kernel.lm
    rating_n_decision_gen = rating_n_decision_prompt | llm | StrOutputParser()
    rating_n_decision = rating_n_decision_gen.invoke(
        {"summary": summary, 
         "strengths_weaknesses": strengths_weaknesses, 
         "clarifications": clarifications, 
         "limitations_ethical": limitations_ethical}
    )
    return {'rating_decision': rating_n_decision}