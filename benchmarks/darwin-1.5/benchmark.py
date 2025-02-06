# /// script
# requires-python = "==3.11.*"
# dependencies = [
#    "shg_ml_benchmarks @ git+https://github.com/modl-uclouvain/shg-ml-benchmarks",
#    "fire>=0.7.0",
#    "numpy>=2.2.2",
#    "openai>=1.61.0",
#    "rouge-score>=0.1.2",
#    "sentencepiece>=0.2.0",
#    "tokenizers>=0.13.3",
#    "torch>=2.6.0",
#    "transformers>=4.28.1",
#    "wandb>=0.19.5",
#    "accelerate >= 0.26",
#    "robocrys >= 0.2.10",
# ]
# ///
"""Modified from https://github.com/MasterAI-EAM/Darwin/commit/d6961e30f58af6943e9b66798098a232fb7f6b42."""

import logging
from enum import Enum
from functools import partial
from pathlib import Path

from robocrys import StructureCondenser, StructureDescriber

logging.basicConfig(level=logging.WARNING)

import torch
from transformers import LlamaForCausalLM, LlamaTokenizer

from shg_ml_benchmarks import run_benchmark
from shg_ml_benchmarks.utils import SHG_BENCHMARK_SPLITS

STRUCTURE_CONDENSER = StructureCondenser(
    mineral_matcher=False, use_symmetry_equivalent_sites=True
)
STRUCTURE_DESCRIBER = StructureDescriber(
    describe_components=True, describe_component_makeup=True
)

CONTEXT_WINDOW: int = 512
"""This seems to be the maximum number of tokens that can be processed by the model at once."""


class LLMInputStructureRepresentation(str, Enum):
    composition = "composition"
    composition_spacegroup = "composition_spacegroup"
    robocrystallographer = "robocrystallographer"
    cif = "cif"


def generate_prompt(instruction, input=None):
    """Prompt generation from Darwin repo."""
    if input:
        return f"""The following is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
    ### Instruction:
    {instruction}
    ### Input:
    {input}
    ### Response:"""
    else:
        return f"""The following is an instruction that describes a task. Write a response that appropriately completes the request.
    ### Instruction:
    {instruction}
    ### Response:"""


def process_response(response):
    """Process response from Darwin repo."""
    try:
        response = response.split("Response: ")[1].split("\n")[0]
    except Exception:
        logging.warning(f"Response could not be processed: {response=}")
        response = None
    return response


def evaluate(
    instruction,
    input=None,
    temperature=0.8,
    top_p=0.75,
    top_k=40,
    do_sample=True,
    repetition_penalty=1.0,
    max_new_tokens=256,
    **kwargs,
):
    """Evaluation method from Darwin repo."""
    prompt = generate_prompt(instruction, input)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")

    if len(input_ids) > CONTEXT_WINDOW - 1:
        logging.error("Input too long, clipping prompt: %s", prompt)
        input_ids = input_ids[: CONTEXT_WINDOW - 1]

    generated_ids = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        repetition_penalty=repetition_penalty,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        **kwargs,
    )
    response = tokenizer.decode(generated_ids[0])
    response = process_response(response)
    torch.cuda.empty_cache()
    return response


def predict_fn(model, structure, task_instruction: str):
    """A prediction function for the shg-ml-benchmarks that receives a
    pymatgen structure, maps it into the chosen string representation,
    then passes it to the LLM and returns the processed prediction.
    """
    if model.input_structure_repr == LLMInputStructureRepresentation.composition:
        input_structure = structure.composition.reduced_formula
    elif (
        model.input_structure_repr
        == LLMInputStructureRepresentation.composition_spacegroup
    ):
        input_structure = f"{structure.composition.reduced_formula} in {structure.get_space_group_info()[0]}"
    elif (
        model.input_structure_repr
        == LLMInputStructureRepresentation.robocrystallographer
    ):
        try:
            input_structure = STRUCTURE_DESCRIBER.describe(
                STRUCTURE_CONDENSER.condense_structure(structure)
            )
        except Exception as exc:
            logging.error(
                "Failed default structure repr generation for input: %s. Error: %s",
                structure.reduced_formula,
                str(exc),
            )
            # Some structures fail symmetrisation
            input_structure = STRUCTURE_DESCRIBER.describe(
                StructureCondenser(
                    mineral_matcher=False, use_symmetry_equivalent_sites=False
                ).condense_structure(structure)
            )
    else:
        raise NotImplementedError(
            f"Input structure representation {model.input_structure_repr} not yet implemented."
        )

    response = evaluate(task_instruction, input=input_structure)
    try:
        response = float(response)
        logging.info("Input: %s - Response: %s", input_structure, response)
    except Exception:
        logging.error(
            "!!! Bad response for input: %s - Response: %s", input_structure, response
        )
        return 0.0
    return response


def load_model():
    model_path = Path(__file__).parent / "model"
    tokenizer = LlamaTokenizer.from_pretrained(model_path)

    model = LlamaForCausalLM.from_pretrained(
        model_path, load_in_8bit=False, torch_dtype=torch.float16, device_map="auto"
    )

    return model, tokenizer


system_prompt: str | None = None

input_structure_repr: LLMInputStructureRepresentation = (
    LLMInputStructureRepresentation.composition
)

model, tokenizer = load_model()

for input_structure_repr in LLMInputStructureRepresentation:
    task_instruction: str = f"Given this description of a crystal structure ({input_structure_repr.value.replace('_', ' ')}), predict its second-harmonic generation coefficient in the Kurtz-Perry form."
    for split in SHG_BENCHMARK_SPLITS:
        model.input_structure_repr = input_structure_repr
        model.in_context_learning = False
        model.label = "darwin-1.5"
        model.tags = f"darwin-1.5-{input_structure_repr.value}"
        if model.in_context_learning:
            model.tags += "-icl"
        else:
            model.tags += "-no-icl"
        logging.info("Running benchmark %s for split %s", model.tags, split)
        try:
            run_benchmark(
                model=model,
                predict_fn=partial(predict_fn, task_instruction=task_instruction),
                task=split,
                train_fn=None,
            )
        except NotImplementedError:
            logging.error(
                "NotImplementedError for split %s, repr %s", split, input_structure_repr
            )
            continue
