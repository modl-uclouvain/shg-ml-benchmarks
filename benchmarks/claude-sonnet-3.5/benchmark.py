import logging
import os
import time
from enum import Enum
from functools import partial

from robocrys import StructureCondenser, StructureDescriber

logging.basicConfig(level=logging.INFO)

from pydantic_ai import Agent

from shg_ml_benchmarks import run_benchmark
from shg_ml_benchmarks.utils import SHG_BENCHMARK_SPLITS

STRUCTURE_CONDENSER = StructureCondenser(
    mineral_matcher=False, use_symmetry_equivalent_sites=True
)
STRUCTURE_DESCRIBER = StructureDescriber(
    describe_components=True, describe_component_makeup=True
)

CONTEXT_WINDOW: int = 128_000


class LLMInputStructureRepresentation(str, Enum):
    composition = "composition"
    composition_spacegroup = "composition_spacegroup"
    robocrystallographer = "robocrystallographer"


def generate_prompt(instruction, input=None):
    if input:
        return f"""The following is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. You will be scored against a ground truth dataset for this task.
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
    try:
        response = float(response)
    except Exception:
        logging.warning(f"Response could not be processed: {response=}")
        response = None
    return response


def evaluate(instruction, input=None):
    """Evaluation method from Darwin repo."""
    prompt = generate_prompt(instruction, input)
    logging.debug("Prompt: %s", prompt)
    response = model.run_sync(prompt).data
    logging.debug("Response: %s", response)
    response = process_response(response)
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
    time.sleep(12)
    return response


system_prompt: str | None = None

input_structure_repr: LLMInputStructureRepresentation = (
    LLMInputStructureRepresentation.composition
)

if os.getenv("ANTHROPIC_API_KEY") is None:
    raise SystemExit("ANTHROPIC_API_KEY not set in environment variables.")

model = Agent("anthropic:claude-3-5-sonnet-latest")


for input_structure_repr in LLMInputStructureRepresentation:
    task_instruction: str = f"Given this description of a crystal structure ({input_structure_repr.value.replace('_', ' ')}), predict its second-harmonic generation coefficient in the Kurtz-Perry form in pm/V. Simply respond with the value which will be read as a raw float. Do not provide any explanation."
    for split in SHG_BENCHMARK_SPLITS:
        model.input_structure_repr = input_structure_repr
        model.in_context_learning = False
        model.label = "claude-sonnet-3.5"
        model.tags = f"{model.label}-{input_structure_repr.value}"
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
