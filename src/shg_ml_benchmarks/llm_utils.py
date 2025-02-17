"""Some useful classes and utilties for benchmarking SHG prediction
performance of LLMs.

"""

import logging
from dataclasses import dataclass
from enum import Enum

import pandas as pd
from pydantic_ai import Agent
from robocrys import StructureCondenser, StructureDescriber

STRUCTURE_CONDENSER = StructureCondenser(
    mineral_matcher=False, use_symmetry_equivalent_sites=True
)
STRUCTURE_DESCRIBER = StructureDescriber(
    describe_components=True, describe_component_makeup=True
)


class LLMInputStructureRepresentation(str, Enum):
    composition = "composition"
    composition_spacegroup = "composition_spacegroup"
    robocrystallographer = "robocrystallographer"


@dataclass
class ModelCard:
    label: str
    """The folder/benchmark name."""

    name: str
    """The model name in pydantic_ai."""

    in_context_learning: bool = False

    input_structure_repr: LLMInputStructureRepresentation = (
        LLMInputStructureRepresentation.composition
    )

    @property
    def tags(self):
        return (
            f"{self.label}-{self.input_structure_repr.value}" + "-icl"
            if self.in_context_learning
            else "-no-icl"
        )


def generate_prompt(instruction=None, input=None):
    if input:
        if instruction:
            return f"""The following is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. You will be scored against a ground truth dataset for this task.
        ### Instruction:
        {instruction}
        ### Input:
        {input}
        ### Response:"""
        else:
            return input

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


def evaluate(instruction, input=None, model=None):
    """Evaluation method from Darwin repo."""
    prompt = generate_prompt(instruction, input)
    logging.info("Prompt: %s", prompt)
    result = model.run_sync(prompt)
    response = result.data
    logging.info("Response: %s", response)
    logging.info("Usage: %s", result.usage())
    response = process_response(response)
    return response


def llm_predict_fn(model, structures, task_instruction: str | None):
    """A prediction function for the shg-ml-benchmarks that receives a
    pymatgen structure, maps it into the chosen string representation,
    then passes it to the LLM and returns the processed prediction.
    """
    if isinstance(structures, list):
        raise NotImplementedError("Batch prediction not yet implemented.")
    structure = structures

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

    response = evaluate(task_instruction, input=input_structure, model=model)
    try:
        response = float(response)
        logging.info("Input: %s - Response: %s", input_structure, response)
    except Exception:
        logging.error(
            "!!! Bad response for input: %s - Response: %s", input_structure, response
        )
        return 0.0

    return response


def llm_train_fn(
    train_df, target, subset: int | None = None, model_card: ModelCard | None = None
):
    """Creates a system prompt with some subset of the training data and returns a model."""

    if model_card is None:
        raise ValueError("Model card must be provided.")

    system_prompt = f"""
Given a description of a crystal structure ({model_card.input_structure_repr.value.replace("_", " ")}), predict its second-harmonic generation (SHG) coefficient in the Kurtz-Perry form in pm/V.
All the structures you see will be non-centrosymmetric.
In our dataset, the SHG coefficients are computed with DFPT at the PBE level.

Most structures exhibit low SHG coefficients (below 10 pm/V), with exemplary materials ranging up to 170 pm/V.
Simply respond with the value which will be read as a raw float, do not provide any explanation.
    """

    if subset is not None:
        # Let pandas print the whole df as a string
        pd.set_option("display.max_rows", None)
        # Choose a subset of experimental structures (i.e., MP entries with ICSD matches)
        reasonable_subset = train_df[
            (train_df["src_theoretical"] == False)  # noqa
            & (train_df["src_bandgap"] < 4)
            & (train_df["src_bandgap"] > 0.05)
            & (train_df["src_ehull"] < 0.001)
            & (train_df["n"] > 1)
        ].sort_values("FOM")[["formula_reduced", "spg_symbol", "dKP_full_neum"]]
        reasonable_subset.rename("dKP_full_neum", "SHG coefficient", inplace=True)

        low_examples = reasonable_subset.head(subset).to_string(index=False)
        high_examples = reasonable_subset.tail(subset).to_string(index=False)

        # Add the pandas dataframe display of some columns
        system_prompt += f"\n\nLow SHG examples include: \n\n{low_examples}"
        system_prompt += f"\n\nHigh SHG examples include: \n\n{high_examples}"

    model = Agent(
        model_card.name,
        system_prompt=system_prompt,
        deps_type=float,
    )
    if model_card:
        model.label = model_card.label
        model.tags = model_card.tags
        model.input_structure_repr = model_card.input_structure_repr
        model.in_context_learning = model_card.in_context_learning
        model.meta = {"system_prompt": system_prompt, "subset": subset}

    return model
