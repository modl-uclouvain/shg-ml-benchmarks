import logging
import os
from functools import partial

logging.basicConfig(level=logging.INFO)

from shg_ml_benchmarks import run_benchmark
from shg_ml_benchmarks.llm_utils import (
    LLMInputStructureRepresentation,
    ModelCard,
    llm_predict_fn,
    llm_train_fn,
)
from shg_ml_benchmarks.utils import SHG_BENCHMARK_SPLITS

system_prompt: str | None = None

input_structure_repr: LLMInputStructureRepresentation = (
    LLMInputStructureRepresentation.composition
)

if os.getenv("ANTHROPIC_API_KEY") is None:
    raise SystemExit("ANTHROPIC_API_KEY not set in environment variables.")

for input_structure_repr in LLMInputStructureRepresentation:
    task_instruction: str | None = None

    # The top n and bottom n structures to use in the system prompt
    subset: int = 200

    if input_structure_repr is LLMInputStructureRepresentation.robocrystallographer:
        continue

    for in_context_learning in [True]:  # , False]:
        for split in SHG_BENCHMARK_SPLITS:
            # Use a different system prompt per split to avoid data leakage

            model_card = ModelCard(
                label="claude-3.5-sonnet",
                name="anthropic:claude-3-5-sonnet-latest",
                input_structure_repr=input_structure_repr,
                in_context_learning=in_context_learning,
            )

            logging.info("Running benchmark %s for split %s", model_card.tags, split)
            try:
                run_benchmark(
                    model=model_card,
                    predict_fn=partial(
                        llm_predict_fn, task_instruction=task_instruction
                    ),
                    train_fn=partial(
                        llm_train_fn,
                        subset=subset if model_card.in_context_learning else None,
                        model_card=model_card,
                    ),
                    task=split,
                    predict_individually=True,
                )
            except NotImplementedError:
                logging.error(
                    "NotImplementedError for split %s, repr %s",
                    split,
                    input_structure_repr,
                )
                continue
