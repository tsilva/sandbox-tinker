from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import tinker
from tinker_cookbook import model_info, renderers
from tinker_cookbook.renderers import TrainOnWhat, get_text_content
from tinker_cookbook.supervised.common import datum_from_model_input_weights

from .data import Example
from .metrics import strict_label, tolerant_label


@dataclass(frozen=True)
class ParsedSample:
    raw_text: str
    decoded_tokens: str
    strict_prediction: str | None
    tolerant_prediction: str | None
    termination: str
    generated_tokens: int


def messages_for(example: Example, include_answer: bool) -> list[renderers.Message]:
    messages: list[renderers.Message] = [
        {"role": "system", "content": example.system_prompt},
        {"role": "user", "content": example.text},
    ]
    if include_answer:
        messages.append({"role": "assistant", "content": example.label})
    return messages


def make_renderer(
    model_name: str, tokenizer: Any, renderer_name: str | None = None
) -> renderers.Renderer:
    resolved = renderer_name or model_info.get_recommended_renderer_name(model_name)
    renderer = renderers.get_renderer(resolved, tokenizer)
    if resolved != "tml_v0":
        raise ValueError(f"Inkling-Small requires tml_v0, resolved {resolved!r}")
    return renderer


def generation_prompt(
    renderer: renderers.Renderer, example: Example, effort: float
) -> tinker.ModelInput:
    # TmlV0Renderer extends the base protocol with explicit effort conditioning.
    return renderer.build_generation_prompt(
        messages_for(example, include_answer=False), effort=effort
    )  # type: ignore[call-arg]


def training_datum(
    renderer: renderers.Renderer,
    tokenizer: Any,
    example: Example,
    effort: float,
    max_length: int,
) -> tuple[tinker.Datum, int]:
    model_input, weights = renderer.build_supervised_example(  # type: ignore[call-arg]
        messages_for(example, include_answer=True),
        train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE,
        effort=effort,
    )
    tokens = model_input.to_ints()
    weight_values = weights.tolist()
    if len(tokens) != len(weight_values):
        raise ValueError("Renderer returned token/weight length mismatch")
    if len(tokens) > max_length:
        tokens = tokens[:max_length]
        weight_values = weight_values[:max_length]
    weighted_tokens = [
        token for token, weight in zip(tokens, weight_values, strict=True) if weight > 0
    ]
    if not weighted_tokens:
        raise ValueError(f"Training label for {example.example_id} has no positive loss weight")
    weighted_text = tokenizer.decode(weighted_tokens)
    if example.label not in weighted_text:
        raise ValueError(
            f"Training label {example.label!r} was truncated or not loss-weighted; "
            f"weighted text={weighted_text!r}"
        )
    datum = datum_from_model_input_weights(
        model_input,
        weights,
        max_length=max_length,
        reduction="mean",
    )
    return datum, len(datum.model_input.to_ints())


def parse_sample(
    renderer: renderers.Renderer,
    tokenizer: Any,
    tokens: list[int],
    labels: tuple[str, ...],
) -> ParsedSample:
    message, termination = renderer.parse_response(tokens)
    raw_text = get_text_content(message)
    termination_value = getattr(termination, "value", str(termination))
    return ParsedSample(
        raw_text=raw_text,
        decoded_tokens=tokenizer.decode(tokens),
        strict_prediction=strict_label(raw_text, labels),
        tolerant_prediction=tolerant_label(raw_text, labels),
        termination=str(termination_value),
        generated_tokens=len(tokens),
    )
