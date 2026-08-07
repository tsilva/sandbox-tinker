from tinker_cookbook.tokenizer_utils import get_tokenizer

from banking77_experiment.data import Example
from banking77_experiment.rendering import (
    generation_prompt,
    make_renderer,
    system_prompt,
    training_datum,
)


def test_real_inkling_renderer_contract():
    tokenizer = get_tokenizer("thinkingmachines/Inkling-Small")
    renderer = make_renderer("thinkingmachines/Inkling-Small", tokenizer)
    example = Example(
        example_id="contract",
        source_split="train",
        source_index=0,
        text="My card was charged twice",
        label_id=0,
        label="card_payment_fee_charged",
    )
    labels = ("card_payment_fee_charged", "cash_withdrawal")
    compact_prompt = "Classify with the exact Banking77 intent label. Output only the label."
    prompt = generation_prompt(renderer, example, labels, "compact", compact_prompt, effort=0.0)
    datum, token_count = training_datum(
        renderer,
        tokenizer,
        example,
        labels,
        "compact",
        compact_prompt,
        effort=0.0,
        max_length=2048,
    )
    assert prompt.to_ints()
    assert token_count > len(prompt.to_ints())
    assert len(datum.model_input.to_ints()) == token_count
    assert sum(datum.loss_fn_inputs["weights"].tolist()) > 0


def test_prompt_variants_make_the_schema_visible_only_in_full_prompt():
    labels = ("card_payment_fee_charged", "cash_withdrawal")
    compact_text = "Classify with the exact Banking77 intent label. Output only the label."
    compact = system_prompt(labels, "compact", compact_text)
    full = system_prompt(labels, "full_taxonomy", compact_text)
    assert "card_payment_fee_charged" not in compact
    assert "cash_withdrawal" not in compact
    assert "card_payment_fee_charged" in full
    assert "cash_withdrawal" in full
