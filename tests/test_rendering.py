from tinker_cookbook.tokenizer_utils import get_tokenizer

from banking77_experiment.data import Example
from banking77_experiment.rendering import generation_prompt, make_renderer, training_datum


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
        system_prompt=(
            "Classify the banking request. Return exactly one label: "
            "card_payment_fee_charged or cash_withdrawal."
        ),
    )
    prompt = generation_prompt(renderer, example, effort=0.0)
    datum, token_count = training_datum(
        renderer,
        tokenizer,
        example,
        effort=0.0,
        max_length=2048,
    )
    assert prompt.to_ints()
    assert token_count > len(prompt.to_ints())
    assert len(datum.model_input.to_ints()) == token_count
    assert sum(datum.loss_fn_inputs["weights"].tolist()) > 0
