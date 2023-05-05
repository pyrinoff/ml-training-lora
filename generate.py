import os
import fire
from app.application import Application


def main(
        input_file: str = "",
        base_model: str = "yahma/llama-7b-hf",
        lora_weights: str = "tloen/alpaca-lora-7b",
        load_8bit: bool = True,
        gradio_server_name: str = "0.0.0.0",  # Allows to listen on all interfaces by providing '0.0.0.0'
        share_gradio: bool = False,
        temperature=0.1,
        top_p=0.75,
        top_k=40,
        num_beams=4,
        max_new_tokens=128,
        prompt_template: str = "",  # The prompt template to use, will default to alpaca.
        trim_special_tokens: bool = True,  # The prompt template to use, will default to alpaca.
):
    # input variables validation
    base_model = base_model or os.environ.get("BASE_MODEL", "")
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"

    application = Application(base_model, lora_weights, load_8bit,
                              temperature, top_p, top_k, num_beams, max_new_tokens, prompt_template,
                              trim_special_tokens
                              )

    print(
        f"----------------------\n"
        f"GENERATION with params\n"
        f"input_file: {input_file}\n"
        f"base_model: {base_model}\n"
        f"lora_weights: {lora_weights}\n"
        f"load_8bit: {load_8bit}\n"
        f"gradio_server_name: {gradio_server_name}\n"
        f"share_gradio: {share_gradio}\n"
        f"temperature: {temperature}\n"
        f"top_p: {top_p}\n"
        f"top_k: {top_k}\n"
        f"num_beams: {num_beams}\n"
        f"max_new_tokens: {max_new_tokens}\n"
        f"prompt_template: {prompt_template}\n"
        f"trim_special_token: {trim_special_tokens}\n"
    )

    # main logic
    if input_file != "":
        application.generate_from_file(input_file)
    else:
        application.run_gradio(gradio_server_name, share_gradio)
    exit(0)


if __name__ == "__main__":
    fire.Fire(main)
