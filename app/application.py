"""
A dedicated helper to manage templates and prompt building.
"""
import json
import os.path as osp
import os
import sys
import gradio as gr
import pandas
import torch
import transformers
from peft import PeftModel
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer

from app.callbacks import Iteratorize, Stream
from app.prompter import Prompter


assert (
    "LlamaTokenizer" in transformers._import_structure["models.llama"]
), "LLaMA is now in HuggingFace's main branch.\nPlease reinstall it: pip uninstall transformers && pip install git+https://github.com/huggingface/transformers.git"


class Application:
    __slots__ = ("device", "tokenizer", "model", "prompter",
                 "settings_base_model", "settings_lora_weights", "settings_load_8bit",
                 "settings_temperature", "settings_top_p", "settings_top_k",
                 "settings_num_beams", "settings_max_new_tokens", "settings_prompt_template"
                 , "settings_trim_special_tokens"
                 )

    def __init__(self,
                 base_model: str,
                 lora_weights: str = None,
                 load_8bit: bool = True,
                 temperature=0.1,
                 top_p=0.75,
                 top_k=40,
                 num_beams=4,
                 max_new_tokens=128,
                 prompt_template: str = "",
                 trim_special_tokens: bool = True
                 ):
        # save settings
        self.settings_base_model = base_model
        self.settings_lora_weights = lora_weights
        self.settings_load_8bit = load_8bit
        self.settings_temperature = temperature
        self.settings_top_p = top_p
        self.settings_top_k = top_k
        self.settings_num_beams = num_beams
        self.settings_max_new_tokens = max_new_tokens
        self.settings_prompt_template = prompt_template
        self.settings_trim_special_tokens = trim_special_tokens

        # initialization
        self.device = self.setup_device()
        self.tokenizer = self.setup_tokenizer(base_model)
        self.model = self.setup_model(base_model, lora_weights, load_8bit)
        self.prompter = self.setup_prompter()
        return

    def setup_device(self) -> str:
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
        try:
            if torch.backends.mps.is_available():
                device = "mps"
        except:  # noqa: E722
            pass
        return device

    def setup_tokenizer(
            self,
            base_model: str,
    ) -> LlamaTokenizer:
        tokenizer = LlamaTokenizer.from_pretrained(base_model)
        return tokenizer

    def setup_prompter(
            self,
            prompt_template: str = "",  # The prompt template to use, will default to alpaca.
    ) -> Prompter:
        prompter = Prompter(prompt_template)
        return prompter

    def setup_model(
            self,
            base_model: str,
            lora_weights: str,
            load_8bit: bool,
    ):
        if self.device == "cuda":
            model = LlamaForCausalLM.from_pretrained(
                base_model,
                load_in_8bit=load_8bit,
                torch_dtype=torch.float16,
                device_map="auto",
            )
            if self.settings_lora_weights is not None:
                model = PeftModel.from_pretrained(
                    model,
                    lora_weights,
                    torch_dtype=torch.float16,
                )
        elif self.device == "mps":
            model = LlamaForCausalLM.from_pretrained(
                base_model,
                device_map={"": self.device},
                torch_dtype=torch.float16,
            )
            if self.settings_lora_weights is not None:
                model = PeftModel.from_pretrained(
                    model,
                    lora_weights,
                    device_map={"": self.device},
                    torch_dtype=torch.float16,
                )
        else:
            model = LlamaForCausalLM.from_pretrained(
                base_model, device_map={"": self.device}, low_cpu_mem_usage=True
            )
            if self.settings_lora_weights is not None:
                model = PeftModel.from_pretrained(
                    model,
                    lora_weights,
                    device_map={"": self.device},
                )

        # unwind broken decapoda-research
        if self.settings_base_model.startswith("decapoda-research"):
            model.config.pad_token_id = self.tokenizer.pad_token_id = 0  # unk
            model.config.bos_token_id = 1
            model.config.eos_token_id = 2

        if not load_8bit:
            model.half()  # seems to fix bugs for some users.

        model.eval()
        if torch.__version__ >= "2" and sys.platform != "win32":
            model = torch.compile(model)
        return model

    def evaluate(
            self,
            instruction,
            input=None,
            temperature=0.1,
            top_p=0.75,
            top_k=40,
            num_beams=4,
            max_new_tokens=128,
            stream_output=False,
            **kwargs,
    ):
        prompt = self.prompter.generate_prompt(instruction, input)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            **kwargs,
        )

        generate_params = {
            "input_ids": input_ids,
            "generation_config": generation_config,
            "return_dict_in_generate": True,
            "output_scores": True,
            "max_new_tokens": max_new_tokens,
        }

        if stream_output:
            with self.generate_with_streaming(**generate_params) as generator:
                for output in generator:
                    # new_tokens = len(output) - len(input_ids[0])
                    decoded_output = self.tokenizer.decode(output)

                    if output[-1] in [self.tokenizer.eos_token_id]:
                        break

                    yield self.prompter.get_response(decoded_output)
            return  # early return for stream_output

        # Without streaming
        with torch.no_grad():
            generation_output = self.model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
            )
        logit_sequence = generation_output.sequences[0]
        output = self.tokenizer.decode(logit_sequence, self.settings_trim_special_tokens)
        # if output[-1] in [self.tokenizer.eos_token_id]: output = output[:-1]
        yield self.prompter.get_response(output)

    # Stream the reply 1 token at a time.
    # This is based on the trick of using 'stopping_criteria' to create an iterator,
    # from https://github.com/oobabooga/text-generation-webui/blob/ad37f396fc8bcbab90e11ecf17c56c97bfbd4a9c/modules/text_generation.py#L216-L243.
    def generate_with_callback(self, callback=None, **kwargs):
        kwargs.setdefault(
            "stopping_criteria", transformers.StoppingCriteriaList()
        )
        kwargs["stopping_criteria"].append(
            Stream(callback_func=callback)
        )
        with torch.no_grad():
            self.model.generate(**kwargs)

    def generate_with_streaming(self, **kwargs):
        return Iteratorize(
            self.generate_with_callback, kwargs, callback=None
        )

    def run_gradio(self,
                   server_name: str = "0.0.0.0",  # Allows to listen on all interfaces by providing '0.0.0.0'
                   share_gradio: bool = False,
                   ):
        gr.Interface(
            fn=self.evaluate,
            inputs=[
                gr.components.Textbox(
                    lines=2,
                    label="Instruction",
                    placeholder="Tell me about alpacas.",
                ),
                gr.components.Textbox(lines=2, label="Input", placeholder="none"),
                gr.components.Slider(
                    minimum=0, maximum=1, value=0.1, label="Temperature"
                ),
                gr.components.Slider(
                    minimum=0, maximum=1, value=0.75, label="Top p"
                ),
                gr.components.Slider(
                    minimum=0, maximum=100, step=1, value=40, label="Top k"
                ),
                gr.components.Slider(
                    minimum=1, maximum=4, step=1, value=4, label="Beams"
                ),
                gr.components.Slider(
                    minimum=1, maximum=2000, step=1, value=128, label="Max tokens"
                ),
                gr.components.Checkbox(label="Stream output"),
            ],
            outputs=[
                gr.inputs.Textbox(
                    lines=5,
                    label="Output",
                )
            ],
            title="🦙🌲 LLAMA",
            description="Model: " + self.settings_base_model + ", lora weights: " + self.settings_lora_weights,
            # noqa: E501
        ).queue().launch(server_name=server_name, share=share_gradio)
        return

    def generate_from_file(self,
                           input_file: str
                           ):
        with open(input_file, "rb") as json_file:
            json_data = json.loads(json_file.read())
        df = pandas.DataFrame(json_data)

        output_list = []
        for index, row in df.iterrows():
            print(row)
            output_string = next(self.evaluate(row['instruction'], row['input'],
                                               self.settings_temperature, self.settings_top_p, self.settings_top_k,
                                               self.settings_num_beams, self.settings_max_new_tokens
                                               ))
            print("OUTPUT STRING: " + output_string)
            output_list.append(output_string)

        df["output"] = pandas.Series(output_list)

        translated_dict = df.to_dict('records')

        output_file = input_file.replace('.json', '') \
                      + '_output_' \
                      + self.settings_base_model.replace('/', '_') \
                      + '--' \
                      + self.settings_lora_weights.replace('/', '_').replace('.', '') \
                      + ".json"

        with open(output_file, 'w') as file:
            json.dump(translated_dict, file)
        return
