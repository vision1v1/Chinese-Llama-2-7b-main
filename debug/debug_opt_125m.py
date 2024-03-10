import os
from transformers import pipeline, set_seed
from transformers import AutoModel, AutoConfig, AutoTokenizer, AutoModelForCausalLM
from transformers.models.opt.configuration_opt import OPTConfig
from transformers.models.opt.modeling_opt import OPTModel, OPTForCausalLM
from transformers.models.gpt2.tokenization_gpt2 import GPT2Tokenizer
from transformers.pipelines.text_generation import TextGenerationPipeline

my_data_dir = os.getenv("my_data_dir")
opt_125m_path = os.path.join(my_data_dir, "pretrained", "facebook", "opt-125m")


def debug_model_card():
    set_seed(32)
    generator: TextGenerationPipeline = pipeline(
        'text-generation', model=opt_125m_path, do_sample=True)
    output = generator("What are we having for dinner?")
    print("output=", output, sep='\n', end='\n\n')


def preprocess(tokenizer, prompt_text, prefix="</s>"):
    inputs = tokenizer(prefix + prompt_text,
                       padding=False,
                       add_special_tokens=False,
                       return_tensors='pt')
    return inputs


def debug_model():
    set_seed(32)
    config = AutoConfig.from_pretrained(opt_125m_path)
    tokenizer = AutoTokenizer.from_pretrained(opt_125m_path)
    # model = AutoModel.from_pretrained(opt_125m_path)
    model = AutoModelForCausalLM.from_pretrained(opt_125m_path)

    prompt_text = "What are we having for dinner?"
    prefix = "</s>"

    inputs = tokenizer(prefix + prompt_text, return_tensors='pt')
    print(inputs.keys())
    # output = model(**inputs)
    generate_kwargs = {'do_sample': True, 'max_length': 21}
    generated_sequence = model.generate(**inputs, **generate_kwargs) # 这里包括 解码逻辑。
    generated_sequence = generated_sequence.tolist()

    text = tokenizer.decode(generated_sequence[0],
                            skip_special_tokens=True,
                            clean_up_tokenization_spaces=True)

    print(text)
    ...


if __name__ == "__main__":

    # debug_model_card()
    debug_model()
    ...
