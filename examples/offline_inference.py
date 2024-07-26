from vllm import LLM, SamplingParams
import torch
from torch._inductor import config

torch._inductor.config.cpp_wrapper = True
torch._inductor.config.profiler_mark_wrapper_call = True
torch._inductor.config.cpp.enable_kernel_profile = True
torch._inductor.config.freezing = True

# Sample prompts.
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

# Create an LLM.
llm = LLM(model="facebook/opt-125m")
# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.

with torch.no_grad():

    llm.llm_engine.model_executor.driver_worker.model_runner.model = torch.compile(
        llm.llm_engine.model_executor.driver_worker.model_runner.model,
        dynamic=True,
    )

    # Warp up run
    outputs = llm.generate(prompts, sampling_params)

    with torch.autograd.profiler.profile(True) as prof:
        outputs = llm.generate(prompts, sampling_params)
    print(prof.key_averages().table(sort_by="self_cpu_time_total"))
    # Print the outputs.
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
