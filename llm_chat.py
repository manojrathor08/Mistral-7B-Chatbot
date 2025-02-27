import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
torch.cuda.empty_cache()

torch.backends.cudnn.benchmark = True  # Optimizes GPU performance
# Load Model and Tokenizer
model_name = "mistralai/Mistral-7B-Instruct-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Enable 8-bit quantization + float16 activations
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,  # Quantizes weights to int8
    llm_int8_threshold=6.0,  # Helps balance accuracy
    llm_int8_skip_modules=["lm_head"],  # Keep final layers in float16
)


model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,  # 8-bit quantization
    torch_dtype=torch.float16,
    device_map="auto"
)
# Compile model for speed optimization
model = torch.compile(model)  # Speeds up inference
# Function to handle multi-turn conversation
def generate_response(prompt_txt):
    # Format the prompt with instruction tokens
    
    prompt = f"<s>[INST] {prompt_txt} [/INST]"

    # Tokenize and generate response
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=128, temperature=0.5, use_cache=True)

    # Decode the generated tokens
    response = tokenizer.decode(output[0], skip_special_tokens=True)

    # Extract the assistant's reply
    response = response.split('[/INST]')[-1].strip()

    return response

