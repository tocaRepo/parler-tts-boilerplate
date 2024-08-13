import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
import soundfile as sf

print(torch.cuda.is_available())
print(torch.__version__)

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(device)
# if you have a good GPU use the large model
model = ParlerTTSForConditionalGeneration.from_pretrained(
    "parler-tts/parler-tts-mini-v1"
).to(device)
tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler-tts-mini-v1")
# Convert model to float16 if using GPU (optional)
if device == "cuda:0":
    # this will allow to run this with GPU with less than 4gb
    model.half()
prompt = "Hey, how are you doing today?"
description = "Jon's voice is monotone yet slightly fast in delivery, with a very close recording that almost has no background noise."

input_ids = tokenizer(description, return_tensors="pt").input_ids.to(device)
prompt_input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

generation = model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)

# if you are not using the model.half and you have a good GPU use the line below, you don't need to cast it back to float 32
# audio_arr = generation.cpu().numpy().squeeze()
# Convert to numpy and then to float32
audio_arr = generation.cpu().numpy().squeeze().astype("float32")
sf.write("parler_tts_out.wav", audio_arr, model.config.sampling_rate)
