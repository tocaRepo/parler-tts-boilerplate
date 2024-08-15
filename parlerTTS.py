import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
import soundfile as sf
import re
import os
import numpy as np

from utils import combine_audio_files

class ParlerTTS:
    def __init__(self, model_name="parler-tts/parler-tts-mini-v1", device=None):
        self.device = device if device else ("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # Load model and tokenizer
        self.model = ParlerTTSForConditionalGeneration.from_pretrained(model_name).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Convert model to float16 if using GPU (optional)
        if self.device == "cuda:0":
            self.model.half()
    
    def generate_speech(self, prompt, description, output_path_prefix,splitInput=False):
        # Tokenize the description
        input_ids = self.tokenizer(description, return_tensors="pt").input_ids.to(self.device)
        
        if(splitInput):
            # Break the prompt into smaller segments
            chunks = self.split_text(prompt)
            
            for i, chunk in enumerate(chunks):
                prompt_input_ids = self.tokenizer(chunk, return_tensors="pt").input_ids.to(self.device)
                
                # Generate audio for each chunk
                generation = self.model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)
                
                # Convert to numpy and save as .wav
                audio_arr = generation.cpu().numpy().squeeze().astype("float32")
                output_path = f"{output_path_prefix}_{i + 1}.wav"
                sf.write(output_path, audio_arr, self.model.config.sampling_rate)
            else:
                 # Generate audio for each chunk
                generation = self.model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)
                
                # Convert to numpy and save as .wav
                audio_arr = generation.cpu().numpy().squeeze().astype("float32") 
                output_path = f"{output_path_prefix}_{i + 1}.wav"
                sf.write(output_path, audio_arr, self.model.config.sampling_rate)   
            print(f"Audio generated and saved to {output_path}")
    
    def split_text(self, text):
        """
        Splits the text into smaller chunks based on new lines or periods.
        It ensures that each chunk is a complete sentence or segment.
        """
        # Split by new lines first
        chunks = text.split('\n')
        
        # Further split each chunk by periods if it's too large
        split_chunks = []
        for chunk in chunks:
            sentences = re.split(r'(?<=[.!?])\s+', chunk.strip())
            split_chunks.extend(sentences)
        
        # Filter out empty chunks
        return [chunk for chunk in split_chunks if chunk]

    

