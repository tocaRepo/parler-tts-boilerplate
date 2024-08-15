import os
import soundfile as sf
import numpy as np

def combine_audio_files(output_path_prefix, combined_output_path):
        """
        Combines multiple .wav files that start with the given prefix into a single .wav file.
        """
        # Handle the case where there is no directory in the prefix
        directory = os.path.dirname(output_path_prefix)
        if directory == "":
            directory = os.getcwd()  # Use current working directory if no directory is specified
        prefix = os.path.basename(output_path_prefix)
        
        # Get all files starting with the specified prefix
        files_to_combine = sorted([f for f in os.listdir(directory) if f.startswith(prefix) and f.endswith(".wav")])
        
        # Initialize an empty list to hold the audio data
        combined_audio = []
        
        for file_name in files_to_combine:
            full_path = os.path.join(directory, file_name)
            # Read each file
            audio_data, sample_rate = sf.read(full_path)
            
            # Append the data to the combined list
            combined_audio.append(audio_data)
        
        # Concatenate all the audio data into one array
        combined_audio = np.concatenate(combined_audio)
        
        # Write the combined audio data into a single .wav file
        sf.write(combined_output_path, combined_audio, sample_rate)
        print(f"Combined audio saved to {combined_output_path}")