# Example usage
from parlerTTS import ParlerTTS
from utils import combine_audio_files


tts = ParlerTTS()

prompt = """Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. 

Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea 

commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. 

Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit 

anim id est laborum."""
description = "Jon's voice is monotone yet slightly fast in delivery, with a very close recording that almost has no background noise."
output_path_prefix = "parler_tts_out"
combined_output_path = "combined_parler_tts_out.wav"

split_input=True
# Generate individual audio files
tts.generate_speech(prompt, description, output_path_prefix,split_input)

if(split_input):
    # Combine the generated audio files into one large file
    combine_audio_files(output_path_prefix, combined_output_path)
