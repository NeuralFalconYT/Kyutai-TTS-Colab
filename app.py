import numpy as np
import torch
from moshi.models.loaders import CheckpointInfo
from moshi.models.tts import DEFAULT_DSM_TTS_REPO, DEFAULT_DSM_TTS_VOICE_REPO, TTSModel

import re
import uuid
import os
import soundfile as sf

checkpoint_info = CheckpointInfo.from_hf_repo(DEFAULT_DSM_TTS_REPO)
tts_model = TTSModel.from_checkpoint_info(
    checkpoint_info, n_q=32, temp=0.6, device=torch.device("cuda"), dtype=torch.half
)


def tts_file_name(text, extension=".wav"):
    save_folder = "./kyutai_tts_result"
    os.makedirs(save_folder, exist_ok=True)

    # Clean and process the text
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower().strip().replace(" ", "_")

    if not text:
        text = "audio"

    truncated_text = text[:20]
    random_string = uuid.uuid4().hex[:8].upper()

    file_name = f"{save_folder}/{truncated_text}_{random_string}{extension}"
    return file_name

def convert_speed_to_padding_bonus(speed):
    # The speed doesn't change significantly. Officially, the maximum effective range is between -2 and +2.
    # When the speed is slowed down (positive padding), TTS generation time increases.
    # Conversely, when speed up (negative padding), the generation time decreases.

    """
    Converts playback speed (0.25x to 2x) to padding_bonus.
    - speed < 1.0 → slower → positive padding_bonus
    - speed > 1.0 → faster → negative padding_bonus
    positive mean slower 
    negative mean faster 
    """
    if speed == 1.0:
        return 1  # Neutral
    elif speed < 1.0:
        return int(round((1.0 - speed) * 10))  # e.g., 0.5x → +10
    else:
        return int(round((1.0 - speed) * 20))  # e.g., 1.5x → -20




def kyutai_tts_demo(text,voice,speed=1,use_ffmpeg=False):
  # print(voice)
  pcms = []
  if use_ffmpeg==True:
    padding_bonus=1
  else:
    padding_bonus=convert_speed_to_padding_bonus(speed)
  # print(padding_bonus)
  def _on_frame(frame):
        # print("Step", len(pcms), end="\r")
        if (frame != -1).all():
            pcm = tts_model.mimi.decode(frame[:, 1:, :]).cpu().numpy()
            pcms.append(np.clip(pcm[0, 0], -1, 1))

  # If you want to make a dialog, you can pass more than one turn [text_speaker_1, text_speaker_2, text_2_speaker_1, ...]
  entries = tts_model.prepare_script([text], padding_between=padding_bonus)
  voice_path = tts_model.get_voice_path(voice)
  # CFG coef goes here because the model was trained with CFG distillation,
  # so it's not _actually_ doing CFG at inference time.
  # Also, if you are generating a dialog, you should have two voices in the list.
  condition_attributes = tts_model.make_condition_attributes(
      [voice_path], cfg_coef=2.0
  )

  all_entries = [entries]
  all_condition_attributes = [condition_attributes]
  with tts_model.mimi.streaming(len(all_entries)):
      result = tts_model.generate(all_entries, all_condition_attributes, on_frame=_on_frame)
  audio = np.concatenate(pcms, axis=-1)

  save_path = tts_file_name(text)
  sf.write(save_path, audio, samplerate=tts_model.mimi.sample_rate)
  # print(f"\n Saved to: {save_path}")
  return save_path


from huggingface_hub import list_repo_files
def reorder_voice_list(voice_list):
    expresso = [v for v in voice_list if v.startswith("expresso")]
    unmute = [v for v in voice_list if v.startswith("unmute-prod-website")]
    vctk = [v for v in voice_list if v.startswith("vctk")]
    cml = [v for v in voice_list if v.startswith("cml-tts/fr")]
    others = [v for v in voice_list if v not in expresso + unmute + vctk + cml]
    return expresso + unmute + vctk + others + cml


def get_available_voices():
    repo_id = "kyutai/tts-voices"
    voice_extensions = (".wav", ".mp3")
    all_files = list_repo_files(repo_id)
    voice_files = [f for f in all_files if f.lower().endswith(voice_extensions)]
    # for path in voice_files:
    #     print(path)
    voice_files = reorder_voice_list(voice_files)
    return voice_files
def get_sample_audio(voice_path):
  return f"./tts-voices/{voice_path}"





import platform
import subprocess
import librosa
import soundfile as sf
import shutil

def is_ffmpeg_installed():
    # This part is useless, we are checking any protable ffmpeg available or not 
    ffmpeg_exe = "ffmpeg.exe" if platform.system() == "Windows" else "ffmpeg"
    try:
        subprocess.run([ffmpeg_exe, "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return True, ffmpeg_exe
    except Exception:
        print("⚠️ FFmpeg not found. Falling back to librosa for audio speedup.")
        return False, ffmpeg_exe

def atempo_chain(factor):
    if 0.5 <= factor <= 2.0:
        return f"atempo={factor:.3f}"
    parts = []
    while factor > 2.0:
        parts.append("atempo=2.0")
        factor /= 2.0
    while factor < 0.5:
        parts.append("atempo=0.5")
        factor *= 2.0
    parts.append(f"atempo={factor:.3f}")
    return ",".join(parts)

def speedup_audio_librosa(input_file, output_file, speedup_factor):
    try:
        y, sr = librosa.load(input_file, sr=None)
        y_stretched = librosa.effects.time_stretch(y, rate=speedup_factor)
        sf.write(output_file, y_stretched, sr)
    except Exception as e:
        # print(f"⚠️ Librosa speedup failed: {e}")
        shutil.copy(input_file, output_file)

def change_speed(input_file, speedup_factor):
    use_ffmpeg, ffmpeg_path = is_ffmpeg_installed()
    base, ext = os.path.splitext(input_file)
    output_file = f"{base}_speed_changed{ext}"

    if use_ffmpeg:
        try:
            subprocess.run(
                [ffmpeg_path, "-i", input_file, "-filter:a", atempo_chain(speedup_factor), output_file, "-y"],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        except Exception as e:
            # print(f"⚠️ FFmpeg speedup error: {e}")
            speedup_audio_librosa(input_file, output_file, speedup_factor)
    else:
        speedup_audio_librosa(input_file, output_file, speedup_factor)

    return output_file


import wave
from pydub import AudioSegment
from pydub.silence import split_on_silence

def remove_silence_function(file_path,minimum_silence=50):
    # Extract file name and format from the provided path
    output_path = file_path.replace(".wav", "_no_silence.wav")
    audio_format = "wav"
    # Reading and splitting the audio file into chunks
    sound = AudioSegment.from_file(file_path, format=audio_format)
    audio_chunks = split_on_silence(sound,
                                    min_silence_len=100,
                                    silence_thresh=-45,
                                    keep_silence=minimum_silence)
    # Putting the file back together
    combined = AudioSegment.empty()
    for chunk in audio_chunks:
        combined += chunk
    combined.export(output_path, format=audio_format)
    return output_path


import gradio as gr
import random
# 1. Fetch available voices from HF
voice_list = get_available_voices()

def get_random_voice(voice_list):
    return random.choice(voice_list)

# Button: When clicked, update dropdown and sample audio
def pick_random_voice_fn():
    voice = get_random_voice(voice_list)
    sample = get_sample_audio(voice)
    return gr.update(value=voice), gr.update(value=sample)



def generate_tts(text, voice_path, speed=1.0, use_ffmpeg=False, remove_silence=False):
    # Step 1: Generate raw TTS audio
    raw_audio_path = kyutai_tts_demo(text, voice_path, speed, use_ffmpeg)

    # Step 2: Optional speed adjustment
    processed_path = change_speed(raw_audio_path, speed) if use_ffmpeg else raw_audio_path

    # Step 3: Optional silence removal
    final_path = remove_silence_function(processed_path, minimum_silence=50) if remove_silence else processed_path
    return final_path, final_path
def ui():
  with gr.Blocks() as demo:
      # gr.Markdown("## 🎤 Kyutai TTS Voice Cloning Demo")
      gr.Markdown("<center><h1 style='font-size: 40px;'>Kyutai TTS</h1></center>")  
      gr.Markdown("🌐 [Official GitHub: kyutai-labs](https://github.com/kyutai-labs/delayed-streams-modeling) — Real-time TTS with optimized, production-ready implementation. (This Gradio app uses a simplified wrapper for easy experimentation.)")
      with gr.Row():
          with gr.Column(scale=2):
              text_input = gr.Textbox(
                  label="Input Text",
                  value="Life is a question asked by the stars and answered by our footsteps.\nEach breath we take writes a chapter in a book with no final page.\nAnd in the end, it's not about what we’ve built, but what we’ve become.",
                  placeholder="Type something...",
                  max_lines=5
              )


              voice_dropdown = gr.Dropdown(choices=voice_list, label="Select Voice", value=voice_list[0])
              generate_button = gr.Button("🚀 Generate Audio")
              with gr.Accordion('⚙️ Audio Settings', open=False):
                  speed = gr.Slider(minimum=0.5, maximum=2, value=1, step=0.1, label='⚡️Speed')
                  external_speed_change = gr.Checkbox(value=False, label='🤖 Use FFmpeg/Librosa for speed change')
                  remove_silence = gr.Checkbox(value=False, label='✂️ Remove Silence From TTS')

          with gr.Column(scale=1):
              sample_audio = gr.Audio(
                  label="Selected Voice Playback",
                  type="filepath",
                  value=get_sample_audio(voice_list[0]),
                  interactive=False,
                  autoplay=True
              )
              pick_random_voice = gr.Button("🎲 Random Voice")

          with gr.Column(scale=2):
              generated_audio = gr.Audio(label="🔊 Generated TTS Output", type="filepath")
              # Faster download link
              audio_file = gr.File(label='📥 Download Audio')

      # When voice is changed from dropdown
      voice_dropdown.change(fn=get_sample_audio, inputs=voice_dropdown, outputs=sample_audio)

      # When random button clicked
      pick_random_voice.click(fn=pick_random_voice_fn, inputs=[], outputs=[voice_dropdown, sample_audio])

      # When generate button clicked
      generate_button.click(
          fn=generate_tts,
          inputs=[text_input, voice_dropdown, speed, external_speed_change,remove_silence],
          outputs=[generated_audio,audio_file]
      )
  return demo

import click
@click.command()
@click.option("--debug", is_flag=True, default=False, help="Enable debug mode.")
@click.option("--share", is_flag=True, default=False, help="Enable sharing of the interface.")
def main(debug, share):
# def main(debug=True, share=True):
    demo = ui()
    demo.queue().launch(debug=debug, share=share)

    #Run on local network
    # laptop_ip="192.168.0.30"
    # port=8080
    # demo.queue().launch(debug=debug, share=share,server_name=laptop_ip,server_port=port)

if __name__ == "__main__":
    main()    
