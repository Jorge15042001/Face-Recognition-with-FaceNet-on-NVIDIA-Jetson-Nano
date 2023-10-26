from speechbrain.pretrained import SpeakerRecognition
import threading
import numpy as np
import sys
import glob
import time


import torch
#  import torchaudio
#  from deepspeaker_model import DeepSpeakerModel  # Example model
#
#  # Load a pre-trained DeepSpeaker model (or initialize your own)
#  model = DeepSpeakerModel()
#
#  # Load and preprocess an audio segment
#  audio, sample_rate = torchaudio.load("speaker_audio.wav")
#  # Apply necessary preprocessing steps (e.g., feature extraction)
#
#  # Obtain the speaker embedding
#  with torch.no_grad():
#      embedding = model.forward(audio)  # This might be the shape (1, 512)
#
class AudioAnalizer:
    def __init__(self, audio_files, name):
        self.verificator = SpeakerRecognition.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="pretrained_models/spkrec-ecapa-voxceleb")
        self.audio_files = audio_files
        self.name = name
        self.thread = None

    def run(self):
        embds = []
        for audio_file in self.audio_files:
            start = time.time()
            print(audio_file)
            audio = self.verificator.load_audio(audio_file).unsqueeze(0)
            emb = self.verificator.encode_batch(audio, None, False)
            emb = emb[0][0]
            elapsed_time = time.time() - start
            print(f"elapse: {elapsed_time}")
            embds.append(np.array(emb))
        embds = np.array(embds)
        print(embds.shape)
        np.savetxt(f"./embedings/{self.name}_voice.emb", np.median(embds,axis=0))
        np.savetxt(f"./embedings/{self.name}_voice.embs", embds)

    def start(self):
        self.thread = threading.Thread(target=self.run)
        self.thread.start()


if __name__ == "__main__":
    folder_name = sys.argv[1]
    name = sys.argv[2]
    files = glob.glob(f"{folder_name}/{name}*")
    audio_analizer = AudioAnalizer(files, name)
    audio_analizer.start()
    audio_analizer.thread.join()
