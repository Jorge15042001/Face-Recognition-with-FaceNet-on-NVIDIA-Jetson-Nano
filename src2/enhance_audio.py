from speechbrain.pretrained import SepformerSeparation as separator
import torchaudio
import sys
import os

from speechbrain.dataio.dataio import write_audio
#
audio_path = sys.argv[1]
output_audio_path = f"{'.'.join(sys.argv[1].split('.')[:-1])}_enhanced.wav"
#
#
model = separator.from_hparams(source="speechbrain/sepformer-whamr-enhancement",
                               savedir='pretrained_models/sepformer-whamr-enhancement')


def enhance_audio(audio_path, output_audio_path):
    enhanced_speech = model.separate_file(path=audio_path)
    write_audio(output_audio_path,
                enhanced_speech.detach().cpu().squeeze(), 8000)


if __name__ == "__main__":
    if os.path.isdir(audio_path):
        audio_files = [x for x in os.listdir(audio_path) if x.endswith(".wav")]

        for audio_file in audio_files:
            audio_output = f"{audio_path}/{'.'.join(audio_file.split('.')[:-1])}_enhanced.wav"
            input_audio_file = f"{audio_path}/{audio_file}"
            print(input_audio_file)
            print(audio_output)
            enhance_audio(input_audio_file, audio_output)
    else:
        enhance_audio(audio_path, output_audio_path)
