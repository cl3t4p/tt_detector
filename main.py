import pandas as pd



from src.audio_utils import open_audio
from src.detector import compute_frame_energy, DecayAverageDetect



raw_sound_file_name = 'STE-011.wav'

file = f'data/raw_sounds/{raw_sound_file_name}'



waveform , sr = open_audio(file)

waveform = waveform.numpy()


decay_detector =  DecayAverageDetect()



detected_times = decay_detector.detect(waveform,sr)


frame_len = sr / 1000.0



for i in range(0,10):
    frame_time = detected_times[i] * frame_len / sr
    print(f"Detection {i} : at {frame_time}")


new_times = []
for i in range(0,len(detected_times)):
    new_times.append(detected_times[i] * frame_len / sr)

print(f"N of detections : {len(detected_times)}")


df = pd.read_csv('data/full.csv')


fulldf = df[df["original-file"] == raw_sound_file_name].reset_index(drop=True)


timestamps = df[['timestamp']].to_numpy()


