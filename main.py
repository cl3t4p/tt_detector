import pandas as pd



import src.plot_helper as plt_help
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


print(f"N of detections : {len(detected_times)}")


df = pd.read_csv('data/full.csv')


fulldf = df[df["original-file"] == raw_sound_file_name].reset_index(drop=True)

reference_times = fulldf["timestamp"].to_numpy()




comparison_df, false_positives = plt_help.compare_timestamps(
    reference_times,
    detected_times,
    tolerance=0.10,   # 30 ms matching tolerance
)


print(comparison_df)

print("\nMiss count:", (~comparison_df["matched"]).sum())
print("Matched count:", comparison_df["matched"].sum())
print("False positives:", false_positives)

# 4) Plot over the file timeline
audio_duration = reference_times.max() + 0.2
plt_help.plot_misses_and_shift_error(
    reference_times,
    comparison_df,
    audio_duration=audio_duration,
    title="STE-011.wav: misses and shift error"
)
