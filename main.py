import pandas as pd



from src.audio_utils import open_audio
from src.detectors import DecayAverageDetect, PowerEnergyCalculator,RMSEnergyCalculator, TeagerEnergyCalculator,SimpleEnergyCalculator,BaseEnergyCalculator



raw_sound_file_name = '01.wav'

file = f'data/raw_sounds/{raw_sound_file_name}'
waveform , sr = open_audio(file)
waveform = waveform.numpy()


energy_calculators = [PowerEnergyCalculator(),RMSEnergyCalculator(),TeagerEnergyCalculator(),SimpleEnergyCalculator()]

def calculate_peaks(energy_calculator : BaseEnergyCalculator):

    print(f"\n\nENERGY : {energy_calculator.__class__.__name__}")
    decay_detector =  DecayAverageDetect(energy_calculator=energy_calculator)
    detected_times = decay_detector.detect(waveform,sr)

    frame_len = sr / 1000.0

    for i in range(0,len(detected_times)):
        frame_time = detected_times[i] * frame_len / sr
        print(f"Detection {i} : at {frame_time}")

    print(f"N of detections : {len(detected_times)}\n\n")




for energy in energy_calculators:
    calculate_peaks(energy)
