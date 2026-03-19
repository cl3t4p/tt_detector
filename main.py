import pandas as pd


import src.audio_utils as audio_utils
from src.audio_utils import open_audio, pad_trunc
from src.detectors import DecayAverageDetect, PowerEnergyCalculator,RMSEnergyCalculator, TeagerEnergyCalculator,SimpleEnergyCalculator,BaseEnergyCalculator



raw_sound_file_name = '01.wav'

file = f'data/raw_sounds/{raw_sound_file_name}'
waveform , sr = open_audio(file)
waveform = waveform.numpy()



energy_calculators = [PowerEnergyCalculator(),RMSEnergyCalculator(),TeagerEnergyCalculator(),SimpleEnergyCalculator()]

def calculate_peaks(energy_calculator : BaseEnergyCalculator):

    print(f"\n\nENERGY : {energy_calculator.__class__.__name__}")
    decay_detector =  DecayAverageDetect(energy_calculator=energy_calculator,return_indexes=False)
    detected_times = decay_detector.detect(waveform,sr)


    for i,det in enumerate(detected_times):
        print(f"Detection {i} : at {det:.2f}")

    print(f"N of detections : {len(detected_times)}\n\n")




for energy in energy_calculators:
    calculate_peaks(energy)
