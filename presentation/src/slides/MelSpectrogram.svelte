<h2>Bounce Representation — Mel Spectrogram</h2>
<p>Each detected bounce is extracted as a <strong style="color:var(--orange)">15 ms clip</strong> and converted to a <strong style="color:var(--blue)">64-bin Mel spectrogram</strong> — the single input to the CNN.</p>
<img src="imgs/14_mel_spectrogram.png" class="graph-img" alt="Mel spectrograms per spin type" />
<!--<div class="formula">
  MelSpec(n, m) = 20 · log₁₀( |STFT · MelFilterBank| + ε ) &nbsp; — &nbsp; top_db = 80
</div> -->
<div class="content-row">
  <div class="card orange-border">
    <h3>Extraction Parameters</h3>
    <ul>
      <li>Window length: <strong>256 samples</strong> (~5.8 ms at 44.1 kHz)</li>
      <li>Hop length: <strong>64 samples</strong> (~1.4 ms), yields ~10 time frames</li>
      <li><strong>64 Mel bins</strong> from 0 to 22 kHz</li>
    </ul>
  </div>
  <div class="card blue-border">
    <h3>Why Mel Scale?</h3>
    <ul>
      <li>Fixed-size <code>64 × 10</code> tensor regardless of clip duration — compatible with CNN</li>
      <li>Amplitude→dB mapping highlights low-energy transient differences</li>
    </ul>
  </div>
</div>
