<h2>Detection Pipeline</h2>
<img src="imgs/00_pipeline_diagram.png" class="graph-img" alt="Pipeline diagram" />
<div class="content-row">
  <div class="card blue-border">
    <h3>Stages 1–4 (Classic Onset Detection)</h3>
    <ul>
      <li>High-pass filter isolates ball-impact frequencies (&gt;10 kHz)</li>
      <li>Frame energy compresses raw samples to facilitate processing</li>
      <li>EMA tracks the running background energy level</li>
      <li>Threshold crossing flags candidate detections</li>
    </ul>
  </div>
  <div class="card orange-border">
    <h3>Stage 5 (Pre-Peak — Novel Addition)</h3>
    <ul>
      <li>Measures <em>sharpness</em> of the energy rise</li>
      <li>Computes: current frame ÷ mean of preceding window</li>
      <li>Rejects "slow" rises that are not physical ball bounces</li>
      <li>Dramatically reduces false positives with zero processing cost</li>
    </ul>
  </div>
</div>
