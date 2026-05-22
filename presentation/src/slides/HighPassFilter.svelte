<h2>Stage 1 — High-Pass Butterworth Filter</h2>
<img src="imgs/01_raw_vs_filtered.png" class="graph-img" alt="Raw vs filtered" />
<div class="content-row">
  <div class="card green-border">
    <h3>Why?</h3>
    <p style="text-align:left;margin-top:8px;">
  A 5th-order Butterworth high-pass filter at <strong style="color:var(--green)">10 kHz</strong> 
  filters out noise like footsteps, voices, and room ambience. Table tennis ball bounces
  consistently exhibit energy around <strong style="color:var(--blue)">11 kHz</strong>, so this cutoff preserves the impact signature while rejecting low-frequency interference.
    </p>
  </div>
</div>
