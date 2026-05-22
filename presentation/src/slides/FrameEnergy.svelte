<h2>Stage 2 — Frame Energy Envelope</h2>
<!--<p>The filtered waveform is split into <strong style="color:var(--purple)">1 ms non-overlapping frames</strong>. The mean absolute amplitude of each frame produces a compact energy envelope. </p>-->
<img src="imgs/02_energy_envelope.png" class="graph-img" alt="Energy envelope" />

<div class="formula">e[k] = (1 / N) × Σ |x[n]|</div>

<div class="content-row">
  <div class="card purple-border">
    <h3>Why Frame Energy?</h3>
    <ul>
      <li>Ball impacts produce a sudden spike  — easy to detect against a quiet background</li>
      <li>Working at frame level is fast and memory-efficient, making real-time processing practical</li>
    </ul>
  </div>
  <div class="card blue-border">
    <h3>Frame Size = 1 ms</h3>
    <ul>
      <li>At SR = 44 100 Hz → 44 samples per frame</li>
      <li>Fine enough to capture sharp peaks</li>
    </ul>
  </div>
</div>
