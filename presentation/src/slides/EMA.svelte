<h2>Stages 3 — Exponential Moving Average &amp; Threshold</h2>
<!--<p>The EMA tracks the <em>background</em> energy level. A threshold set at <strong style="color:var(--red)">3×EMA</strong> flags candidate detections when the signal abruptly exceeds normal levels.</p> -->
<img src="imgs/03_ema_threshold.png" class="graph-img" alt="EMA and threshold" />

<div class="formula">E<sub>k+1</sub> = γ · E<sub>k</sub> + (1 − γ) · e<sub>k</sub> &nbsp;&nbsp; γ = 0.9</div>

<div class="content-row">
  <div class="card orange-border">
    <h3>Why Exponential Moving Average?</h3>
    <ul>
      <li>Adapts to slowly changing background conditions (e.g. recording getting louder)</li>
      <li>Single parameter γ controls memory length — γ=0.9 ≈ 10-frame effective window</li>
    </ul>
  </div>
  <div class="card red-border">
    <h3>Properties</h3>
    <ul>
      <li>Detection fires when: <code>e[k] ≥ 3 · EMA[k]</code></li>
      <li>Timeout of 100 ms prevents double-detections per bounce</li>
      <li><strong>Problem:</strong> this alone still passes false positives from gradual noise ramps</li>
    </ul>
  </div>
</div>
