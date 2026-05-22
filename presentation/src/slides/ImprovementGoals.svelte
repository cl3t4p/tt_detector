<h2>Improving the Detection Design</h2>

<div class="content-row" style="margin-top: 24px; gap: 24px; align-items: flex-start">
  <div class="card orange-border" style="flex: 1">
    <h3>Problem — Noise inflating scores</h3>
    <p style="margin-top: 8px; text-align: left;">
      The dataset contains <strong style="color:var(--orange)">noise-labelled events</strong> — entries where
      racket-type, spin-magnitude and spin-direction are all <code>none</code>, or surface is <code>noise</code>.
      Without filtering, these inflate every detector's apparent score.
    </p>
    <ul style="margin-top: 10px">
      <li>Noise entries are <strong>excluded from valid GT</strong> when benchmarking</li>
      <li>Any prediction landing near a noise timestamp is <strong style="color:var(--orange)">forced to remain a FP</strong></li>
      <li>This gives a more honest picture of real detection quality</li>
    </ul>
  </div>

  <div class="card blue-border" style="flex: 1">
    <h3>Goal — Better Detectors</h3>
    <ul style="margin-top: 8px">
      <li>The EMA threshold alone gives high recall but poor precision — it fires on anything that spikes above background</li>
      <li>The <strong style="color:var(--green)">pre-peak gate</strong> is a first step: it cuts FP by <strong>−66 %</strong> with zero extra cost</li>
    </ul>
  </div>
</div>

<style>
  code {
    background: #010409;
    border-radius: 4px;
    padding: 1px 5px;
    font-size: 0.85em;
    color: var(--orange);
  }
  .card p {
    color: var(--text-muted);
    line-height: 1.6;
  }
</style>
