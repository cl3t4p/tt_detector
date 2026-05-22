<h2>Conclusion</h2>

<div class="content-row" style="margin-top: 24px; gap: 24px; align-items: flex-start">
  <div class="card" style="flex: 1">
    <h3>Pre-Peak Gate</h3>
    <p>
      Adding a <strong style="color:var(--green)">pre-peak ratio check</strong> before confirming a bounce
      detection significantly reduces false positives caused by background noise and non-bounce
      transients.
    </p>
    <ul>
      <li>Paper baseline: <span style="color:var(--blue)"><strong>1 234 FP</strong></span></li>
      <li>With pre-peak gate: <span style="color:var(--green)"><strong>422 FP</strong></span> <span class="tag tag-green">−66 %</span></li>
    </ul>
  </div>


</div>

<div class="card" style="margin-top: 24px">
  <h3>Future Work</h3>
  <ul style="color:var(--text-muted)">
    <li>Stack additional filters on top of the pre-peak gate — e.g. checking the <strong style="color:var(--fg)">duration of the rising slope</strong>, the peak width, or the decay shape to further discriminate real bounces from noise bursts.</li>
    <li>Cross-validate detections against a <strong style="color:var(--fg)">camera-based ground truth system</strong> to get frame-accurate labels and reduce annotation uncertainty.</li>
    <li>Collect <strong style="color:var(--fg)">more diverse data</strong> — different tables, paddle rubbers, ball types, and recording environments — to stress-test robustness and avoid overfitting to current conditions.</li>
  </ul>
</div>

<style>
  ul {
    margin: 10px 0 0 18px;
    line-height: 1.8;
  }
  .card p {
    color: var(--text-muted);
    line-height: 1.6;
    margin-top: 8px;
  }
</style>
