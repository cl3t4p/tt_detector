<h2>Energy Calculation Methods — Comparison</h2>
<p>The same <strong style="color:var(--orange)">DecayAverage detector</strong> was run with four different energy functions evaluated across all recording files.</p>
<img src="imgs/02b_energy_method_comparison.png" class="graph-img" alt="Energy method comparison" />

<div class="content-row">
  <div class="card blue-border">
    <h3>Methods Tested</h3>
    <ul>
      <li><strong style="color:var(--blue)">MAE (Simple)</strong> — mean absolute value per frame</li>
      <li><strong style="color:var(--blue)">RMS</strong> — root mean square, penalises large peaks more</li>
      <li><strong style="color:var(--blue)">Power</strong> — mean squared amplitude</li>
    </ul>
  </div>
  <div class="card green-border">
    <h3>Takeaway</h3>
    <ul>
      <li>MAE and RMS lead on precision — fewer false positives</li>
      <li>All methods show similar recall and F1, confirming the <strong>bottleneck is the threshold logic</strong>, not the energy function</li>
      <li>MAE is chosen for its simplicity and speed</li>
    </ul>
  </div>
</div>
