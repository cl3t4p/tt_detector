<h2>CNN Classifier — Results</h2>
<!--<p>A lightweight <strong style="color:var(--blue)">6-layer CNN</strong> trained end-to-end on Mel spectrograms — <strong style="color:var(--orange)">25 k parameters</strong>, two separate heads for surface and spin.</p>
-->
<img src="imgs/15_cnn_results.png" class="graph-img" alt="CNN classification results" />
<div class="stat-row">
  <div class="stat">
    <div class="stat-val" style="color:var(--green)">94.0 %</div>
    <div class="stat-lbl">Surface Accuracy<br/><em>13 classes</em></div>
  </div>
  <div class="stat">
    <div class="stat-val" style="color:var(--blue)">94.9 %</div>
    <div class="stat-lbl">Spin Accuracy<br/><em>3 classes</em></div>
  </div>
</div>
<div class="content-row">
  <div class="card red-border">
    <h3>Key Observations</h3>
    <ul>
      <li>Most surface misclassifications occur between <strong>similar racket rubbers</strong></li>
      <li>Spin confusion is mainly <strong>back ↔ top</strong> — physically the most similar impacts</li>
      <li>No-spin is classified near-perfectly as it has the most training samples</li>
    </ul>
  </div>
</div>
