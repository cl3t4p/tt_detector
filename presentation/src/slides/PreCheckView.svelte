<h2>Pre-Peak — Frame View</h2>
<!--<p>Zoomed in to the moment of detection. The orange bars are the <strong style="color:var(--orange)">pre-peak window</strong>; the coloured bar at position 8 is the candidate frame.</p> -->
<img src="imgs/12_prepeak_ratio_explainer.png" class="graph-img" alt="Step-by-step" />

<div class="formula">ratio = e[i] / mean(e[i−window : i])</div>

<div class="content-row">
  <div class="card green-border">
    <h3>Case A — Sharp onset</h3>
    <p style="margin-top:8px;text-align:left;">The 8 frames before the candidate are essentially silent. The ratio is &gt;18
      the detection is accepted and a bounce timestamp is recorded.</p>
  </div>
  <div class="card red-border">
    <h3>Case B — Gradual onset</h3>
    <p style="margin-top:8px;text-align:left;">Energy has been climbing for several frames. The pre-peak mean is already elevated. The ratio falls below 18 the value is discarded.</p>
  </div>
</div>
