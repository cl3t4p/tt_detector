<script>
</script>

<h2>All Detectors — Benchmark</h2>

<table>
  <thead>
    <tr>
      <th>Detector</th>
      <th>Strategy</th>
      <th>Precision</th>
      <th>Recall</th>
      <th>F1</th>
      <th>FP</th>
    </tr>
  </thead>
  <tbody>
    <tr class="row-ours">
      <td><strong style="color:var(--green)">Decay Avg + Pre-Peak</strong> <span class="tag tag-green">new</span></td>
      <td>EMA threshold + pre-peak ratio gate</td>
      <td style="color:var(--green)"><strong>0.771</strong></td>
      <td style="color:var(--green)"><strong>0.987</strong></td>
      <td style="color:var(--green)"><strong>0.821</strong></td>
      <td style="color:var(--green)"><strong>422</strong></td>
    </tr>
    <tr>
      <td><strong style="color:var(--orange)">SuperFlux</strong></td>
      <td>Spectral flux + max-filter vibrato suppression</td>
      <td style="color:var(--orange)">0.701</td>
      <td style="color:var(--orange)">0.959</td>
      <td style="color:var(--orange)">0.740</td>
      <td style="color:var(--orange)">997</td>
    </tr>
    <tr>
      <td><strong style="color:var(--blue)">Decay Average</strong> <span class="tag tag-blue">paper</span></td>
      <td>EMA threshold — no pre-peak gate</td>
      <td style="color:var(--blue)">0.569</td>
      <td style="color:var(--blue)">0.883</td>
      <td style="color:var(--blue)">0.649</td>
      <td style="color:var(--blue)">1 234</td>
    </tr>
    <tr>
      <td><strong style="color:var(--red)">Spectral Flux</strong></td>
      <td>Half-wave-rectified STFT magnitude diff</td>
      <td style="color:var(--red)">0.052</td>
      <td style="color:var(--red)">0.083</td>
      <td style="color:var(--red)">0.057</td>
      <td style="color:var(--red)">1 615</td>
    </tr>
  </tbody>
</table>

<p class="table-note">Evaluated on 13 annotated files — tolerance 5 ms</p>

<div class="content-row" style="margin-top:18px">
  <div class="card" style="flex:1 1 auto; max-width:760px">
    <h3>Baseline Detectors <span style="color:var(--text-muted);font-weight:400;font-size:0.85rem">(single-file, 30 GT bounces)</span></h3>
    <table style="margin-top:12px">
      <thead>
        <tr><th>Detector</th><th>Strategy</th><th>Detections</th></tr>
      </thead>
      <tbody>
        <tr>
          <td>Adjacent Frame</td>
          <td>Compare consecutive frame energies</td>
          <td>37 / 30</td>
        </tr>
        <tr>
          <td>Moving Average</td>
          <td>Sliding-window mean threshold</td>
          <td>39 / 30</td>
        </tr>
        <tr>
          <td>Scipy Peaks</td>
          <td><code>scipy.signal.find_peaks</code> on energy</td>
          <td>35 / 30</td>
        </tr>
      </tbody>
    </table>
  </div>
</div>

<style>
  .table-note {
    font-size: 0.78rem;
    color: var(--text-muted);
    margin-top: 6px;
    text-align: center;
  }
  code {
    background: #010409;
    border-radius: 4px;
    padding: 1px 5px;
    font-size: 0.85em;
    color: var(--orange);
  }
  .row-ours td {
    background: rgba(63,185,80,.05);
  }
</style>
