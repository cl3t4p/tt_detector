<h2>Pipeline</h2>

<img src="imgs/11_full_pipeline_diagram.png" class="graph-img" alt="Full pipeline" />

<div class="content-row" style="margin-top: 2rem;">
  <div class="card blue-border">
    <h3>Signal Processing</h3>
    <ul>
      <li>High-pass filter + frame energy + EMA threshold detect candidate peaks in real-time</li>
      <li>Pre-peak ratio rejects false positives before extraction</li>
    </ul>
  </div>
  <div class="card purple-border">
    <h3>Classification</h3>
    <ul>
      <li>Each detected peak is extracted as a 15 ms audio clip</li>
      <li>Converted to a Mel spectrogram and fed into a CNN classifier</li>
      <li>CNN outputs: <em>surface</em> and <em>spin</em></li>
    </ul>
  </div>
</div>

