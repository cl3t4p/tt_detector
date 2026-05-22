<script>
</script>

<h2>Signal Analysis — Frequency Spectra (FFT)</h2>
<p>Each bounce clip is <strong style="color:var(--blue)">15 ms</strong> long — we examine its frequency content across surfaces and spin types.</p>
<img src="imgs/13_fft_analysis.png" class="graph-img" alt="FFT frequency analysis" />
<div class="content-row">
  <div class="card purple-border">
    <h3>By Spin</h3>
    <ul>
      <li>Spin differences are subtle — all three curves overlap significantly</li>
      <li><strong>Backspin</strong> vs <strong>topspin</strong> mainly differ in the 5–15 kHz band</li>
      <li>This motivates the use of Mel spectrograms + CNN rather than hand-crafted rules</li>
    </ul>
  </div>
</div>
