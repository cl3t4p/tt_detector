<h2>Stage 5 — Pre-Peak Ratio Check</h2>
<p>After passing the EMA threshold, each candidate detection undergoes a <strong style="color:var(--orange)">sharpness test</strong>: the current frame energy must be at least <code>pre_peak_ratio</code> times larger than the mean of the preceding <code>pre_peak_window</code> frames.</p>

<pre><span class="cm"># After EMA threshold fires at frame i:</span>
<span class="kw">if</span> <span class="var">self</span>.pre_peak_ratio &gt; <span class="num">0</span> <span class="kw">and</span> i &gt;= <span class="var">self</span>.pre_peak_window:

    <span class="cm"># Mean energy in the window BEFORE the candidate frame</span>
    <span class="var">pre_mean</span> = energy[i - <span class="var">self</span>.pre_peak_window : i].mean()

    <span class="cm"># Compute sharpness ratio</span>
    <span class="kw">if</span> <span class="var">pre_mean</span> &gt; <span class="num">1e-10</span> <span class="kw">and</span> e / <span class="var">pre_mean</span> &lt; <span class="var">self</span>.pre_peak_ratio:
        <span class="kw">continue</span>  <span class="cm"># REJECT — rise is too gradual</span>

<span class="cm"># Passes all checks → record timestamp</span>
peaks.append(timestamp)</pre>

<div class="content-row">
  <div class="card orange-border">
    <h3>Parameters</h3>
    <table>
      <tr><th>Parameter</th><th>Default</th><th>Role</th></tr>
      <tr><td><code>pre_peak_ratio</code></td><td>18.0</td><td>Minimum required sharpness multiplier</td></tr>
      <tr><td><code>pre_peak_window</code></td><td>8</td><td>History window length (ms frames)</td></tr>
    </table>
  </div>
  <div class="card blue-border">
    <h3>Design Choices</h3>
    <ul>
      <li><code>pre_mean &gt; 1e-10</code> guard: avoid division by near-zero silence</li>
      <li>Window of 8 ms: captures the relevant pre-onset context without including the rise itself</li>
      <li>Ratio 18×: empirically tuned to ball bounce acoustics</li>
      <li>Setting <code>pre_peak_ratio = 0</code> disables the check entirely</li>
    </ul>
  </div>
</div>
