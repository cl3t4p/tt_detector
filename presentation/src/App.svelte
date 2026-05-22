<script>
  import Intro from './slides/Intro.svelte'
  import SlideFullPipeline from './slides/SlideFullPipeline.svelte'
  import DetectionPipeline from './slides/DetectionPipeline.svelte'
  import HighPassFilter from './slides/HighPassFilter.svelte'
  import FrameEnergy from './slides/FrameEnergy.svelte'
  import EnergyMethods from './slides/EnergyMethods.svelte'
  import EMA from './slides/EMA.svelte'
  import PreCheckView from './slides/PreCheckView.svelte'
  import PreCheckComparison from './slides/PreCheckComparison.svelte'
  import SignalSpinAnalysis from './slides/SignalSpinAnalysis.svelte'
  import SignalSurfaceAnalysis from './slides/SignalSurfaceAnalysis.svelte'
  import MelSpectrogram from './slides/MelSpectrogram.svelte'
  import CNNResults from './slides/CNNResults.svelte'
  import ImprovementGoals from './slides/ImprovementGoals.svelte'
  import DecayComparison from './slides/DecayComparison.svelte'
  import AllDetectors from './slides/AllDetectors.svelte'
  import Conclusion from './slides/Conclusion.svelte'
  import DetectionOldPipeline from "./slides/DetectionOldPipeline.svelte";



  const slideComponents = [
  	{title : 'Overview',component : Intro},
  	{title : 'Full Pipeline',component : SlideFullPipeline},
  	{title : 'Original Detection Pipeline',component : DetectionOldPipeline},
  	{title : 'High-Pass',component : HighPassFilter},
    {title : 'Energy',component : FrameEnergy},
  	{title : 'EMA',component : EMA},
  	{title : 'FFT Spin Analysis',component : SignalSpinAnalysis},
    {title : 'FFT Surface Analysis',component : SignalSurfaceAnalysis},
  	{title : 'Mel Spectrogram',component : MelSpectrogram},
  	{title : 'CNN Results',component : CNNResults},
    {title : 'Improvements',component : ImprovementGoals},
    {title : 'Energy Methods',component : EnergyMethods},
    {title : 'Detection Pipeline',component : DetectionPipeline},
    {title : 'Pre-Peak',component : PreCheckView},
    {title : 'Comparison',component : PreCheckComparison},
    {title : 'Decay vs Pre-Peak',component : DecayComparison},
    {title : 'All Detectors',component : AllDetectors},
    {title : 'Conclusion',component : Conclusion},
  ]

  const total = slideComponents.length

  let current = 0
  let exiting = {}
  let navEl

  $: progress = (current / (total - 1)) * 100

  $: if (navEl) {
    const active = navEl.querySelectorAll('a')[current]
    if (active) active.scrollIntoView({ block: 'nearest', inline: 'center', behavior: 'smooth' })
  }

  function goTo(n, direction) {
    if (n < 0 || n >= total) return
    const old = current
    exiting = { ...exiting, [old]: direction === 'next' ? 'left' : 'right' }
    setTimeout(() => {
      const copy = { ...exiting }
      delete copy[old]
      exiting = copy
    }, 380)
    current = n
  }

  function handleKey(e) {
    if (e.key === 'ArrowRight' || e.key === 'ArrowDown' || e.key === ' ') goTo(current + 1, 'next')
    if (e.key === 'ArrowLeft'  || e.key === 'ArrowUp')                    goTo(current - 1, 'prev')
    if (e.key === 'Home')                                                  goTo(0, 'prev')
    if (e.key === 'End')                                                   goTo(total - 1, 'next')
  }
</script>

<svelte:window on:keydown={handleKey} />

<nav bind:this={navEl}>
  <span class="nav-title">TT Detector</span>
  {#each slideComponents as label, i}
    <a class:nav-active={i === current} on:click={() => goTo(i, i > current ? 'next' : 'prev')}>{label.title}</a>
  {/each}
</nav>

<button class="arrow-btn" id="btn-prev" disabled={current === 0} on:click={() => goTo(current - 1, 'prev')}>&#8592;</button>
<button class="arrow-btn" id="btn-next" disabled={current === total - 1} on:click={() => goTo(current + 1, 'next')}>&#8594;</button>

<div id="progress-bar" style="width: {progress}%"></div>

<div class="deck">
  {#each slideComponents as SlideComp, i}
    <section
      class="slide"
      class:active={i === current}
      class:exit-left={exiting[i] === 'left'}
      class:exit-right={exiting[i] === 'right'}
    >
      <div class="slide-num">{String(i + 1).padStart(2, '0')} / {String(total).padStart(2, '0')}</div>
      <svelte:component this={SlideComp.component} />
    </section>
  {/each}
</div>
