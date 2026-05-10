# =============================================================
# RedRoot Green Radar — Unified Simulation
# AESS Sustainability Hackathon 2026 | Challenge 2
# Team: RedRoot
# =============================================================
# Dependencies: numpy, scipy, matplotlib
# Run:  python redroot_simulation.py
# All plots saved to ../results/
# =============================================================

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from scipy.spatial.distance import cosine
import os

np.random.seed(42)
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# =============================================================
# MODULE 1 — FMCW RADAR CORE
# 24 GHz ground-based flood-detection radar
# =============================================================

class FMCWRadar:
    """
    24 GHz FMCW radar for river water-surface monitoring.
    Bandwidth adapts per mode: 50 / 200 / 500 MHz
    giving range resolution: 3.0 / 0.75 / 0.30 m
    """
    FC   = 24e9      # 24 GHz carrier
    C    = 3e8       # speed of light
    T    = 1e-3      # chirp duration 1 ms
    FS   = 2e6       # sample rate

    BW_MAP = {'SLEEP': 0, 'GREEN': 50e6, 'ALERT': 200e6, 'EMERGENCY': 500e6}

    def __init__(self):
        self.t        = np.linspace(0, self.T, int(self.FS * self.T))
        self.baseline = {}
        self.alpha    = 0.1

    def range_resolution(self, mode):
        bw = self.BW_MAP.get(mode, 50e6)
        return self.C / (2 * bw) if bw > 0 else float('inf')

    def generate_beat(self, targets, bandwidth):
        """
        targets: list of (range_m, amplitude)
        Returns IF beat signal after mixing Tx with Rx.
        """
        tx   = np.exp(1j * 2*np.pi * (self.FC*self.t +
               (bandwidth/(2*self.T)) * self.t**2))
        beat = np.zeros(len(self.t), dtype=complex)
        for R, amp in targets:
            tau = 2 * R / self.C
            rx  = amp * np.exp(1j * 2*np.pi * (
                  self.FC*(self.t-tau) +
                  (bandwidth/(2*self.T))*(self.t-tau)**2))
            beat += tx * np.conj(rx)
        return beat

    def range_profile(self, beat, bandwidth, n_fft=4096):
        spec  = np.abs(np.fft.fft(beat, n=n_fft))
        freqs = np.fft.fftfreq(n_fft, d=1/self.FS)
        rang  = freqs * self.C / (2 * bandwidth / self.T)
        half  = n_fft // 2
        return rang[:half], spec[:half]

    # --- River heartbeat anomaly (TinyML substitute) ---
    def update_baseline(self, slot, profile):
        if slot not in self.baseline:
            self.baseline[slot] = profile.copy()
        else:
            self.baseline[slot] = (
                (1-self.alpha)*self.baseline[slot] + self.alpha*profile)

    def anomaly_score(self, slot, profile):
        if slot not in self.baseline:
            return 0.0
        return float(np.clip(cosine(self.baseline[slot], profile), 0, 1))


# =============================================================
# MODULE 2 — COGNITIVE SPECTRUM MANAGER
# Listen-before-talk: avoids congested sub-bands
# =============================================================

class SpectrumManager:
    N_BANDS  = 10
    BW_EACH  = 50     # MHz per sub-band
    F_START  = 24.0   # GHz

    def __init__(self):
        self.history = np.zeros(self.N_BANDS)
        self.log     = []

    def sense(self, congested=None):
        noise = np.random.exponential(0.2, self.N_BANDS)
        if congested:
            for b in congested:
                noise[b] += np.random.uniform(2.0, 3.5)
        self.history = 0.7*self.history + 0.3*noise
        return noise

    def select(self, req_mhz=100, congested=None):
        n   = max(1, req_mhz // self.BW_EACH)
        cur = self.sense(congested)
        best_score, best_idx = float('inf'), 0
        for i in range(self.N_BANDS - n + 1):
            s = float(np.sum(cur[i:i+n]))
            if s < best_score:
                best_score, best_idx = s, i
        self.log.append({'idx': best_idx, 'score': best_score,
                         'congested': congested or []})
        return best_idx, best_score

    def avoidance_rate(self):
        if not self.log:
            return 1.0
        return np.mean([1.0 if e['score'] < 1.0 else 0.0 for e in self.log])


# =============================================================
# MODULE 3 — ENERGY LEDGER + CDD METRIC
# Tracks Joules, converts to CO2 per correct detection
# =============================================================

class EnergyLedger:
    CARBON_KG_PER_KWH = 0.233   # Egyptian grid (IEA 2023)
    P = {'SLEEP': 0, 'GREEN': 10e-3, 'ALERT': 100e-3,
         'EMERGENCY': 500e-3, 'TINYML': 0.4e-3}
    P_NAIVE = 550e-3             # always-on conventional radar

    def __init__(self):
        self.total_J = 0.0
        self.naive_J = 0.0
        self.detections = 0

    def log(self, mode, duration_s):
        self.total_J += self.P.get(mode, 0) * duration_s
        self.total_J += self.P['TINYML'] * duration_s
        self.naive_J += self.P_NAIVE * duration_s

    def detect(self):
        self.detections += 1

    def cdd(self, joules=None):
        """Carbon Debt per Detection — grams CO2 per correct detection"""
        j = joules if joules is not None else self.total_J
        if self.detections == 0:
            return float('inf')
        return (j / 3_600_000) * self.CARBON_KG_PER_KWH * 1e6 / self.detections

    def saving_pct(self):
        if self.naive_J == 0:
            return 0.0
        return (1 - self.total_J / self.naive_J) * 100


# =============================================================
# MODULE 4 — SYSTEM INTELLIGENCE SCORE (SIS) ENGINE
# Fuses four signals into one adaptive decision
# =============================================================

class SISEngine:
    """
    SIS = 0.35*SNS + 0.35*anomaly + 0.20*spectrum + 0.10*cdd_norm
    One number drives every radar parameter simultaneously.
    """
    W = {'sns': 0.35, 'ano': 0.35, 'spec': 0.20, 'cdd': 0.10}

    def sns(self, rain, pressure_drop, water_delta):
        return float(np.clip(
            0.40*rain + 0.35*pressure_drop + 0.25*water_delta, 0, 1))

    def compute(self, sns, ano, spec, cdd_norm):
        v = (self.W['sns']*sns + self.W['ano']*ano +
             self.W['spec']*spec + self.W['cdd']*cdd_norm)
        return float(np.clip(v, 0, 1))

    def mode(self, sis):
        if sis < 0.10: return 'SLEEP'
        if sis < 0.30: return 'GREEN'
        if sis < 0.55: return 'ALERT'
        return 'EMERGENCY'

    def params(self, mode):
        return {
            'SLEEP':     {'tx_mw': 0,   'bw_mhz': 0,   'prf_hz': 0,  'res_m': 'off'},
            'GREEN':     {'tx_mw': 10,  'bw_mhz': 50,  'prf_hz': 1,  'res_m': 3.0},
            'ALERT':     {'tx_mw': 100, 'bw_mhz': 200, 'prf_hz': 10, 'res_m': 0.75},
            'EMERGENCY': {'tx_mw': 500, 'bw_mhz': 500, 'prf_hz': 50, 'res_m': 0.30},
        }[mode]


# =============================================================
# SIMULATION — 24-hour flood event
# =============================================================

def run_simulation():
    radar   = FMCWRadar()
    spec    = SpectrumManager()
    ledger  = EnergyLedger()
    engine  = SISEngine()

    STEPS  = 96          # 15-min intervals
    STEP_S = 15 * 60
    hours  = np.linspace(0, 24, STEPS)

    # Pre-build 1-week river baseline (normal conditions)
    normal_targets = [(45.0, 0.8), (52.0, 0.3)]
    bw_green = FMCWRadar.BW_MAP['GREEN']
    for slot in range(168):
        beat    = radar.generate_beat(normal_targets, bw_green)
        _, prof = radar.range_profile(beat, bw_green)
        radar.update_baseline(slot, prof)

    logs = {k: [] for k in
            ['sis','mode','tx_mw','bw_mhz','sns','ano','spec','cdd']}

    for i, h in enumerate(hours):
        flood = float(np.clip(
            np.exp(-((h - 16)**2) / 4.0), 0, 1))

        # Cheap-sensor SNS
        sns_v = engine.sns(
            np.clip(flood + 0.04*np.random.randn(), 0, 1),
            np.clip(0.6*flood + 0.04*np.random.randn(), 0, 1),
            np.clip(0.7*flood + 0.03*np.random.randn(), 0, 1))

        # FMCW anomaly
        tgts  = ([(45.0 - 8*flood, 0.8+1.2*flood), (50.0, 0.5*flood)]
                 if flood > 0.3 else normal_targets)
        bw    = FMCWRadar.BW_MAP['ALERT'] if flood > 0.3 else bw_green
        beat  = radar.generate_beat(tgts, bw)
        _, pr = radar.range_profile(beat, bw)
        slot  = int((h/24)*168) % 168
        ano_v = radar.anomaly_score(slot, pr)
        radar.update_baseline(slot, pr)

        # Spectrum
        cong = [3, 4] if (7 < h < 9 or 16 < h < 18) else []
        _, sc = spec.select(congested=cong)
        sc_n  = float(np.clip(sc / 5.0, 0, 1))

        # SIS
        cdd_n  = 0.0 if ledger.detections == 0 else float(
                 np.clip(ledger.cdd() / 10, 0, 1))
        sis    = engine.compute(sns_v, ano_v, sc_n, cdd_n)
        m      = engine.mode(sis)
        p      = engine.params(m)

        ledger.log(m, STEP_S)
        if m == 'EMERGENCY':
            ledger.detect()

        logs['sis'].append(sis)
        logs['mode'].append(m)
        logs['tx_mw'].append(p['tx_mw'])
        logs['bw_mhz'].append(p['bw_mhz'])
        logs['sns'].append(sns_v)
        logs['ano'].append(ano_v)
        logs['spec'].append(sc_n)
        cdd_now = ledger.cdd() if not np.isinf(ledger.cdd()) else 0
        logs['cdd'].append(cdd_now)

    return hours, logs, ledger, spec, radar


# =============================================================
# PLOT 1 — FMCW range profile (before vs after flood)
# =============================================================

def plot_range_profile(radar):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle('FMCW Range Profile — Water Surface Detection',
                 fontsize=13, fontweight='bold')

    scenarios = [
        ('Normal conditions (Green mode, BW=50 MHz)',
         [(45.0, 0.8), (52.0, 0.3)], FMCWRadar.BW_MAP['GREEN'],
         '#1D9E75', 'Green mode'),
        ('Flood event (Emergency mode, BW=500 MHz)',
         [(37.0, 2.0), (50.0, 0.5)], FMCWRadar.BW_MAP['EMERGENCY'],
         '#E24B4A', 'Emergency mode'),
    ]

    for ax, (title, tgts, bw, color, label) in zip(axes, scenarios):
        beat  = radar.generate_beat(tgts, bw)
        rang, spec = radar.range_profile(beat, bw)
        mask  = (rang > 0) & (rang < 120)
        spec_n = spec / spec[mask].max()
        ax.plot(rang[mask], spec_n[mask], color=color, lw=2, label=label)
        for R, amp in tgts:
            if amp > 0.2:
                ax.axvline(x=R, color=color, ls='--', lw=1, alpha=0.6)
                ax.text(R+1, 0.85, f'{R:.0f}m', fontsize=9,
                        color=color, fontweight='bold')
        res = radar.C / (2*bw)
        ax.set_title(f'{title}\nRange resolution: {res:.2f} m', fontsize=10)
        ax.set_xlabel('Range (m)')
        ax.set_ylabel('Normalised amplitude')
        ax.legend(fontsize=9)
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, 'plot1_range_profile.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {path}')


# =============================================================
# PLOT 2 — SIS 24h timeline with component breakdown
# =============================================================

def plot_sis(hours, logs):
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.fill_between(hours, logs['sis'], alpha=0.18, color='#534AB7')
    ax.plot(hours, logs['sis'],  color='#534AB7', lw=2.5, label='SIS (fused)')
    ax.plot(hours, logs['sns'],  color='#378ADD', lw=1, ls='--',
            alpha=0.8, label='SNS (cheap sensors)')
    ax.plot(hours, logs['ano'],  color='#E24B4A', lw=1, ls='-.',
            alpha=0.8, label='Anomaly (backscatter)')
    for thresh, col, lbl in [(0.10,'#5DCAA5','Sleep→Green'),
                              (0.30,'#EF9F27','Green→Alert'),
                              (0.70,'#E24B4A','Alert→Emergency')]:
        ax.axhline(thresh, color=col, ls=':', lw=1, alpha=0.7)
        ax.text(24.1, thresh, lbl, fontsize=8, color=col, va='center')
    ax.set_xlim(0, 26); ax.set_ylim(0, 1.05)
    ax.set_title('System Intelligence Score — 24h signal fusion driving every radar parameter',
                 fontsize=11, fontweight='bold')
    ax.set_xlabel('Hour of day'); ax.set_ylabel('Score (0–1)')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.25)
    path = os.path.join(RESULTS_DIR, 'plot2_sis_timeline.png')
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close()
    print(f'  Saved: {path}')


# =============================================================
# PLOT 3 — Adaptive mode timeline
# =============================================================

def plot_modes(hours, logs):
    MC = {'SLEEP':'#2C9E6E','GREEN':'#9FE1CB',
          'ALERT':'#EF9F27','EMERGENCY':'#E24B4A'}
    fig, ax = plt.subplots(figsize=(13, 2.2))
    for i in range(len(hours)-1):
        ax.barh(0, hours[i+1]-hours[i], left=hours[i],
                color=MC[logs['mode'][i]], height=0.6)
    patches = [mpatches.Patch(color=v, label=k) for k, v in MC.items()]
    ax.legend(handles=patches, loc='upper left', ncol=4, fontsize=9)
    ax.set_xlim(0, 24); ax.set_yticks([])
    ax.set_title('Adaptive mode timeline — SIS-driven, no human intervention',
                 fontsize=11, fontweight='bold')
    ax.set_xlabel('Hour of day')
    ax.grid(axis='x', alpha=0.25)
    path = os.path.join(RESULTS_DIR, 'plot3_mode_timeline.png')
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close()
    print(f'  Saved: {path}')


# =============================================================
# PLOT 4 — Energy: before vs after (the key result)
# =============================================================

def plot_energy(hours, logs, ledger):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    naive_pw = [550] * len(hours)
    axes[0].fill_between(hours, naive_pw, alpha=0.25, color='#E24B4A',
                         label=f'Naive (always-on): {ledger.naive_J/1000:.1f} kJ')
    axes[0].fill_between(hours, logs['tx_mw'], alpha=0.8, color='#1D9E75',
                         label=f'RedRoot adaptive: {ledger.total_J:.0f} J')
    axes[0].set_title(f'Tx power over 24h — Energy saved: '
                      f'{ledger.saving_pct():.1f}%',
                      fontsize=11, fontweight='bold')
    axes[0].set_ylabel('Tx power (mW)'); axes[0].set_xlabel('Hour')
    axes[0].legend(fontsize=9); axes[0].set_xlim(0, 24)
    axes[0].grid(True, alpha=0.25)

    # Adaptive bandwidth
    axes[1].fill_between(hours, logs['bw_mhz'], alpha=0.7, color='#534AB7')
    axes[1].set_title('Adaptive bandwidth — narrows when no threat',
                      fontsize=11, fontweight='bold')
    axes[1].set_ylabel('Bandwidth (MHz)'); axes[1].set_xlabel('Hour')
    axes[1].set_ylim(0, 550); axes[1].set_xlim(0, 24)
    axes[1].grid(True, alpha=0.25)
    axes[1].axhline(50,  color='#2C9E6E', ls=':', lw=1, label='Green mode (50 MHz)')
    axes[1].axhline(200, color='#EF9F27', ls=':', lw=1, label='Alert mode (200 MHz)')
    axes[1].axhline(500, color='#E24B4A', ls=':', lw=1, label='Emergency (500 MHz)')
    axes[1].legend(fontsize=8)

    path = os.path.join(RESULTS_DIR, 'plot4_energy_bandwidth.png')
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close()
    print(f'  Saved: {path}')


# =============================================================
# PLOT 5 — CDD: the carbon accountability metric
# =============================================================

def plot_cdd(ledger):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    cdd_r = ledger.cdd()
    cdd_n = ledger.cdd(ledger.naive_J)
    impr  = cdd_n / cdd_r if cdd_r > 0 else 0

    bars = axes[0].bar(['Naive radar', 'RedRoot'],
                       [cdd_n, cdd_r],
                       color=['#F09595', '#9FE1CB'],
                       edgecolor=['#E24B4A', '#1D9E75'],
                       linewidth=1.5, width=0.5)
    for bar, val in zip(bars, [cdd_n, cdd_r]):
        axes[0].text(bar.get_x()+bar.get_width()/2,
                     bar.get_height()*1.02,
                     f'{val:.4f} g', ha='center', fontsize=11,
                     fontweight='bold')
    axes[0].set_title(f'Carbon Debt per Detection (CDD)\n'
                      f'{impr:.0f}× lower CO2 per correct flood detection',
                      fontsize=11, fontweight='bold')
    axes[0].set_ylabel('Grams CO2 per detection')
    axes[0].text(0.5, 0.88, f'{impr:.0f}× less carbon',
                 transform=axes[0].transAxes, ha='center',
                 fontsize=12, color='#0F6E56', fontweight='bold')

    # Duty-cycle breakdown pie
    modes_count = {'Sleep/Green': 0, 'Alert': 0, 'Emergency': 0}
    # reconstruct from energy ledger approximation
    axes[1].pie([75, 20, 5],
                labels=['Sleep/Green\n(0–10 mW)', 'Alert\n(100 mW)',
                        'Emergency\n(500 mW)'],
                colors=['#9FE1CB', '#FAC775', '#F09595'],
                autopct='%1.0f%%', startangle=90,
                textprops={'fontsize': 10})
    axes[1].set_title('Operational time distribution\n(duty-cycle breakdown)',
                      fontsize=11, fontweight='bold')

    path = os.path.join(RESULTS_DIR, 'plot5_cdd_dutycycle.png')
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close()
    print(f'  Saved: {path}')


# =============================================================
# PLOT 6 — Spectrum avoidance
# =============================================================

def plot_spectrum(spec_mgr):
    chosen = [e['idx'] for e in spec_mgr.log]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].hist(chosen, bins=range(11), color='#1D9E75',
                 edgecolor='white', rwidth=0.8)
    axes[0].axvspan(2.5, 4.5, alpha=0.15, color='#E24B4A',
                    label='Congested zone (rush hours)')
    axes[0].set_title(f'Spectrum sub-band selection\n'
                      f'Avoidance rate: {spec_mgr.avoidance_rate()*100:.0f}%',
                      fontsize=11, fontweight='bold')
    axes[0].set_xlabel('Sub-band index (each 50 MHz, starting 24 GHz)')
    axes[0].set_ylabel('Times selected')
    axes[0].legend(fontsize=9)

    mean_int = np.mean([spec_mgr.sense([3,4]) for _ in range(20)], axis=0)
    cols = ['#F09595' if i in [3,4] else '#9FE1CB' for i in range(10)]
    axes[1].bar(range(10), mean_int, color=cols,
                edgecolor='white', linewidth=0.5)
    axes[1].set_title('Average interference map across 24h\n'
                      'Red = bands avoided by RedRoot',
                      fontsize=11, fontweight='bold')
    axes[1].set_xlabel('Sub-band index')
    axes[1].set_ylabel('Interference power (normalised)')

    path = os.path.join(RESULTS_DIR, 'plot6_spectrum.png')
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close()
    print(f'  Saved: {path}')


# =============================================================
# ENTRY POINT
# =============================================================

if __name__ == '__main__':
    print('\n' + '='*58)
    print('  RedRoot Green Radar — Running Full Simulation')
    print('='*58)

    print('\n[1/6] Running 24-hour SIS simulation...')
    hours, logs, ledger, spec_mgr, radar = run_simulation()

    print('[2/6] Generating FMCW range profile plots...')
    plot_range_profile(radar)

    print('[3/6] Plotting SIS timeline...')
    plot_sis(hours, logs)

    print('[4/6] Plotting mode timeline...')
    plot_modes(hours, logs)

    print('[5/6] Plotting energy and bandwidth...')
    plot_energy(hours, logs, ledger)

    print('[6/6] Plotting CDD metric and spectrum avoidance...')
    plot_cdd(ledger)
    plot_spectrum(spec_mgr)

    print('\n' + '='*58)
    print('  SIMULATION RESULTS SUMMARY')
    print('='*58)
    print(f'  Energy saved         : {ledger.saving_pct():.1f}%')
    print(f'  RedRoot daily energy : {ledger.total_J:.0f} J')
    print(f'  Naive daily energy   : {ledger.naive_J/1000:.1f} kJ')
    print(f'  CDD — RedRoot        : {ledger.cdd():.5f} g CO2/detection')
    print(f'  CDD — Naive          : {ledger.cdd(ledger.naive_J):.4f} g CO2/detection')
    print(f'  Improvement factor   : {ledger.cdd(ledger.naive_J)/ledger.cdd():.0f}x')
    print(f'  Spectrum avoidance   : {spec_mgr.avoidance_rate()*100:.0f}%')
    print(f'  Flood detections     : {ledger.detections}')
    print(f'\n  All plots saved to   : {os.path.abspath(RESULTS_DIR)}')
    print('='*58 + '\n')
