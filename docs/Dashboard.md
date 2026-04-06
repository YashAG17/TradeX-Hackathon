# Dashboard Documentation

The TradeX Dashboard (`dashboard.py`) is an interactive Gradio-based UI for running, analyzing, and comparing market surveillance episodes. It provides real-time visualizations of AMM state, agent actions, signal intensity, and grading outcomes.

---

## Application Architecture

### Entry Points

| Mode | Trigger | Behavior |
|------|---------|----------|
| **HF Space** | `SPACE_ID` or `HF_SPACE_ID` env var detected | Combines Playground + Dashboard in `gr.Tabs` |
| **OpenEnv** | Default local mode | Serves FastAPI OpenEnv server |
| **Standalone** | `python dashboard.py` | Runs Gradio dashboard only |

### Tech Stack

| Library | Purpose |
|---------|---------|
| **Gradio >= 6.11.0** | UI framework (Blocks, Tabs, Plots, Dataframes, Markdown) |
| **Plotly >= 5.24.0** | All chart/graph rendering via `plotly.graph_objects` and `plotly.subplots` |
| **NumPy >= 1.26.0** | Array operations (cumulative reward, matrix transposition) |
| **openenv-core >= 0.2.2** | OpenEnv environment server framework |

---

## Dashboard Tabs

The dashboard contains **4 tabs**: Episode Runner, Policy Comparison, Telemetry Viewer, and About.

---

## Tab 1: Episode Runner

The main interactive tab for running surveillance episodes. Users select a task, policy, and seed, then click "Run" to execute a full episode and view all visualizations.

### Input Controls

| Control | Type | Options | Default | Purpose |
|---------|------|---------|---------|---------|
| **Task** | Dropdown | `burst_detection`, `pattern_manipulation_detection`, `full_market_surveillance` | `full_market_surveillance` | Selects the surveillance scenario |
| **Policy** | Dropdown | `Heuristic`, `Always Allow`, `Random` | `Heuristic` | Selects the decision-making policy |
| **Seed** | Number | 0-999999 (0 = random) | 42 | Fixed seed for reproducibility |
| **Run** | Button (primary, lg) | -- | -- | Triggers `run_full_episode()` |

### Task Definitions

| Task | Difficulty | Steps | Initial Bot Confidence | Profile |
|------|-----------|-------|----------------------|---------|
| `burst_detection` | Easy | 50 | 0.25 | Burst-heavy (burst_bias=0.8, pattern_bias=0.2) |
| `pattern_manipulation_detection` | Medium | 50 | 0.35 | Pattern-heavy (burst_bias=0.2, pattern_bias=0.8) |
| `full_market_surveillance` | Hard | 60 | 0.30 | Mixed (burst_bias=0.5, pattern_bias=0.5) |

### Policy Definitions

| Policy | Behavior |
|--------|----------|
| **Heuristic** | Threshold-based rules from `meverse/baseline_policy.py` |
| **Always Allow** | Returns `ALLOW` every step (sanity-check lower bound) |
| **Random** | Picks randomly from `[ALLOW, FLAG, BLOCK, MONITOR]` |

---

### Output Visualizations

#### 1. Episode Summary (Markdown)

**Type**: `gr.Markdown`

**What it shows**: Episode metadata and final score in a formatted table with a visual score bar.

**Contents**:
- Task name
- Policy name
- Seed value
- Total steps completed
- Total reward
- Average reward per step
- Final score with a visual progress bar (ASCII-style)

**Data source**: Computed from `EpisodeState` and the grade dictionary returned by `env.grade()`.

---

#### 2. AMM Final State Gauges

**Type**: `gr.Plot` (Plotly `go.Indicator` gauge chart)

**What it shows**: Four gauge indicators displaying the final state of the AMM after the episode completes.

**Sub-elements** (4 gauges in a 1x4 subplot):

| Gauge | Range | Color | What it means |
|-------|-------|-------|---------------|
| **Price** | 50-150 | Blue (`#0088cc`) | Final AMM price (`reserve_y / reserve_x`). Shows where the market settled after all interventions. |
| **Bot Confidence** | 0-1 | Red (`#ff4757`) | Final bot confidence level. Lower values mean the bot was successfully deterred; higher values mean it grew bolder. |
| **Health** | 0-1 | Green (`#2ed573`) | Final health index of the market. Reflects how well normal market participation was preserved. |
| **Volatility** | 0-0.5 | Amber (`#ffa502`) | Final volatility measurement. Higher values indicate more erratic price movement. |

**Chart type**: `go.Indicator` with `mode="gauge+number"`

**Built by**: `_make_amm_gauges()` at `dashboard.py:717-759`

---

#### 3. Reward Timeline Chart

**Type**: `gr.Plot` (Plotly dual-axis bar + line chart)

**What it shows**: Per-step rewards as color-coded bars AND cumulative reward as a dotted line over time.

**Visual encoding**:
- **Bars**: Colored by the action taken at each step
  - ALLOW = Green (`#2ed573`)
  - FLAG = Amber (`#ffa502`)
  - BLOCK = Red (`#ff4757`)
  - MONITOR = Blue (`#3498db`)
- **Line**: Teal (`#00c9a7`) dotted line on secondary y-axis showing cumulative reward trajectory

**Hover data**: Step number, reward value, action taken, ground truth label

**Animation**: Bars grow from zero, line rises from zero (Plotly frames)

**What it means**: The bar colors reveal the agent's action pattern over time. A good policy will show green (ALLOW) during normal periods and red/amber (BLOCK/FLAG) during suspicious periods. The cumulative line shows whether the agent is accumulating positive or negative reward overall.

**Built by**: `_make_reward_chart()` at `dashboard.py:256-322`

---

#### 4. Action Distribution Chart

**Type**: `gr.Plot` (Plotly stacked bar chart)

**What it shows**: How many times each action was taken, broken down by ground truth label.

**X-axis**: Action types (ALLOW, FLAG, BLOCK, MONITOR)

**Stacked segments**:
- **Green**: Actions taken on normal activity
- **Red**: Actions taken on suspicious activity

**What it means**: This chart reveals the agent's decision distribution. For a good policy:
- ALLOW bar should be mostly green (correctly allowing normal activity)
- BLOCK/FLAG bars should be mostly red (correctly catching suspicious activity)
- Heavy green in BLOCK/FLAG indicates false positives
- Heavy red in ALLOW indicates false negatives

**Animation**: Stacked bars grow from zero

**Built by**: `_make_action_dist_chart()` at `dashboard.py:325-379`

---

#### 5. Signal Heatmap

**Type**: `gr.Plot` (Plotly `go.Heatmap`)

**What it shows**: Intensity of 6 surveillance signals over time.

**Y-axis** (6 signals):
| Signal | What it measures |
|--------|-----------------|
| **Burst** | Acute high-frequency trading activity (0.0-1.0) |
| **Pattern** | Rhythmic/coordinated trading patterns (0.0-1.0) |
| **Suspicion** | Composite suspicion metric (0.0-1.0) |
| **Manipulation** | Deliberate manipulation confidence (0.0-1.0) |
| **Frequency** | Trade frequency rate |
| **Slippage** | Price impact from recent trades |

**X-axis**: Step number (0 to N)

**Color scale** (intensity):
- Dark background (`#0f1117`) = No signal
- Dark blue (`#0a3d5c`) = Low signal
- Blue (`#0088cc`) = Moderate signal
- Amber (`#ffa502`) = High signal
- Red (`#ff4757`) = Critical signal

**What it means**: This heatmap reveals when and which signals spiked during the episode. Horizontal bands of red/amber indicate sustained suspicious activity. A good agent should take BLOCK/FLAG actions when these bands appear. The pattern signal is particularly important for `pattern_manipulation_detection` tasks.

**Animation**: Fades in from zero matrix

**Built by**: `_make_signal_heatmap()` at `dashboard.py:382-426`

---

#### 6. AMM State Evolution Chart

**Type**: `gr.Plot` (Plotly 2-row subplot with shared x-axis)

**What it shows**: Tracks AMM state variables across all steps of the episode.

**Row 1 -- "AMM Price & Liquidity"**:
| Line | Color | Style | Axis | What it means |
|------|-------|-------|------|---------------|
| Price | Blue (`#0088cc`) | Solid | Primary y-axis | Current AMM price (`reserve_y / reserve_x`). Drifts as trades occur. |
| Liquidity | Teal (`#00c9a7`) | Dashed | Secondary y-axis | Pool liquidity (`2 * sqrt(k)`). Changes as simulated trades move reserves. |

**Row 2 -- "Bot Confidence, Volatility & Health"** (all on 0-1 scale):
| Line | Color | Style | What it means |
|------|-------|-------|---------------|
| Bot Confidence | Red (`#ff4757`) | Solid | How emboldened the bot is. Decreases after successful BLOCKs, increases after missed detections. |
| Volatility | Amber (`#ffa502`) | Dotted | Price erratic-ness. Spikes during burst attacks. |
| Health Index | Green (`#2ed573`) | Dash-dot | Market health. Drops on false positives (blocking normal activity). |

**What it means**: This chart shows the dynamic evolution of the market state. The bot confidence line is particularly important -- it should trend downward for a good policy (the bot is being deterred). Price and liquidity show the market impact of the simulated trades. Health should stay high for a policy that doesn't overblock.

**Built by**: `_make_amm_chart()` at `dashboard.py:429-530`

---

#### 7. Grade Radar Chart

**Type**: `gr.Plot` (Plotly `go.Scatterpolar` radar/polar chart)

**What it shows**: Visual breakdown of the 5 grading components that determine the final episode score.

**Axes** (5 radial dimensions, range 0-1):
| Component | Weight | What it measures |
|-----------|--------|-----------------|
| **Detection** | 50% | How well the agent caught suspicious activity |
| **False Positive** | 20% | How well the agent avoided flagging/blocking normal activity |
| **False Negative** | 15% | How well the agent avoided missing suspicious activity |
| **Health** | 10% | How well the agent preserved healthy market participation |
| **Overblocking** | 5% | How well the agent avoided blocking normal users |

**Visual encoding**:
- Fill: Semi-transparent teal (`rgba(0,200,167,0.15)`)
- Line: Teal (`#00c9a7`), width 2
- Range: 0 (center) to 1 (outer edge)

**What it means**: The shape of the radar reveals the agent's strengths and weaknesses. A large, balanced shape indicates a well-rounded policy. A spike in Detection but a dip in False Positive means the agent catches threats but also overreacts. A score >= 0.6 overall is considered passing.

**Animation**: Expands from center (radar grows outward)

**Built by**: `_make_grade_chart()` at `dashboard.py:533-598`

---

#### 8. Action vs Ground Truth (Confusion Matrix)

**Type**: `gr.Plot` (Plotly `go.Heatmap` confusion matrix)

**What it shows**: 2x4 matrix showing actions taken versus true labels.

**Y-axis** (2 rows): `normal`, `suspicious`

**X-axis** (4 columns): `ALLOW`, `MONITOR`, `FLAG`, `BLOCK`

**Color scale**:
- Dark (`#1a1d27`) = Zero counts
- Blue (`#0088cc`) = Low counts
- Teal (`#00c9a7`) = High counts

**Text overlay**: Count numbers displayed in each cell (white, size 16)

**What it means**: This is the definitive accuracy chart. Ideal distribution:
- `normal` row: Heavy on ALLOW, light on MONITOR, zero on FLAG/BLOCK
- `suspicious` row: Heavy on BLOCK/FLAG, light on MONITOR, zero on ALLOW

Off-diagonal entries reveal errors:
- `normal` + BLOCK/FLAG = False positives
- `suspicious` + ALLOW = False negatives

**Animation**: Fades in from zero

**Built by**: `_make_confusion_chart()` at `dashboard.py:601-658`

---

#### 9. Step-by-Step Log Table

**Type**: `gr.Dataframe`

**What it shows**: Detailed per-step log of the entire episode.

**Columns**:
| Column | Type | Description |
|--------|------|-------------|
| Step | number | Step index (1-based) |
| Action | str | Action taken (ALLOW, FLAG, BLOCK, MONITOR) |
| Label | str | Ground truth (normal, suspicious) |
| Reward | str | Reward earned for this step |
| Burst | str | Burst indicator value |
| Pattern | str | Pattern indicator value |
| Suspicion | str | Suspiciousness score |
| Manipulation | str | Manipulation score |

**Properties**: Non-interactive, text-wrapped

**What it means**: This is the raw data behind all charts. Use it to inspect individual steps, understand why specific actions were taken, and trace the agent's decision logic step by step.

**Built by**: `_make_step_table()` at `dashboard.py:701-714`

---

## Tab 2: Policy Comparison

This tab runs the same episode (same task, same seed) across all three baseline policies and compares their results side-by-side.

### Input Controls

| Control | Type | Options | Purpose |
|---------|------|---------|---------|
| **Task** | Dropdown | All 3 tasks | Select task to compare on |
| **Seed** | Number | 1-999999 | Fixed seed for fair comparison |
| **Compare** | Button (primary) | -- | Triggers `compare_policies()` |

### Output Visualizations

#### 10. Policy Comparison Chart

**Type**: `gr.Plot` (Plotly grouped bar chart)

**What it shows**: Side-by-side comparison of Heuristic, Always Allow, and Random policies.

**X-axis**: Policy names (Heuristic, Always Allow, Random)

**Grouped bars** (6 metrics per policy):
| Metric | Color | What it means |
|--------|-------|---------------|
| Final Score | Teal (`#00c9a7`) | Overall episode score (0-1) |
| Detection | Blue (`#0088cc`) | Detection component score |
| False Pos. | Green (`#2ed573`) | False positive component score |
| False Neg. | Red (`#ff4757`) | False negative component score |
| Health | Light blue | Health component score |
| Overblocking | Amber (`#ffa502`) | Overblocking component score |

**Y-axis**: Score (0-1)

**What it means**: This chart reveals which policy performs best and why. The Heuristic policy should dominate across most metrics. Always Allow will have high False Negatives. Random will be inconsistent across all metrics.

**Built by**: `compare_policies()` at `dashboard.py:766-849`

#### 11. Policy Comparison Summary

**Type**: `gr.Markdown`

**What it shows**: Markdown table with full numeric breakdown per policy.

**Columns**: Policy, Score, Detection, FP, FN, Health, Overblocking, Total Reward

---

## Tab 3: Telemetry Viewer

This tab allows users to upload and visualize telemetry files from previous runs.

### Input Controls

| Control | Type | Purpose |
|---------|------|---------|
| **File Upload** | `gr.File` | Upload `.jsonl` telemetry files (single file, elem_id="telem-upload") |

### Output Visualizations

#### 12. Telemetry Rewards Chart

**Type**: `gr.Plot` (Plotly bar chart)

**What it shows**: Replays reward timeline from uploaded telemetry file.

**Bars**: Color-coded by action taken (same color scheme as Episode Runner)

**What it means**: Allows post-hoc analysis of any episode. Useful for debugging LLM policy runs from `inference.py` by uploading the debug telemetry file.

**Built by**: `load_telemetry()` at `dashboard.py:869-916`

#### 13. Telemetry Summary

**Type**: `gr.Markdown`

**What it shows**: Task name, model name, step count, total reward, and final score (if available).

---

## Tab 4: About

**Type**: `gr.Markdown`

**What it shows**: Static documentation explaining TradeX, tasks, actions, scoring components, and AMM dynamics.

---

## Color Palette

### Core Colors

| Name | Hex | Usage |
|------|-----|-------|
| bg | `#0f1117` | Page background |
| surface | `#1a1d27` | Card/block backgrounds |
| border | `#2a2d3a` | Borders, grid lines |
| text | `#e4e6eb` | Primary text |
| muted | `#8b8fa3` | Secondary text, labels |
| accent | `#00c9a7` | Teal -- primary accent, buttons, cumulative line |
| accent2 | `#0088cc` | Blue -- price line, secondary accent |
| danger | `#ff4757` | Red -- BLOCK, suspicious, bot confidence |
| warning | `#ffa502` | Amber -- FLAG, volatility |
| success | `#2ed573` | Green -- ALLOW, normal, health |
| info | `#3498db` | Light blue -- MONITOR |

### Action Colors

| Action | Color | Hex |
|--------|-------|-----|
| ALLOW | Green | `#2ed573` |
| FLAG | Amber | `#ffa502` |
| BLOCK | Red | `#ff4757` |
| MONITOR | Blue | `#3498db` |

### Label Colors

| Label | Color | Hex |
|-------|-------|-----|
| suspicious | Red | `#ff4757` |
| normal | Green | `#2ed573` |

---

## Gradio Theme

- **Base**: `gr.themes.Base`
- **Primary hue**: Custom teal scale (c500=`#00c9a7`)
- **Secondary hue**: Custom blue scale (c500=`#0088cc`)
- **Neutral hue**: Custom dark scale (c950=`#0f1117`)
- **Font**: Inter, system-ui, sans-serif
- All purple/violet tones explicitly overridden via CSS

---

## CSS Animations

| Animation | Target | Duration | Effect |
|-----------|--------|----------|--------|
| `chartFadeUp` | Plot containers | 0.7s | Fade + slide up |
| `chartScaleIn` | Radar charts | -- | Fade + scale |
| `gaugePopIn` | Gauge indicators | 0.8s | Pop-in with overshoot |
| `tableSlideIn` | Data tables | 0.5s | Slide-in |

- Hover lift on chart cards with teal shadow
- Staggered animation delays for sequential chart appearance

---

## Data Flow

```
User selects Task + Policy + Seed
         |
         v
  run_full_episode()
         |
         v
  MarketSurveillanceEnvironment.reset(task, seed)
         |
         +--> AMMState created (reserve_x=1000, reserve_y=100000)
         |
         v
  Loop while not done:
    |
    +--> Policy selects action (Heuristic/Always Allow/Random)
    |
    +--> env.step(SurveillanceAction(action_type=action))
    |      |
    |      +--> Compute reward for action
    |      +--> Update AMM state
    |      +--> Generate next observation
    |      +--> Record: action, label, reward, AMM state, signals
    |
    +--> EpisodeState accumulates all step data
         |
         v
  env.grade() --> {score, detection, false_positive, false_negative, health, overblocking}
         |
         v
  9 chart builders generate visualizations
         |
         v
  All outputs returned to Gradio UI components
```

---

## Visualization Summary

| # | Name | Chart Type | Location | Data Source |
|---|------|-----------|----------|-------------|
| 1 | AMM Final State Gauges | 4x Indicator/Gauge | Episode Runner | Final AMM state |
| 2 | Reward Timeline | Dual-axis bar + line | Episode Runner | Per-step + cumulative rewards |
| 3 | Action Distribution | Stacked bar | Episode Runner | Action counts by ground truth |
| 4 | Signal Heatmap | Heatmap | Episode Runner | 6 signals x N steps |
| 5 | AMM State Evolution | 2-row subplot (5 lines) | Episode Runner | Price, liquidity, bot_conf, volatility, health |
| 6 | Grade Radar | Radar/Polar | Episode Runner | 5 grading component scores |
| 7 | Confusion Matrix | Heatmap (2x4) | Episode Runner | Actions vs true labels |
| 8 | Step Log | Data table | Episode Runner | Per-step detailed log |
| 9 | Policy Comparison | Grouped bar chart | Policy Comparison | 3 policies x 6 metrics |
| 10 | Telemetry Replay | Bar chart | Telemetry Viewer | Rewards from uploaded .jsonl |

All charts use:
- Dark theme with transparent backgrounds (`paper_bgcolor="rgba(0,0,0,0)"`)
- Custom color palette (no purple)
- Plotly animation frames for entrance animations
- Consistent typography (Inter font, 10-16px sizes)
- Hover templates with formatted values
