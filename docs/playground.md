# Playground Documentation

The TradeX Playground is the interactive OpenEnv web interface that allows users to manually interact with the `MarketSurveillanceEnvironment` by sending actions step-by-step, observing the environment's responses, and grading episodes in real time.

---

## Overview

The Playground is **not custom code** in this repository. It is provided by the `openenv-core` library's web interface module and is dynamically integrated into the TradeX application when running in Hugging Face Space mode.

### Location in Application

The Playground is built in `app.py` at lines 40-80 via the `_build_space_app()` function. It is conditionally imported and wrapped alongside the Dashboard in a tabbed Gradio interface.

---

## Application Modes

| Mode | Playground Available | Behavior |
|------|---------------------|----------|
| **HF Space** (`SPACE_ID` detected) | Yes (if `openenv-core` web interface imports successfully) | Combined tabbed app: "Playground" + "Dashboard" |
| **HF Space** (import fails) | No | Falls back to dashboard-only |
| **OpenEnv** (default local) | Yes (served via FastAPI) | OpenEnv server at configured port |
| **Standalone** (`python dashboard.py`) | No | Gradio dashboard only |

---

## How the Playground is Built

### Import Chain

The Playground relies on the following imports from `openenv-core`:

```python
from openenv.core.env_server.web_interface import (
    OPENENV_GRADIO_CSS,
    OPENENV_GRADIO_THEME,
    WebInterfaceManager,
    _extract_action_fields,
    _is_chat_env,
    build_gradio_app,
    get_gradio_display_title,
    get_quick_start_markdown,
    load_environment_metadata,
)
```

### Construction Steps

1. **Load metadata**: `load_environment_metadata(MarketSurveillanceEnvironment, "amm-market-surveillance")` reads `openenv.yaml` to get task definitions, descriptions, and display configuration.

2. **Create web manager**: `WebInterfaceManager(MarketSurveillanceEnvironment, SurveillanceAction, SurveillanceObservation, metadata)` initializes the interface manager that bridges the environment with the Gradio UI.

3. **Extract action fields**: `_extract_action_fields(SurveillanceAction)` inspects the Pydantic model to auto-generate input forms for the `action_type` field.

4. **Determine UI type**: `_is_chat_env(SurveillanceAction)` checks if the action model uses a chat-style interface (it does not for TradeX).

5. **Generate quick-start docs**: `get_quick_start_markdown(metadata, SurveillanceAction, SurveillanceObservation)` produces introductory documentation from the metadata and model schemas.

6. **Build the app**: `build_gradio_app(web_manager, action_fields, metadata, is_chat_env, title, quick_start_md)` constructs the full Gradio Blocks interface.

---

## Playground UI Elements

The Playground auto-generates its UI based on the `SurveillanceAction` and `SurveillanceObservation` Pydantic models. Here is what it provides:

### Action Input

| Element | Type | Source | Purpose |
|---------|------|--------|---------|
| **Action Type** | Dropdown/Selector | `SurveillanceAction.action_type` | Select one of: `ALLOW`, `FLAG`, `BLOCK`, `MONITOR` |

The action input is auto-generated from the `SurveillanceAction` Pydantic model:

```python
class SurveillanceAction(Action):
    action_type: SurveillanceActionType = Field(
        ...,
        description="Final surveillance response to the current market activity.",
    )
```

Where `SurveillanceActionType = Literal["ALLOW", "FLAG", "BLOCK", "MONITOR"]`.

### Observation Display

The Playground displays the full `SurveillanceObservation` returned by the environment after each step. This includes all 16+ fields:

| Field | Type | What it shows |
|-------|------|---------------|
| `current_amm_price` | float | Current AMM price (`reserve_y / reserve_x`) |
| `liquidity_snapshot` | float | Pool liquidity (`2 * sqrt(k)`) |
| `recent_trade_count` | int | Number of trades in the current window |
| `trades_in_window` | list[float] | Last 5 trade sizes |
| `trade_frequency` | float | Trades per unit time |
| `average_trade_size` | float | Mean trade size |
| `maximum_trade_size` | float | Largest single trade |
| `recent_slippage_impact` | float | Aggregate price displacement |
| `time_gap_mean` | float | Average time between trades |
| `time_gap_min` | float | Shortest time gap |
| `recent_time_gaps` | list[float] | Last 5 inter-trade intervals |
| `recent_price_impacts` | list[float] | Last 5 per-trade price impacts |
| `burst_indicator` | float | High-frequency burst score (0.0-1.0) |
| `pattern_indicator` | float | Rhythmic coordination score (0.0-1.0) |
| `suspiciousness_score` | float | Composite suspicion metric (0.0-1.0) |
| `manipulation_score` | float | Deliberate manipulation confidence (0.0-1.0) |
| `step_num` | int | Current step number |
| `max_steps` | int | Total steps in the episode |
| `task_name` | str | Name of the active task |

### Controls

| Control | Purpose |
|---------|---------|
| **Reset** | Starts a new episode with a fresh environment state |
| **Step** | Sends the selected action to the environment and receives the next observation |
| **Grade** | Ends the episode and computes the final score breakdown |
| **Task Selector** | Chooses which task to run (`burst_detection`, `pattern_manipulation_detection`, `full_market_surveillance`) |

### Quick-Start Documentation

The Playground includes auto-generated markdown documentation that explains:
- What the environment does
- What actions are available
- What observations mean
- How to interact with the environment

This is generated from the `openenv.yaml` metadata and the Pydantic model field descriptions.

---

## Data Flow

```
User opens Playground tab
         |
         v
  WebInterfaceManager initializes
         |
         +--> Loads environment metadata from openenv.yaml
         +--> Extracts action fields from SurveillanceAction model
         +--> Generates quick-start markdown
         |
         v
  User selects Task
         |
         v
  User clicks "Reset"
         |
         v
  MarketSurveillanceEnvironment.reset(task, seed)
         |
         +--> Creates AMMState (reserve_x=1000, reserve_y=100000)
         +--> Sets bot_confidence based on task
         +--> Returns initial SurveillanceObservation
         |
         v
  Observation displayed in Playground UI
         |
         v
  User selects Action Type (ALLOW/FLAG/BLOCK/MONITOR)
         |
         v
  User clicks "Step"
         |
         v
  env.step(SurveillanceAction(action_type=selected_action))
         |
         +--> Computes reward for action
         +--> Updates AMM state (reserves, bot_confidence, volatility, health)
         +--> Generates next observation (procedural trading window)
         +--> Returns new SurveillanceObservation
         |
         v
  New observation displayed in Playground UI
         |
         v
  (Repeat Step until done=True)
         |
         v
  User clicks "Grade" (or episode ends naturally)
         |
         v
  env.grade() --> {score, detection, false_positive, false_negative, health, overblocking}
         |
         v
  Grade breakdown displayed in Playground UI
```

---

## HF Space Integration

When running as a Hugging Face Space, the Playground and Dashboard are combined into a single Gradio app with tabs:

```
┌─────────────────────────────────────────────────┐
│  TradeX Surveillance Dashboard                  │
├─────────────────────────────────────────────────┤
│  [ Playground ]  [ Dashboard ]                  │  <- gr.Tabs
├─────────────────────────────────────────────────┤
│                                                 │
│  Playground tab content                         │
│  - Action input (auto-generated from model)     │
│  - Observation display                          │
│  - Reset / Step / Grade controls                │
│  - Quick-start documentation                    │
│                                                 │
└─────────────────────────────────────────────────┘
```

### Styling

The combined app uses merged CSS from both sources:
- **OpenEnv CSS**: `OPENENV_GRADIO_CSS` from `openenv-core`
- **Dashboard CSS**: Custom dark theme animations from `dashboard.py`
- **Space CSS**: Custom tab styling (transparent background, bordered tab nav, bold tab labels)

### Theme Resolution

The theme priority is:
1. OpenEnv theme (if available)
2. Dashboard theme (fallback)

---

## OpenEnv Server Mode

When running in non-Space mode (default local), the Playground is served through the FastAPI OpenEnv server:

```bash
python app.py
```

This serves the environment at the configured port (default 7860) with the OpenEnv web interface accessible via browser. The interface provides the same functionality but is served through a different stack (FastAPI + OpenEnv web UI vs. Gradio Blocks).

---

## Environment Metadata

The Playground uses metadata from `openenv.yaml` for display and configuration:

```yaml
spec_version: 1
name: amm-market-surveillance
type: space
runtime: fastapi
app: app:app
port: 7860
HF: true

metadata:
  title: "Bot-aware Market Surveillance in Simulated AMM Trading"
  description: "An OpenEnv reinforcement learning benchmark where agents monitor AMM-style market activity and choose ALLOW, FLAG, BLOCK, or MONITOR responses to suspicious bot-like behavior."
  tags:
    - openenv
    - reinforcement-learning
    - market-surveillance
    - anomaly-detection
    - adversarial-monitoring

tasks:
  - name: burst_detection
    description: "Task 1 - Burst Detection. Catch abrupt high-frequency suspicious bursts while minimizing harm to normal traffic."
    difficulty: easy
  - name: pattern_manipulation_detection
    description: "Task 2 - Pattern-based Manipulation Detection. Detect repeated timing and size signatures that imply coordinated manipulation."
    difficulty: medium
  - name: full_market_surveillance
    description: "Task 3 - Full Market Surveillance. Balance burst, pattern, and false-positive control across mixed activity."
    difficulty: hard
```

This metadata drives:
- The display title shown in the Playground header
- The task selector dropdown options
- The quick-start documentation content
- Tag-based categorization on Hugging Face Spaces

---

## Action Model Details

The `SurveillanceAction` model defines what the user can do in the Playground:

```python
class SurveillanceAction(Action):
    action_type: SurveillanceActionType = Field(
        ...,
        description="Final surveillance response to the current market activity.",
    )
```

The model includes a validator that normalizes input:
- Strips whitespace from action strings
- Converts to uppercase
- Handles JSON-encoded params if present

This ensures robust input handling regardless of how the action is submitted.

---

## Observation Model Details

The `SurveillanceObservation` model defines what the user sees after each step:

```python
class SurveillanceObservation(Observation):
    current_amm_price: float
    liquidity_snapshot: float
    recent_trade_count: int
    trades_in_window: List[float]
    trade_frequency: float
    average_trade_size: float
    maximum_trade_size: float
    recent_slippage_impact: float
    time_gap_mean: float
    time_gap_min: float
    recent_time_gaps: List[float]
    recent_price_impacts: List[float]
    burst_indicator: float
    pattern_indicator: float
    suspiciousness_score: float
    manipulation_score: float
    step_num: int
    max_steps: int
    task_name: str
```

The Playground displays all these fields so users can understand the full state of the environment after each action.

---

## Comparison: Playground vs Dashboard

| Aspect | Playground | Dashboard |
|--------|-----------|-----------|
| **Purpose** | Manual step-by-step interaction | Automated episode execution with visualizations |
| **Interaction** | User selects action each step | User selects policy, runs full episode |
| **Visualizations** | Raw observation display | 10+ charts, gauges, heatmaps, tables |
| **Use case** | Learning, exploration, debugging | Analysis, comparison, demonstration |
| **Data flow** | One step at a time | Full episode at once |
| **Grading** | Manual trigger | Automatic after episode completes |
| **Policy** | Human-driven | Heuristic/Always Allow/Random |

---

## Key Files

| File | Role |
|------|------|
| `app.py` | Integrates Playground with Dashboard in HF Space mode |
| `openenv.yaml` | Metadata that configures Playground display |
| `meverse/models.py` | Defines `SurveillanceAction` and `SurveillanceObservation` models |
| `meverse/server/meverse_environment.py` | The `MarketSurveillanceEnvironment` class that the Playground interacts with |
| `meverse/server/app.py` | FastAPI server that serves the OpenEnv interface |

---

## Dependencies

The Playground requires `openenv-core` with the web interface module. This is specified in `meverse/server/requirements.txt` and installed during the Docker build process.

If the web interface import fails (version mismatch, missing deps), the application gracefully falls back to dashboard-only mode without the Playground tab.
