# AI-Assisted Aircraft Taxiing Route Optimization

An AI-powered simulation system for autonomous aircraft ground movement at airports. Models the taxiway network of Ahmedabad International Airport (VAAH) as a weighted directed graph and uses Dijkstra's algorithm with a dynamic cost function to compute optimal taxi routes — reducing taxi time by 25–55% over standard ATC routing.

> **Indian Patent Granted** — Application No. 202511036424  
> **Paper submitted** to Springer Scopus-indexed ICCIS 2026, BITS Pilani Goa  
> **Preprint published** on Research Square (under peer review)

---

## What It Does

- Models Ahmedabad Airport taxiway network as a **weighted directed graph** (33 nodes, 58 edges) based on official AAI aerodrome chart data
- Implements a **dynamic cost function** adjusting route weights for real-time weather (clear/rain/fog) and traffic congestion
- Uses **Dijkstra's algorithm** with a smart runway exit selection module — evaluates all available exits to find the optimal starting point for each gate assignment
- Validates across **4 operational scenarios** and **11 gate destinations**
- Automatically **reroutes around blocked taxiways** (baseline ATC routes fail; AI finds alternatives)
- Estimates **annual CO₂ savings of 997–3,482 tonnes** at a single airport

---

## Algorithm & System Design

| Component | Details |
|---|---|
| Graph Model | Weighted Directed Graph (NetworkX) |
| Routing Algorithm | Dijkstra's Algorithm |
| Cost Function | Base time × weather multiplier + congestion penalty |
| Exit Selection | Smart module evaluates all runway exits per gate |
| Weather Conditions | Clear (×1.0), Rain (×1.4), Dense Fog (×2.0) |
| Congestion Levels | None / Light (+15s) / Medium (+40s) / Heavy (+80s) |

---

## Key Results

| Scenario | Avg Improvement | Max Improvement |
|---|---|---|
| Clear Weather | 25–35% | 55% |
| Rain + Traffic | 30–40% | ~55% |
| Dense Fog + Congestion | 35–50% | ~55% |
| Blocked Taxiway | AI reroutes; ATC fails | — |

---

## Tech Stack

- Python 3.x
- NetworkX (graph modeling + Dijkstra)
- Matplotlib (visualization)
- NumPy (numerical computation)

---

## Project Structure

```
TAXIING_SIMULATION/
├── airport_graph.py          # Airport taxiway network graph model (VAAH)
├── routing.py                # Dynamic cost function + Dijkstra routing engine
├── run_simulation.py         # Full 4-scenario simulation with chart generation
├── enhanced_simulation.py    # Smart exit selection + emissions analysis
├── airport_layout.png        # Airport taxiway network visualization
├── map_s1_clear.png          # Route map: clear weather scenario
├── map_s4_blocked.png        # Route map: blocked taxiway rerouting
├── map_smart_ai_route.png    # Best AI route visualization
├── map_standard_atc_route.png# Standard ATC route for comparison
├── chart1_time_comparison.png
├── chart2_improvement_pct.png
├── chart3_distance_comparison.png
├── chart4_per_case_s1.png
├── chart5_weather_impact.png
├── chart6_smart_exit_clear.png
├── chart7_smart_exit_improvement.png
├── chart8_fuel_emissions.png
├── chart9_blocked_resilience.png
└── chart10_distance_saved.png
```

---

## Setup & Run

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/taxiing-simulation.git
cd taxiing-simulation

# Install dependencies
pip install networkx matplotlib numpy

# Run basic routing test
python routing.py

# Run full 4-scenario simulation + generate all charts
python run_simulation.py

# Run enhanced simulation with smart exit selection
python enhanced_simulation.py
```

---

## Environmental Impact

Based on ICAO Engine Emissions Databank (narrow-body aircraft, idle taxi):
- Fuel flow: ~11 kg/min per aircraft
- CO₂ factor: 3.16 kg per kg fuel
- At 80 arrivals/day: **997–3,482 tonnes CO₂ saved annually** per airport

---

## Research & Patent

- **Patent**: Indian Patent Application No. 202511036424 (Granted)
- **Conference**: ICCIS 2026, BITS Pilani Goa (Springer, Scopus-indexed)
- **Preprint**: Published on Research Square

---

## Authors

- Eashan Nananya
- Apeksha Dongre
- Pooja Gupta

Manipal University Jaipur — B.Tech CSE (Data Science), Semester VI
