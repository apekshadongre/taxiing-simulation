"""
Full Scenario Simulation
=========================
Runs 4 operational scenarios across multiple origin-destination pairs.
Compares AI-optimized routes vs standard ATC baseline routes.
Generates all charts and data needed for the conference paper.

Scenarios:
  1. Clear weather, no congestion (baseline conditions)
  2. Rain with moderate traffic
  3. Dense fog with heavy congestion
  4. Blocked taxiway (taxiway segment closed)
"""

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import json
from airport_graph import create_ahmedabad_airport, visualize_airport
from routing import (compute_dynamic_cost, find_optimal_route,
                     get_baseline_route, compute_route_cost)


# ================================================================
# DEFINE TEST CASES (origin-destination pairs)
# ================================================================

# We test multiple realistic combinations:
# Aircraft landing on RWY 23 (most common at Ahmedabad, 70% of operations)
# exiting at different points, going to different gates

TEST_CASES = [
    # (runway_exit, gate, description)
    ('RWY_C', 'STAND_70', 'RWY23 Exit C → Stand 70 (Domestic West)'),
    ('RWY_C', 'STAND_80', 'RWY23 Exit C → Stand 80 (Domestic Center)'),
    ('RWY_C', 'STAND_90', 'RWY23 Exit C → Stand 90 (Domestic East)'),
    ('RWY_C', 'STAND_45', 'RWY23 Exit C → Stand 45 (International)'),
    ('RWY_C', 'STAND_28', 'RWY23 Exit C → Stand 28 (Apron 3)'),
    ('RWY_D', 'STAND_70', 'RWY23 Exit D → Stand 70 (Domestic West)'),
    ('RWY_D', 'STAND_80', 'RWY23 Exit D → Stand 80 (Domestic Center)'),
    ('RWY_D', 'STAND_90', 'RWY23 Exit D → Stand 90 (Domestic East)'),
    ('RWY_D', 'STAND_45', 'RWY23 Exit D → Stand 45 (International)'),
    ('RWY_D', 'STAND_28', 'RWY23 Exit D → Stand 28 (Apron 3)'),
]


# ================================================================
# DEFINE SCENARIOS
# ================================================================

SCENARIOS = {
    'S1_Clear': {
        'name': 'Scenario 1: Clear Weather',
        'weather': 'clear',
        'congested_edges': {},
        'blocked_edges': [],
        'description': 'Normal visibility, no traffic congestion, all taxiways open',
    },
    'S2_Rain': {
        'name': 'Scenario 2: Rain + Moderate Traffic',
        'weather': 'rain',
        'congested_edges': {
            ('P_C', 'P_K'): 'medium',
            ('P_K', 'P_C'): 'medium',
            ('P_K', 'P_D'): 'light',
            ('P_D', 'P_K'): 'light',
        },
        'blocked_edges': [],
        'description': 'Rain (RVR 200-550m), moderate traffic on main taxiway P',
    },
    'S3_Fog': {
        'name': 'Scenario 3: Dense Fog + Heavy Congestion',
        'weather': 'fog',
        'congested_edges': {
            ('P_C', 'P_K'): 'heavy',
            ('P_K', 'P_C'): 'heavy',
            ('P_K', 'P_D'): 'heavy',
            ('P_D', 'P_K'): 'heavy',
            ('P_F', 'P_C'): 'medium',
            ('P_C', 'P_F'): 'medium',
        },
        'blocked_edges': [],
        'description': 'Dense fog (RVR <200m), heavy congestion on central taxiway',
    },
    'S4_Blocked': {
        'name': 'Scenario 4: Blocked Taxiway',
        'weather': 'clear',
        'congested_edges': {
            ('P_C', 'P_K'): 'light',
            ('P_K', 'P_C'): 'light',
        },
        'blocked_edges': [
            ('P_F', 'APRON1_WEST'),    # Apron 1 west entry blocked
        ],
        'description': 'Clear weather, but Apron 1 west entry blocked (ground vehicle incident)',
    },
}


# ================================================================
# RUN ALL SIMULATIONS
# ================================================================

def run_all_simulations():
    """
    Runs every test case under every scenario.
    Returns a structured results dictionary.
    """
    
    G = create_ahmedabad_airport()
    all_results = {}
    
    for scenario_id, scenario in SCENARIOS.items():
        print(f"\n{'='*70}")
        print(f"  {scenario['name']}")
        print(f"  {scenario['description']}")
        print(f"{'='*70}")
        
        # Compute dynamic costs for this scenario
        G_dyn = compute_dynamic_cost(
            G,
            weather=scenario['weather'],
            congested_edges=scenario['congested_edges'],
            blocked_edges=scenario['blocked_edges'],
        )
        
        scenario_results = []
        
        for (source, dest, desc) in TEST_CASES:
            # AI-optimized route
            ai_result = find_optimal_route(G_dyn, source, dest)
            
            # Baseline route
            baseline_path = get_baseline_route(source, dest)
            if baseline_path:
                baseline_cost = compute_route_cost(G_dyn, baseline_path)
            else:
                baseline_cost = {'total_cost_s': float('inf'), 
                                'total_distance_m': 0, 'total_time_min': float('inf')}
            
            # Compute improvement
            if (baseline_cost['total_cost_s'] > 0 and 
                ai_result['total_cost_s'] < float('inf') and
                baseline_cost['total_cost_s'] < float('inf')):
                time_saved = baseline_cost['total_cost_s'] - ai_result['total_cost_s']
                pct_improvement = (time_saved / baseline_cost['total_cost_s']) * 100
                dist_saved = baseline_cost['total_distance_m'] - ai_result['total_distance_m']
            else:
                time_saved = 0
                pct_improvement = 0
                dist_saved = 0
            
            case_result = {
                'source': source,
                'destination': dest,
                'description': desc,
                'ai_path': ai_result.get('path'),
                'ai_time_s': ai_result.get('total_cost_s', float('inf')),
                'ai_distance_m': ai_result.get('total_distance_m', 0),
                'ai_time_min': ai_result.get('total_time_min', float('inf')),
                'baseline_path': baseline_path,
                'baseline_time_s': baseline_cost.get('total_cost_s', float('inf')),
                'baseline_distance_m': baseline_cost.get('total_distance_m', 0),
                'baseline_time_min': baseline_cost.get('total_time_min', float('inf')),
                'time_saved_s': round(time_saved, 1),
                'distance_saved_m': dist_saved,
                'pct_improvement': round(pct_improvement, 1),
            }
            
            scenario_results.append(case_result)
            
            # Print summary for this case
            status = "✓" if ai_result.get('path') else "✗"
            print(f"  {status} {desc}")
            print(f"      AI: {ai_result.get('total_cost_s', 'N/A')}s ({ai_result.get('total_distance_m', 0)}m) | "
                  f"Baseline: {baseline_cost.get('total_cost_s', 'N/A')}s ({baseline_cost.get('total_distance_m', 0)}m) | "
                  f"Saved: {time_saved:.1f}s ({pct_improvement:.1f}%)")
        
        all_results[scenario_id] = scenario_results
    
    return all_results


# ================================================================
# GENERATE CHARTS
# ================================================================

def generate_charts(all_results):
    """Generates all figures needed for the conference paper."""
    
    # ---- CHART 1: Average time comparison across scenarios ----
    fig, ax = plt.subplots(figsize=(10, 6))
    
    scenario_names = []
    avg_ai_times = []
    avg_baseline_times = []
    
    for scenario_id, results in all_results.items():
        scenario_names.append(SCENARIOS[scenario_id]['name'].replace('Scenario ', 'S'))
        
        ai_times = [r['ai_time_s'] for r in results if r['ai_time_s'] < float('inf')]
        bl_times = [r['baseline_time_s'] for r in results if r['baseline_time_s'] < float('inf')]
        
        avg_ai_times.append(np.mean(ai_times) if ai_times else 0)
        avg_baseline_times.append(np.mean(bl_times) if bl_times else 0)
    
    x = np.arange(len(scenario_names))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, avg_baseline_times, width, label='Standard ATC Route',
                   color='#E74C3C', alpha=0.85)
    bars2 = ax.bar(x + width/2, avg_ai_times, width, label='AI-Optimized Route',
                   color='#2ECC71', alpha=0.85)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 5,
                f'{height:.0f}s', ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 5,
                f'{height:.0f}s', ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Operational Scenario', fontsize=11)
    ax.set_ylabel('Average Taxi Time (seconds)', fontsize=11)
    ax.set_title('Average Taxi Time: AI-Optimized vs Standard ATC Route', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(scenario_names, fontsize=9)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.2)
    
    plt.tight_layout()
    plt.savefig('chart1_time_comparison.png', dpi=300, bbox_inches='tight')
    print("Saved: chart1_time_comparison.png")
    plt.close()
    
    
    # ---- CHART 2: Percentage improvement per scenario ----
    fig, ax = plt.subplots(figsize=(10, 6))
    
    avg_improvements = []
    for scenario_id, results in all_results.items():
        improvements = [r['pct_improvement'] for r in results if r['pct_improvement'] > 0]
        avg_improvements.append(np.mean(improvements) if improvements else 0)
    
    bars = ax.bar(scenario_names, avg_improvements, color=['#3498DB', '#F39C12', '#E74C3C', '#9B59B6'],
                  alpha=0.85)
    
    for bar, val in zip(bars, avg_improvements):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.3,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.set_xlabel('Operational Scenario', fontsize=11)
    ax.set_ylabel('Average Time Improvement (%)', fontsize=11)
    ax.set_title('Route Optimization Improvement Across Scenarios', fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.2)
    
    plt.tight_layout()
    plt.savefig('chart2_improvement_pct.png', dpi=300, bbox_inches='tight')
    print("Saved: chart2_improvement_pct.png")
    plt.close()
    
    
    # ---- CHART 3: Distance comparison ----
    fig, ax = plt.subplots(figsize=(10, 6))
    
    avg_ai_dist = []
    avg_bl_dist = []
    
    for scenario_id, results in all_results.items():
        ai_d = [r['ai_distance_m'] for r in results if r['ai_distance_m'] > 0]
        bl_d = [r['baseline_distance_m'] for r in results if r['baseline_distance_m'] > 0]
        avg_ai_dist.append(np.mean(ai_d) if ai_d else 0)
        avg_bl_dist.append(np.mean(bl_d) if bl_d else 0)
    
    bars1 = ax.bar(x - width/2, avg_bl_dist, width, label='Standard ATC Route',
                   color='#E74C3C', alpha=0.85)
    bars2 = ax.bar(x + width/2, avg_ai_dist, width, label='AI-Optimized Route',
                   color='#2ECC71', alpha=0.85)
    
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 10,
                f'{height:.0f}m', ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 10,
                f'{height:.0f}m', ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Operational Scenario', fontsize=11)
    ax.set_ylabel('Average Taxi Distance (meters)', fontsize=11)
    ax.set_title('Average Taxi Distance: AI-Optimized vs Standard ATC Route', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(scenario_names, fontsize=9)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.2)
    
    plt.tight_layout()
    plt.savefig('chart3_distance_comparison.png', dpi=300, bbox_inches='tight')
    print("Saved: chart3_distance_comparison.png")
    plt.close()
    
    
    # ---- CHART 4: Per-case breakdown for Scenario 1 (detailed) ----
    fig, ax = plt.subplots(figsize=(14, 6))
    
    s1_results = all_results['S1_Clear']
    case_labels = [f"{r['source'].replace('RWY_','')}→{r['destination'].replace('STAND_','S')}" 
                   for r in s1_results]
    ai_t = [r['ai_time_s'] for r in s1_results]
    bl_t = [r['baseline_time_s'] for r in s1_results]
    
    x2 = np.arange(len(case_labels))
    
    ax.bar(x2 - width/2, bl_t, width, label='Standard ATC Route', color='#E74C3C', alpha=0.85)
    ax.bar(x2 + width/2, ai_t, width, label='AI-Optimized Route', color='#2ECC71', alpha=0.85)
    
    ax.set_xlabel('Origin → Destination', fontsize=11)
    ax.set_ylabel('Taxi Time (seconds)', fontsize=11)
    ax.set_title('Per-Route Time Comparison — Clear Weather (Scenario 1)', fontsize=13, fontweight='bold')
    ax.set_xticks(x2)
    ax.set_xticklabels(case_labels, fontsize=8, rotation=25, ha='right')
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.2)
    
    plt.tight_layout()
    plt.savefig('chart4_per_case_s1.png', dpi=300, bbox_inches='tight')
    print("Saved: chart4_per_case_s1.png")
    plt.close()


    # ---- CHART 5: Weather impact on taxi time ----
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Pick one representative route: RWY_D -> STAND_80
    target_src, target_dst = 'RWY_D', 'STAND_80'
    
    weather_labels = []
    ai_times_weather = []
    bl_times_weather = []
    
    for scenario_id in ['S1_Clear', 'S2_Rain', 'S3_Fog']:
        results = all_results[scenario_id]
        for r in results:
            if r['source'] == target_src and r['destination'] == target_dst:
                weather_labels.append(SCENARIOS[scenario_id]['name'].replace('Scenario ', 'S'))
                ai_times_weather.append(r['ai_time_s'])
                bl_times_weather.append(r['baseline_time_s'])
                break
    
    x3 = np.arange(len(weather_labels))
    
    ax.bar(x3 - width/2, bl_times_weather, width, label='Standard ATC Route',
           color='#E74C3C', alpha=0.85)
    ax.bar(x3 + width/2, ai_times_weather, width, label='AI-Optimized Route',
           color='#2ECC71', alpha=0.85)
    
    for i, (bl_val, ai_val) in enumerate(zip(bl_times_weather, ai_times_weather)):
        ax.text(i - width/2, bl_val + 5, f'{bl_val:.0f}s', ha='center', fontsize=9)
        ax.text(i + width/2, ai_val + 5, f'{ai_val:.0f}s', ha='center', fontsize=9)
    
    ax.set_xlabel('Weather Condition', fontsize=11)
    ax.set_ylabel('Taxi Time (seconds)', fontsize=11)
    ax.set_title(f'Weather Impact on Taxi Time ({target_src} → {target_dst})', fontsize=13, fontweight='bold')
    ax.set_xticks(x3)
    ax.set_xticklabels(weather_labels, fontsize=10)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.2)
    
    plt.tight_layout()
    plt.savefig('chart5_weather_impact.png', dpi=300, bbox_inches='tight')
    print("Saved: chart5_weather_impact.png")
    plt.close()


# ================================================================
# GENERATE ROUTE VISUALIZATIONS FOR KEY SCENARIOS
# ================================================================

def generate_route_maps(all_results):
    """Generates airport map visualizations with highlighted routes."""
    
    G = create_ahmedabad_airport()
    
    # Map 1: Clear weather optimal route
    s1 = all_results['S1_Clear']
    for r in s1:
        if r['source'] == 'RWY_D' and r['destination'] == 'STAND_80':
            visualize_airport(G,
                title="Scenario 1: AI-Optimized Route (Clear Weather)\nRWY_D → STAND_80",
                highlighted_path=r['ai_path'],
                filename="map_s1_clear.png")
            break
    
    # Map 2: Blocked taxiway — system reroutes
    s4 = all_results['S4_Blocked']
    for r in s4:
        if r['source'] == 'RWY_C' and r['destination'] == 'STAND_70':
            visualize_airport(G,
                title="Scenario 4: AI Reroute (Apron 1 West Entry Blocked)\nRWY_C → STAND_70",
                highlighted_path=r['ai_path'],
                filename="map_s4_blocked.png")
            break


# ================================================================
# PRINT SUMMARY TABLE
# ================================================================

def print_summary_table(all_results):
    """Prints a formatted summary table of all results."""
    
    print("\n" + "="*90)
    print("  COMPLETE RESULTS SUMMARY")
    print("="*90)
    
    for scenario_id, results in all_results.items():
        scenario = SCENARIOS[scenario_id]
        print(f"\n{'─'*90}")
        print(f"  {scenario['name']}")
        print(f"  {scenario['description']}")
        print(f"{'─'*90}")
        print(f"  {'Route':<35} {'AI Time':>10} {'ATC Time':>10} {'Saved':>10} {'Improv.':>10}")
        print(f"  {'─'*75}")
        
        total_ai = 0
        total_bl = 0
        count = 0
        
        for r in results:
            label = f"{r['source'].replace('RWY_','')} → {r['destination'].replace('STAND_','S')}"
            ai_t = f"{r['ai_time_s']:.0f}s" if r['ai_time_s'] < float('inf') else "N/A"
            bl_t = f"{r['baseline_time_s']:.0f}s" if r['baseline_time_s'] < float('inf') else "N/A"
            saved = f"{r['time_saved_s']:.0f}s" if r['time_saved_s'] > 0 else "0s"
            pct = f"{r['pct_improvement']:.1f}%" if r['pct_improvement'] > 0 else "0.0%"
            
            print(f"  {label:<35} {ai_t:>10} {bl_t:>10} {saved:>10} {pct:>10}")
            
            if r['ai_time_s'] < float('inf') and r['baseline_time_s'] < float('inf'):
                total_ai += r['ai_time_s']
                total_bl += r['baseline_time_s']
                count += 1
        
        if count > 0:
            avg_ai = total_ai / count
            avg_bl = total_bl / count
            avg_saved = avg_bl - avg_ai
            avg_pct = (avg_saved / avg_bl) * 100
            print(f"  {'─'*75}")
            print(f"  {'AVERAGE':<35} {avg_ai:>9.0f}s {avg_bl:>9.0f}s {avg_saved:>9.0f}s {avg_pct:>9.1f}%")
    
    print(f"\n{'='*90}")


# ================================================================
# FUEL AND EMISSIONS ESTIMATION
# ================================================================

def estimate_fuel_and_emissions(all_results):
    """
    Estimates fuel savings and emission reductions.
    
    Assumptions (from ICAO Engine Emissions Databank):
    - Average taxi fuel flow: 10-12 kg/min for narrow-body aircraft (A320/B737)
    - CO2 emission factor: 3.16 kg CO2 per kg of jet fuel
    - NOx emission factor: 0.014 kg NOx per kg of jet fuel
    - We use 11 kg/min as average
    """
    
    FUEL_FLOW_KG_PER_MIN = 11.0    # kg/min for twin-engine idle
    CO2_FACTOR = 3.16              # kg CO2 per kg fuel
    NOX_FACTOR = 0.014             # kg NOx per kg fuel
    
    print("\n" + "="*90)
    print("  FUEL CONSUMPTION AND EMISSIONS ESTIMATES")
    print("="*90)
    print(f"  Assumptions: Fuel flow = {FUEL_FLOW_KG_PER_MIN} kg/min | "
          f"CO2 = {CO2_FACTOR} kg/kg fuel | NOx = {NOX_FACTOR} kg/kg fuel")
    print(f"{'─'*90}")
    
    for scenario_id, results in all_results.items():
        scenario = SCENARIOS[scenario_id]
        
        total_time_saved_s = 0
        count = 0
        
        for r in results:
            if r['time_saved_s'] > 0:
                total_time_saved_s += r['time_saved_s']
                count += 1
        
        if count > 0:
            avg_time_saved_min = (total_time_saved_s / count) / 60
            fuel_saved_kg = avg_time_saved_min * FUEL_FLOW_KG_PER_MIN
            co2_saved_kg = fuel_saved_kg * CO2_FACTOR
            nox_saved_kg = fuel_saved_kg * NOX_FACTOR
            
            # Scale to daily operations (assume 80 arrivals per day at AMD)
            daily_fuel_saved = fuel_saved_kg * 80
            daily_co2_saved = co2_saved_kg * 80
            annual_fuel_saved = daily_fuel_saved * 365
            annual_co2_saved = daily_co2_saved * 365
            
            print(f"\n  {scenario['name']}:")
            print(f"    Avg time saved per flight:    {avg_time_saved_min:.2f} min")
            print(f"    Fuel saved per flight:        {fuel_saved_kg:.2f} kg")
            print(f"    CO2 reduced per flight:       {co2_saved_kg:.2f} kg")
            print(f"    NOx reduced per flight:       {nox_saved_kg:.4f} kg")
            print(f"    --- Scaled to 80 arrivals/day ---")
            print(f"    Daily fuel saved:             {daily_fuel_saved:.0f} kg ({daily_fuel_saved/1000:.1f} tonnes)")
            print(f"    Daily CO2 reduced:            {daily_co2_saved:.0f} kg ({daily_co2_saved/1000:.1f} tonnes)")
            print(f"    Annual fuel saved:            {annual_fuel_saved/1000:.0f} tonnes")
            print(f"    Annual CO2 reduced:           {annual_co2_saved/1000:.0f} tonnes")
    
    print(f"\n{'='*90}")


# ================================================================
# MAIN
# ================================================================

if __name__ == "__main__":
    
    print("="*70)
    print("  AIRPORT TAXIING ROUTE OPTIMIZATION - FULL SIMULATION")
    print("  Ahmedabad International Airport (VAAH)")
    print("="*70)
    
    # Run all simulations
    all_results = run_all_simulations()
    
    # Print complete summary table
    print_summary_table(all_results)
    
    # Estimate fuel and emissions
    estimate_fuel_and_emissions(all_results)
    
    # Generate all charts
    print("\nGenerating charts...")
    generate_charts(all_results)
    
    # Generate route maps
    print("\nGenerating route maps...")
    generate_route_maps(all_results)
    
    print("\n" + "="*70)
    print("  SIMULATION COMPLETE")
    print("  All charts and maps saved in current directory")
    print("="*70)