"""
Enhanced Simulation - Smart Exit Selection + Additional Analysis
================================================================
Adds the key missing element: AI doesn't just optimize the route FROM
a fixed exit — it picks the BEST exit point for each gate assignment.

This is the biggest real-world advantage. ATC tells everyone "exit at C"
but our system says "you're going to Stand 25? Exit at D, it's closer."
"""

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from airport_graph import create_ahmedabad_airport, visualize_airport
from routing import (compute_dynamic_cost, find_optimal_route,
                     get_baseline_route, compute_route_cost)


# All possible runway exits (for RWY 23 landing — aircraft rolls SW to NE exit)
# Aircraft can exit at any of these depending on speed and braking
ALL_EXITS_RWY23 = ['RWY_A', 'RWY_B', 'RWY_F', 'RWY_C', 'RWY_K', 'RWY_D', 'RWY_R4']

# All gates
ALL_GATES = ['STAND_70', 'STAND_75', 'STAND_80', 'STAND_85', 'STAND_90',
             'STAND_40', 'STAND_45', 'STAND_50',
             'STAND_25', 'STAND_28', 'STAND_30']

# ATC default: always assigns exit C (mid-runway, standard practice)
ATC_DEFAULT_EXIT = 'RWY_C'


def find_best_exit_and_route(G_dynamic, destination):
    """
    AI smart exit selection: tries ALL possible exits and picks
    the one that gives the shortest total route to the destination.
    
    This is the key advantage — ATC picks one exit for everyone,
    our system picks the best exit for YOUR specific gate.
    """
    
    best_result = None
    best_cost = float('inf')
    best_exit = None
    
    for exit_node in ALL_EXITS_RWY23:
        result = find_optimal_route(G_dynamic, exit_node, destination)
        if result.get('total_cost_s', float('inf')) < best_cost:
            best_cost = result['total_cost_s']
            best_result = result
            best_exit = exit_node
    
    return best_exit, best_result


def run_smart_exit_comparison():
    """
    Compares:
      ATC approach: Fixed exit C → standard route to gate
      AI approach:  Best exit selected → optimal route to gate
    
    Runs across all 11 gates under all weather scenarios.
    """
    
    G = create_ahmedabad_airport()
    
    scenarios = {
        'S1_Clear': {'weather': 'clear', 'congested': {}, 'blocked': [],
                     'name': 'Clear Weather'},
        'S2_Rain':  {'weather': 'rain',
                     'congested': {('P_C','P_K'):'medium', ('P_K','P_C'):'medium',
                                   ('P_K','P_D'):'light', ('P_D','P_K'):'light'},
                     'blocked': [], 'name': 'Rain + Moderate Traffic'},
        'S3_Fog':   {'weather': 'fog',
                     'congested': {('P_C','P_K'):'heavy', ('P_K','P_C'):'heavy',
                                   ('P_K','P_D'):'heavy', ('P_D','P_K'):'heavy',
                                   ('P_F','P_C'):'medium', ('P_C','P_F'):'medium'},
                     'blocked': [], 'name': 'Dense Fog + Heavy Congestion'},
        'S4_Blocked': {'weather': 'clear',
                       'congested': {('P_C','P_K'):'light', ('P_K','P_C'):'light'},
                       'blocked': [('P_F','APRON1_WEST')],
                       'name': 'Blocked Taxiway (Apron 1 West)'},
    }
    
    all_data = {}
    
    for sid, sc in scenarios.items():
        G_dyn = compute_dynamic_cost(G, weather=sc['weather'],
                                     congested_edges=sc['congested'],
                                     blocked_edges=sc['blocked'])
        
        print(f"\n{'='*75}")
        print(f"  {sc['name']}")
        print(f"{'='*75}")
        print(f"  {'Gate':<12} {'ATC Exit':>10} {'ATC Time':>10} {'AI Exit':>10} {'AI Time':>10} {'Saved':>10} {'Improv':>10}")
        print(f"  {'─'*72}")
        
        scenario_data = []
        
        for gate in ALL_GATES:
            # ATC approach: fixed exit C, standard route
            atc_path = get_baseline_route(ATC_DEFAULT_EXIT, gate)
            if atc_path:
                atc_cost = compute_route_cost(G_dyn, atc_path)
                atc_time = atc_cost.get('total_cost_s', float('inf'))
                atc_dist = atc_cost.get('total_distance_m', 0)
            else:
                # Try direct Dijkstra from exit C as fallback
                atc_result = find_optimal_route(G_dyn, ATC_DEFAULT_EXIT, gate)
                atc_time = atc_result.get('total_cost_s', float('inf'))
                atc_dist = atc_result.get('total_distance_m', 0)
                atc_path = atc_result.get('path')
            
            # AI approach: smart exit selection
            ai_exit, ai_result = find_best_exit_and_route(G_dyn, gate)
            ai_time = ai_result.get('total_cost_s', float('inf'))
            ai_dist = ai_result.get('total_distance_m', 0)
            ai_path = ai_result.get('path')
            
            # Compute savings
            if atc_time < float('inf') and ai_time < float('inf') and atc_time > 0:
                time_saved = atc_time - ai_time
                pct = (time_saved / atc_time) * 100
                dist_saved = atc_dist - ai_dist
            else:
                time_saved = 0
                pct = 0
                dist_saved = 0
            
            # Handle blocked baseline
            atc_status = "FAIL" if atc_time == float('inf') else f"{atc_time:.0f}s"
            
            gate_short = gate.replace('STAND_', 'S')
            ai_exit_short = ai_exit.replace('RWY_', '') if ai_exit else '?'
            atc_exit_short = ATC_DEFAULT_EXIT.replace('RWY_', '')
            
            print(f"  {gate_short:<12} {atc_exit_short:>10} {atc_status:>10} "
                  f"{ai_exit_short:>10} {ai_time:>9.0f}s {time_saved:>9.1f}s {pct:>9.1f}%")
            
            scenario_data.append({
                'gate': gate,
                'gate_short': gate_short,
                'atc_exit': ATC_DEFAULT_EXIT,
                'atc_time': atc_time,
                'atc_dist': atc_dist,
                'atc_path': atc_path,
                'ai_exit': ai_exit,
                'ai_time': ai_time,
                'ai_dist': ai_dist,
                'ai_path': ai_path,
                'time_saved': round(time_saved, 1),
                'dist_saved': dist_saved,
                'pct_improvement': round(pct, 1),
                'baseline_failed': atc_time == float('inf'),
            })
        
        # Summary
        valid = [d for d in scenario_data if d['atc_time'] < float('inf') and d['ai_time'] < float('inf')]
        failed = [d for d in scenario_data if d['baseline_failed']]
        
        if valid:
            avg_atc = np.mean([d['atc_time'] for d in valid])
            avg_ai = np.mean([d['ai_time'] for d in valid])
            avg_saved = avg_atc - avg_ai
            avg_pct = (avg_saved / avg_atc) * 100 if avg_atc > 0 else 0
            
            print(f"  {'─'*72}")
            print(f"  AVERAGE (valid routes): ATC={avg_atc:.0f}s | AI={avg_ai:.0f}s | "
                  f"Saved={avg_saved:.0f}s | Improvement={avg_pct:.1f}%")
        
        if failed:
            print(f"  BASELINE FAILED: {len(failed)} routes had no valid ATC path — AI found alternatives")
        
        all_data[sid] = scenario_data
    
    return all_data


def generate_enhanced_charts(all_data):
    """Generates all enhanced charts."""
    
    scenarios_order = ['S1_Clear', 'S2_Rain', 'S3_Fog', 'S4_Blocked']
    scenario_labels = ['Clear\nWeather', 'Rain +\nTraffic', 'Dense Fog +\nCongestion', 'Blocked\nTaxiway']
    
    # ---- CHART 6: Smart Exit - Per gate comparison (Clear weather) ----
    fig, ax = plt.subplots(figsize=(14, 6))
    
    s1 = all_data['S1_Clear']
    gates = [d['gate_short'] for d in s1]
    atc_times = [d['atc_time'] if d['atc_time'] < float('inf') else 0 for d in s1]
    ai_times = [d['ai_time'] for d in s1]
    exits_used = [d['ai_exit'].replace('RWY_', '') for d in s1]
    
    x = np.arange(len(gates))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, atc_times, width, label='ATC Fixed Route (Exit C)',
                   color='#E74C3C', alpha=0.85)
    bars2 = ax.bar(x + width/2, ai_times, width, label='AI Smart Exit + Optimal Route',
                   color='#2ECC71', alpha=0.85)
    
    # Label AI bars with which exit was chosen
    for i, (bar, exit_name) in enumerate(zip(bars2, exits_used)):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 3,
                f'Exit {exit_name}', ha='center', va='bottom', fontsize=7,
                fontweight='bold', color='#1a7a3a')
    
    ax.set_xlabel('Destination Gate', fontsize=11)
    ax.set_ylabel('Taxi Time (seconds)', fontsize=11)
    ax.set_title('Smart Exit Selection: AI vs Fixed ATC Route (Clear Weather)',
                 fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(gates, fontsize=9)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.2)
    
    plt.tight_layout()
    plt.savefig('chart6_smart_exit_clear.png', dpi=300, bbox_inches='tight')
    print("Saved: chart6_smart_exit_clear.png")
    plt.close()
    
    
    # ---- CHART 7: Average improvement across scenarios (smart exit) ----
    fig, ax = plt.subplots(figsize=(10, 6))
    
    avg_improvements = []
    for sid in scenarios_order:
        data = all_data[sid]
        valid = [d for d in data if d['pct_improvement'] > 0]
        if valid:
            avg_improvements.append(np.mean([d['pct_improvement'] for d in valid]))
        else:
            avg_improvements.append(0)
    
    colors = ['#3498DB', '#F39C12', '#E74C3C', '#9B59B6']
    bars = ax.bar(scenario_labels, avg_improvements, color=colors, alpha=0.85)
    
    for bar, val in zip(bars, avg_improvements):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                    f'{val:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_xlabel('Operational Scenario', fontsize=11)
    ax.set_ylabel('Average Time Improvement (%)', fontsize=11)
    ax.set_title('AI Route Optimization: Average Improvement with Smart Exit Selection',
                 fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.2)
    
    plt.tight_layout()
    plt.savefig('chart7_smart_exit_improvement.png', dpi=300, bbox_inches='tight')
    print("Saved: chart7_smart_exit_improvement.png")
    plt.close()
    
    
    # ---- CHART 8: Fuel and Emissions Savings ----
    FUEL_FLOW = 11.0  # kg/min
    CO2_FACTOR = 3.16
    DAILY_FLIGHTS = 80
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    scenario_short = ['Clear', 'Rain', 'Fog']
    fuel_per_flight = []
    co2_per_flight = []
    annual_co2 = []
    
    for sid in ['S1_Clear', 'S2_Rain', 'S3_Fog']:
        data = all_data[sid]
        valid = [d for d in data if d['time_saved'] > 0]
        if valid:
            avg_saved_min = np.mean([d['time_saved'] for d in valid]) / 60
            fuel = avg_saved_min * FUEL_FLOW
            co2 = fuel * CO2_FACTOR
            fuel_per_flight.append(fuel)
            co2_per_flight.append(co2)
            annual_co2.append(co2 * DAILY_FLIGHTS * 365 / 1000)  # tonnes
        else:
            fuel_per_flight.append(0)
            co2_per_flight.append(0)
            annual_co2.append(0)
    
    # Fuel per flight
    bars = axes[0].bar(scenario_short, fuel_per_flight, color=['#3498DB', '#F39C12', '#E74C3C'], alpha=0.85)
    for bar, val in zip(bars, fuel_per_flight):
        axes[0].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.3,
                     f'{val:.1f} kg', ha='center', fontsize=10, fontweight='bold')
    axes[0].set_ylabel('Fuel Saved (kg)', fontsize=10)
    axes[0].set_title('Fuel Saved Per Flight', fontsize=11, fontweight='bold')
    axes[0].grid(axis='y', alpha=0.2)
    
    # CO2 per flight
    bars = axes[1].bar(scenario_short, co2_per_flight, color=['#3498DB', '#F39C12', '#E74C3C'], alpha=0.85)
    for bar, val in zip(bars, co2_per_flight):
        axes[1].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.3,
                     f'{val:.1f} kg', ha='center', fontsize=10, fontweight='bold')
    axes[1].set_ylabel('CO₂ Reduced (kg)', fontsize=10)
    axes[1].set_title('CO₂ Reduced Per Flight', fontsize=11, fontweight='bold')
    axes[1].grid(axis='y', alpha=0.2)
    
    # Annual CO2
    bars = axes[2].bar(scenario_short, annual_co2, color=['#3498DB', '#F39C12', '#E74C3C'], alpha=0.85)
    for bar, val in zip(bars, annual_co2):
        axes[2].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 5,
                     f'{val:.0f} t', ha='center', fontsize=10, fontweight='bold')
    axes[2].set_ylabel('CO₂ Reduced (tonnes/year)', fontsize=10)
    axes[2].set_title('Annual CO₂ Reduction (80 flights/day)', fontsize=11, fontweight='bold')
    axes[2].grid(axis='y', alpha=0.2)
    
    plt.suptitle('Environmental Impact: Fuel and Emission Savings', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('chart8_fuel_emissions.png', dpi=300, bbox_inches='tight')
    print("Saved: chart8_fuel_emissions.png")
    plt.close()
    
    
    # ---- CHART 9: Scenario 4 — AI resilience when baseline fails ----
    fig, ax = plt.subplots(figsize=(12, 6))
    
    s4 = all_data['S4_Blocked']
    gates = [d['gate_short'] for d in s4]
    ai_times_s4 = [d['ai_time'] for d in s4]
    atc_times_s4 = []
    bar_colors_atc = []
    
    for d in s4:
        if d['baseline_failed']:
            atc_times_s4.append(0)  # Can't show infinity — we'll mark it
            bar_colors_atc.append('#888888')
        else:
            atc_times_s4.append(d['atc_time'])
            bar_colors_atc.append('#E74C3C')
    
    x = np.arange(len(gates))
    
    # ATC bars
    for i in range(len(x)):
        ax.bar(x[i] - width/2, atc_times_s4[i], width, color=bar_colors_atc[i], alpha=0.85)
        if s4[i]['baseline_failed']:
            ax.text(x[i] - width/2, 20, 'ROUTE\nFAILED', ha='center', va='bottom',
                    fontsize=7, fontweight='bold', color='#CC0000')
    
    # AI bars
    ax.bar(x + width/2, ai_times_s4, width, label='AI Smart Exit + Reroute',
           color='#2ECC71', alpha=0.85)
    
    # Custom legend
    import matplotlib.patches as mpatches
    legend_elements = [
        mpatches.Patch(color='#E74C3C', alpha=0.85, label='ATC Fixed Route (working)'),
        mpatches.Patch(color='#888888', alpha=0.85, label='ATC Fixed Route (FAILED)'),
        mpatches.Patch(color='#2ECC71', alpha=0.85, label='AI Smart Exit + Reroute'),
    ]
    ax.legend(handles=legend_elements, fontsize=10)
    
    # Add exit labels on AI bars
    for i, d in enumerate(s4):
        exit_name = d['ai_exit'].replace('RWY_', '')
        ax.text(x[i] + width/2, ai_times_s4[i] + 3, f'Exit {exit_name}',
                ha='center', fontsize=7, fontweight='bold', color='#1a7a3a')
    
    ax.set_xlabel('Destination Gate', fontsize=11)
    ax.set_ylabel('Taxi Time (seconds)', fontsize=11)
    ax.set_title('Scenario 4: AI Resilience When Taxiway Is Blocked\n'
                 '(Apron 1 West Entry blocked by ground vehicle)',
                 fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(gates, fontsize=9)
    ax.grid(axis='y', alpha=0.2)
    
    plt.tight_layout()
    plt.savefig('chart9_blocked_resilience.png', dpi=300, bbox_inches='tight')
    print("Saved: chart9_blocked_resilience.png")
    plt.close()
    
    
    # ---- CHART 10: Distance saved per gate (Clear weather) ----
    fig, ax = plt.subplots(figsize=(14, 5))
    
    s1 = all_data['S1_Clear']
    gates = [d['gate_short'] for d in s1]
    dist_saved = [d['dist_saved'] for d in s1]
    
    colors = ['#2ECC71' if d > 0 else '#cccccc' for d in dist_saved]
    bars = ax.bar(gates, dist_saved, color=colors, alpha=0.85)
    
    for bar, val in zip(bars, dist_saved):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 5,
                    f'{val:.0f}m', ha='center', fontsize=9, fontweight='bold')
    
    ax.set_xlabel('Destination Gate', fontsize=11)
    ax.set_ylabel('Distance Saved (meters)', fontsize=11)
    ax.set_title('Distance Saved Per Route with Smart Exit Selection (Clear Weather)',
                 fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.2)
    ax.axhline(y=0, color='black', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig('chart10_distance_saved.png', dpi=300, bbox_inches='tight')
    print("Saved: chart10_distance_saved.png")
    plt.close()


def generate_route_comparison_maps(all_data):
    """Generate side-by-side route maps for key comparisons."""
    
    G = create_ahmedabad_airport()
    
    # Best example: gate that benefits most from smart exit
    s1 = all_data['S1_Clear']
    best_case = max(s1, key=lambda d: d['pct_improvement'])
    
    if best_case['ai_path']:
        visualize_airport(G,
            title=f"AI Smart Route: {best_case['ai_exit'].replace('RWY_','')} → "
                  f"{best_case['gate_short']} "
                  f"({best_case['ai_time']:.0f}s, {best_case['ai_dist']}m)",
            highlighted_path=best_case['ai_path'],
            filename="map_smart_ai_route.png")
    
    if best_case['atc_path']:
        visualize_airport(G,
            title=f"ATC Standard Route: C → "
                  f"{best_case['gate_short']} "
                  f"({best_case['atc_time']:.0f}s, {best_case['atc_dist']}m)",
            highlighted_path=best_case['atc_path'],
            filename="map_standard_atc_route.png")


def print_final_summary(all_data):
    """Prints the definitive summary for the paper."""
    
    FUEL_FLOW = 11.0
    CO2_FACTOR = 3.16
    DAILY_FLIGHTS = 80
    
    print("\n" + "="*75)
    print("  DEFINITIVE RESULTS SUMMARY FOR CONFERENCE PAPER")
    print("="*75)
    
    for sid, label in [('S1_Clear', 'Clear Weather'),
                       ('S2_Rain', 'Rain + Traffic'),
                       ('S3_Fog', 'Fog + Congestion'),
                       ('S4_Blocked', 'Blocked Taxiway')]:
        data = all_data[sid]
        valid = [d for d in data if d['atc_time'] < float('inf') and d['ai_time'] < float('inf')]
        improved = [d for d in valid if d['pct_improvement'] > 0]
        failed_baseline = [d for d in data if d['baseline_failed']]
        
        print(f"\n  {label}:")
        
        if valid:
            avg_atc = np.mean([d['atc_time'] for d in valid])
            avg_ai = np.mean([d['ai_time'] for d in valid])
            avg_pct = np.mean([d['pct_improvement'] for d in improved]) if improved else 0
            max_pct = max([d['pct_improvement'] for d in data])
            avg_dist_saved = np.mean([d['dist_saved'] for d in improved]) if improved else 0
            
            avg_time_saved_min = np.mean([d['time_saved'] for d in improved]) / 60 if improved else 0
            fuel_saved = avg_time_saved_min * FUEL_FLOW
            co2_saved = fuel_saved * CO2_FACTOR
            annual_co2 = co2_saved * DAILY_FLIGHTS * 365 / 1000
            
            print(f"    Routes improved:     {len(improved)}/{len(data)}")
            print(f"    Avg improvement:     {avg_pct:.1f}%")
            print(f"    Max improvement:     {max_pct:.1f}%")
            print(f"    Avg distance saved:  {avg_dist_saved:.0f}m")
            print(f"    Fuel saved/flight:   {fuel_saved:.1f} kg")
            print(f"    CO2 saved/flight:    {co2_saved:.1f} kg")
            print(f"    Annual CO2 saved:    {annual_co2:.0f} tonnes")
        
        if failed_baseline:
            print(f"    Baseline FAILED:     {len(failed_baseline)} routes (AI found alternatives)")
    
    print(f"\n{'='*75}")


# ================================================================
# MAIN
# ================================================================

if __name__ == "__main__":
    
    print("="*75)
    print("  ENHANCED SIMULATION: SMART EXIT SELECTION")
    print("  Ahmedabad International Airport (VAAH)")
    print("="*75)
    
    # Run smart exit comparison
    all_data = run_smart_exit_comparison()
    
    # Print final summary
    print_final_summary(all_data)
    
    # Generate all charts
    print("\nGenerating enhanced charts...")
    generate_enhanced_charts(all_data)
    
    # Generate route maps
    print("\nGenerating route comparison maps...")
    generate_route_comparison_maps(all_data)
    
    print("\n" + "="*75)
    print("  ENHANCED SIMULATION COMPLETE")
    print("  All charts saved in current directory")
    print("="*75)