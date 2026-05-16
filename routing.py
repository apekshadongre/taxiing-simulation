"""
Routing Engine - Dynamic Cost Function + Dijkstra
===================================================
Implements AI-assisted route optimization for airport taxiing.

The dynamic cost function adjusts edge weights based on:
  1. Base cost (distance + time)
  2. Weather multiplier (clear / rain / fog)
  3. Traffic congestion penalty (occupied taxiway segments)
  4. Blocked taxiways (completely removed from routing)
"""

import networkx as nx
import numpy as np
from airport_graph import create_ahmedabad_airport, visualize_airport, print_graph_summary


# ================================================================
# DYNAMIC COST FUNCTION
# ================================================================

# Weather conditions and their multipliers
# These increase traversal time and reduce safe taxi speed
WEATHER_CONDITIONS = {
    'clear':  {'multiplier': 1.0, 'description': 'Clear visibility (RVR > 550m)'},
    'rain':   {'multiplier': 1.4, 'description': 'Rain / reduced visibility (RVR 200-550m)'},
    'fog':    {'multiplier': 2.0, 'description': 'Dense fog (RVR < 200m)'},
}

# Traffic congestion levels for individual edges
# Simulates other aircraft or ground vehicles on a taxiway segment
CONGESTION_PENALTY = {
    'none':   0,      # No traffic on this segment
    'light':  15,     # Minor delay (seconds added)
    'medium': 40,     # Moderate delay
    'heavy':  80,     # Significant delay — near standstill
}


def compute_dynamic_cost(G, weather='clear', congested_edges=None, blocked_edges=None):
    """
    Computes dynamic edge costs based on current conditions.
    
    The cost of each edge is:
        dynamic_cost = (base_time_s * weather_multiplier) + congestion_penalty
    
    Args:
        G:                The airport DiGraph (original, unmodified)
        weather:          One of 'clear', 'rain', 'fog'
        congested_edges:  Dict of {(u, v): congestion_level} where level is
                          'light', 'medium', or 'heavy'
        blocked_edges:    List of (u, v) tuples — edges completely blocked
    
    Returns:
        G_dynamic:  A copy of G with updated 'dynamic_cost' attribute on edges
                    and blocked edges removed
    """
    
    if congested_edges is None:
        congested_edges = {}
    if blocked_edges is None:
        blocked_edges = []
    
    # Work on a copy so original graph stays clean
    G_dynamic = G.copy()
    
    # Get weather multiplier
    w_mult = WEATHER_CONDITIONS[weather]['multiplier']
    
    # Remove blocked edges
    for (u, v) in blocked_edges:
        if G_dynamic.has_edge(u, v):
            G_dynamic.remove_edge(u, v)
        # Also remove reverse direction if it exists
        if G_dynamic.has_edge(v, u):
            G_dynamic.remove_edge(v, u)
    
    # Compute dynamic cost for all remaining edges
    for u, v, data in G_dynamic.edges(data=True):
        base_time = data['base_time_s']
        
        # Apply weather multiplier to base time
        weather_adjusted_time = base_time * w_mult
        
        # Apply congestion penalty if this edge is congested
        cong_level = congested_edges.get((u, v), 'none')
        cong_penalty = CONGESTION_PENALTY[cong_level]
        
        # Total dynamic cost (in seconds)
        dynamic_cost = weather_adjusted_time + cong_penalty
        
        # Store on edge
        G_dynamic[u][v]['dynamic_cost'] = round(dynamic_cost, 1)
        G_dynamic[u][v]['weather_multiplier'] = w_mult
        G_dynamic[u][v]['congestion_level'] = cong_level
        G_dynamic[u][v]['congestion_penalty_s'] = cong_penalty
    
    return G_dynamic


# ================================================================
# DIJKSTRA ROUTING
# ================================================================

def find_optimal_route(G_dynamic, source, destination):
    """
    Finds the shortest (minimum cost) path using Dijkstra's algorithm.
    
    Args:
        G_dynamic:    Graph with 'dynamic_cost' edge attribute
        source:       Starting node (runway exit)
        destination:  Target node (gate/stand)
    
    Returns:
        dict with:
            'path':           List of nodes in the optimal route
            'total_cost_s':   Total dynamic cost in seconds
            'total_distance_m': Total distance in meters
            'num_segments':   Number of taxiway segments
            'segment_details': List of dicts with per-segment info
            'instructions':   Human-readable taxiing instructions
    """
    
    try:
        # Run Dijkstra
        path = nx.dijkstra_path(G_dynamic, source, destination, weight='dynamic_cost')
        total_cost = nx.dijkstra_path_length(G_dynamic, source, destination, weight='dynamic_cost')
    except nx.NetworkXNoPath:
        return {
            'path': None,
            'total_cost_s': float('inf'),
            'error': f'No route found from {source} to {destination}'
        }
    
    # Collect segment details
    total_distance = 0
    segments = []
    instructions = []
    
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        edge_data = G_dynamic[u][v]
        
        total_distance += edge_data['distance_m']
        
        segments.append({
            'from': u,
            'to': v,
            'taxiway': edge_data['taxiway'],
            'distance_m': edge_data['distance_m'],
            'base_time_s': edge_data['base_time_s'],
            'dynamic_cost_s': edge_data['dynamic_cost'],
            'congestion': edge_data.get('congestion_level', 'none'),
        })
        
        # Generate human-readable instruction
        twy = edge_data['taxiway']
        dist = edge_data['distance_m']
        if 'RWY' in u:
            instructions.append(f"Exit runway via Taxiway {twy}")
        elif 'APRON' in v or 'STAND' in v:
            instructions.append(f"Proceed to {v.replace('_', ' ')} ({dist}m)")
        else:
            instructions.append(f"Follow Taxiway {twy} to {v} ({dist}m)")
    
    return {
        'path': path,
        'total_cost_s': round(total_cost, 1),
        'total_distance_m': total_distance,
        'total_time_min': round(total_cost / 60, 2),
        'num_segments': len(segments),
        'segment_details': segments,
        'instructions': instructions,
    }


# ================================================================
# BASELINE ROUTE (what a pilot would take without the system)
# ================================================================

def get_baseline_route(source, destination):
    """
    Returns a fixed 'standard' route that a pilot would typically follow
    based on ATC instructions — NOT optimized, just the most common/default path.
    
    In real life, ATC gives a fixed route like:
    "Exit runway at Charlie, taxi via Papa to your gate"
    
    This doesn't account for traffic, weather, or which exit is actually closest.
    The baseline always uses a fixed mid-runway exit and the full parallel taxiway.
    """
    
    # Standard ATC approach: always exit at C (mid-runway), take P all the way
    # This is deliberately suboptimal for many gate assignments
    
    baseline_routes = {
        # For RWY 23 landings (arriving from northeast, landing heading southwest)
        # ATC typically assigns exit C or D regardless of gate location
        'RWY_C': {
            'STAND_70': ['RWY_C', 'P_C', 'P_F', 'APRON1_WEST', 'STAND_70'],
            'STAND_75': ['RWY_C', 'P_C', 'P_F', 'APRON1_WEST', 'STAND_75'],
            'STAND_80': ['RWY_C', 'P_C', 'P_F', 'APRON1_WEST', 'APRON1_CENTER', 'STAND_80'],
            'STAND_85': ['RWY_C', 'P_C', 'P_F', 'APRON1_WEST', 'APRON1_CENTER', 'STAND_85'],
            'STAND_90': ['RWY_C', 'P_C', 'APRON1_EAST', 'STAND_90'],
            'STAND_40': ['RWY_C', 'P_C', 'P_K', 'APRON2_ENTRY', 'STAND_40'],
            'STAND_45': ['RWY_C', 'P_C', 'P_K', 'APRON2_ENTRY', 'STAND_45'],
            'STAND_50': ['RWY_C', 'P_C', 'P_K', 'APRON2_ENTRY', 'STAND_50'],
            'STAND_25': ['RWY_C', 'P_C', 'P_K', 'P_D', 'APRON3_ENTRY', 'STAND_25'],
            'STAND_28': ['RWY_C', 'P_C', 'P_K', 'P_D', 'APRON3_ENTRY', 'STAND_28'],
            'STAND_30': ['RWY_C', 'P_C', 'P_K', 'P_D', 'APRON3_ENTRY', 'STAND_30'],
        },
        'RWY_D': {
            'STAND_70': ['RWY_D', 'P_D', 'P_K', 'P_C', 'P_F', 'APRON1_WEST', 'STAND_70'],
            'STAND_75': ['RWY_D', 'P_D', 'P_K', 'P_C', 'P_F', 'APRON1_WEST', 'STAND_75'],
            'STAND_80': ['RWY_D', 'P_D', 'P_K', 'P_C', 'P_F', 'APRON1_WEST', 'APRON1_CENTER', 'STAND_80'],
            'STAND_85': ['RWY_D', 'P_D', 'P_K', 'P_C', 'P_F', 'APRON1_WEST', 'APRON1_CENTER', 'STAND_85'],
            'STAND_90': ['RWY_D', 'P_D', 'P_K', 'P_C', 'APRON1_EAST', 'STAND_90'],
            'STAND_40': ['RWY_D', 'P_D', 'P_K', 'APRON2_ENTRY', 'STAND_40'],
            'STAND_45': ['RWY_D', 'P_D', 'P_K', 'APRON2_ENTRY', 'STAND_45'],
            'STAND_50': ['RWY_D', 'P_D', 'P_K', 'APRON2_ENTRY', 'STAND_50'],
            'STAND_25': ['RWY_D', 'P_D', 'APRON3_ENTRY', 'STAND_25'],
            'STAND_28': ['RWY_D', 'P_D', 'APRON3_ENTRY', 'STAND_28'],
            'STAND_30': ['RWY_D', 'P_D', 'APRON3_ENTRY', 'STAND_30'],
        },
    }
    
    if source in baseline_routes and destination in baseline_routes[source]:
        return baseline_routes[source][destination]
    
    return None


def compute_route_cost(G_dynamic, path):
    """
    Computes the total cost of a given path on the dynamic graph.
    
    Args:
        G_dynamic: Graph with dynamic_cost attributes
        path: List of nodes
    
    Returns:
        dict with total_cost_s, total_distance_m, total_time_min
    """
    if path is None:
        return {'total_cost_s': float('inf'), 'total_distance_m': 0, 'total_time_min': float('inf')}
    
    total_cost = 0
    total_distance = 0
    
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        if G_dynamic.has_edge(u, v):
            total_cost += G_dynamic[u][v]['dynamic_cost']
            total_distance += G_dynamic[u][v]['distance_m']
        else:
            return {'total_cost_s': float('inf'), 'total_distance_m': 0, 
                    'total_time_min': float('inf'), 'error': f'Edge {u}->{v} not available'}
    
    return {
        'total_cost_s': round(total_cost, 1),
        'total_distance_m': total_distance,
        'total_time_min': round(total_cost / 60, 2),
    }


# ================================================================
# PRINT ROUTE DETAILS
# ================================================================

def print_route(result, label="ROUTE"):
    """Pretty-prints a route result."""
    
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    
    if result.get('path') is None:
        print(f"  ERROR: {result.get('error', 'No route found')}")
        return
    
    print(f"  Path: {' -> '.join(result['path'])}")
    print(f"  Total distance: {result['total_distance_m']}m")
    print(f"  Total time:     {result['total_cost_s']}s ({result['total_time_min']} min)")
    print(f"  Segments:       {result['num_segments']}")
    print()
    
    if 'instructions' in result:
        print("  PILOT INSTRUCTIONS:")
        for i, instr in enumerate(result['instructions'], 1):
            print(f"    {i}. {instr}")
    
    print(f"{'='*60}")


# ================================================================
# TEST - Run a sample route
# ================================================================

if __name__ == "__main__":
    
    # Build airport
    G = create_ahmedabad_airport()
    
    print("\n" + "="*60)
    print("  ROUTING ENGINE TEST")
    print("="*60)
    
    # --- Test 1: Clear weather, no traffic ---
    print("\n>>> SCENARIO: Clear weather, no congestion")
    print("    Landing on RWY 23, assigned to STAND 80 (Apron 1)")
    
    G_dyn = compute_dynamic_cost(G, weather='clear')
    
    # Find optimal route from runway exit D to Stand 80
    result_optimal = find_optimal_route(G_dyn, 'RWY_D', 'STAND_80')
    print_route(result_optimal, "AI-OPTIMIZED ROUTE (Dijkstra)")
    
    # Compare with baseline (ATC standard route)
    baseline_path = get_baseline_route('RWY_D', 'STAND_80')
    if baseline_path:
        baseline_cost = compute_route_cost(G_dyn, baseline_path)
        print(f"\n  BASELINE (Standard ATC Route):")
        print(f"  Path: {' -> '.join(baseline_path)}")
        print(f"  Total distance: {baseline_cost['total_distance_m']}m")
        print(f"  Total time:     {baseline_cost['total_cost_s']}s ({baseline_cost['total_time_min']} min)")
        
        # Improvement
        if baseline_cost['total_cost_s'] > 0 and result_optimal['total_cost_s'] > 0:
            time_saved = baseline_cost['total_cost_s'] - result_optimal['total_cost_s']
            pct_saved = (time_saved / baseline_cost['total_cost_s']) * 100
            print(f"\n  >> TIME SAVED: {time_saved:.1f}s ({pct_saved:.1f}% improvement)")
    
    # --- Test 2: Fog + congestion ---
    print("\n\n>>> SCENARIO: Dense fog, heavy traffic on P_C -> P_K")
    
    congested = {
        ('P_C', 'P_K'): 'heavy',
        ('P_K', 'P_C'): 'heavy',
    }
    
    G_dyn_fog = compute_dynamic_cost(G, weather='fog', congested_edges=congested)
    
    result_fog = find_optimal_route(G_dyn_fog, 'RWY_D', 'STAND_80')
    print_route(result_fog, "AI-OPTIMIZED ROUTE (Fog + Congestion)")
    
    # Visualize the clear-weather optimal route
    visualize_airport(G, 
                      title="AI-Optimized Route: RWY_D to STAND_80 (Clear Weather)",
                      highlighted_path=result_optimal['path'],
                      filename="route_optimal_clear.png")
    
    print("\nRouting engine test complete!")