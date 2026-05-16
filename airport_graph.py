"""
Ahmedabad Airport (VAAH) - Taxiway Network Graph Model
========================================================
Based on AAI Aerodrome Chart (AIP Supplement 222/2025)
Runway 05/23 | 3,505m | Single runway operations

This file builds the airport taxiway network as a weighted directed graph
for use in the AI-assisted taxiing route optimization simulation.
"""

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


def create_ahmedabad_airport():
    """
    Creates a directed graph representing Ahmedabad Airport's taxiway network.
    
    Node types:
        - runway_exit: Points where aircraft exit the runway onto a taxiway
        - taxiway:     Intersections along taxiways
        - apron:       Entry points to apron/parking areas
        - gate:        Aircraft parking stands
    
    Edge weights:
        - distance_m:  Physical length of the taxiway segment (meters)
        - base_speed:  Normal taxi speed on that segment (knots)
        - base_time_s: Time to traverse at base speed (seconds)
        - taxiway:     Name of the taxiway (e.g., 'A', 'P', 'R')
    
    Coordinate system:
        - x-axis: Along runway direction (RWY 05 end = left, RWY 23 end = right)
        - y-axis: Perpendicular (runway at bottom, terminals above)
        - Units: meters (approximate, for visualization)
    
    Returns:
        G: networkx.DiGraph with all nodes and edges
    """
    
    G = nx.DiGraph()
    
    # ================================================================
    # NODE DEFINITIONS
    # ================================================================
    # Positions are approximate, based on AAI aerodrome chart.
    # Runway 05/23 is 3,505m long.
    # Taxiway intersection distances from RWY 05 threshold:
    #   TWY A: ~150m | TWY B: ~186m | TWY F: ~400m
    #   TWY C: ~960m | TWY K: ~1400m | TWY D: ~1794m
    #   TWY R4: ~1884m | TWY R1: ~2996m
    # Parallel taxiway P offset ~200m northwest of runway centerline.
    # Aprons/terminals offset ~400-600m from runway.
    
    nodes = {
        # --- RUNWAY EXIT POINTS ---
        # Where cross-taxiways meet the runway
        # Aircraft use these to exit after landing
        'RWY_A':  (150,   0),    # TWY A at runway (near RWY 05 end)
        'RWY_B':  (186,   0),    # TWY B at runway
        'RWY_F':  (400,   0),    # TWY F at runway
        'RWY_C':  (960,   0),    # TWY C at runway
        'RWY_K':  (1400,  0),    # TWY K at runway
        'RWY_D':  (1794,  0),    # TWY D at runway
        'RWY_R4': (1884,  0),    # TWY R4 at runway
        'RWY_R1': (2996,  0),    # TWY R1 at runway (near RWY 23 end)
        
        # --- PARALLEL TAXIWAY P NODES ---
        # Main parallel taxiway running alongside the runway
        # Aircraft use this as the "highway" between runway exits and aprons
        'P_A':  (150,  200),     # TWY P at A intersection
        'P_B':  (186,  200),     # TWY P at B intersection
        'P_F':  (400,  200),     # TWY P at F intersection
        'P_C':  (960,  200),     # TWY P at C intersection
        'P_K':  (1400, 200),     # TWY P at K intersection
        'P_D':  (1794, 200),     # TWY P at D intersection
        'P_R4': (1884, 200),     # TWY P at R4 intersection (end of TWY P)
        
        # --- EXTENDED PARALLEL TAXIWAY R NODES ---
        # TWY R extends the parallel taxiway from R4 to R1
        'R_MID': (2440, 200),    # Midpoint of TWY R
        'R_R1':  (2996, 200),    # TWY R at R1 intersection
        
        # --- APRON ENTRY/JUNCTION NODES ---
        # These connect the parallel taxiway to the apron areas
        'APRON1_EAST':  (960,  400),   # Apron 1 entry from TWY C side
        'APRON1_WEST':  (400,  400),   # Apron 1 entry from TWY F side
        'APRON1_CENTER':(680,  400),   # Apron 1 central junction
        'APRON2_ENTRY': (1400, 400),   # Apron 2 entry from TWY K side
        'APRON3_ENTRY': (1794, 400),   # Apron 3 entry from TWY D side
        
        # --- GATE / PARKING STAND NODES ---
        # Apron 1 stands (domestic terminal T1 area)
        # Stands 70-94 in real life; we model representative ones
        'STAND_70': (300,  520),   # Western end of Apron 1
        'STAND_75': (480,  520),
        'STAND_80': (660,  520),
        'STAND_85': (840,  520),
        'STAND_90': (960,  520),   # Eastern end of Apron 1
        
        # Apron 2 stands (international terminal T2 area)
        'STAND_40': (1300, 520),
        'STAND_45': (1400, 520),
        'STAND_50': (1500, 520),
        
        # Apron 3 stands (cargo/additional terminal area)
        'STAND_25': (1700, 520),
        'STAND_28': (1794, 520),
        'STAND_30': (1900, 520),
    }
    
    # Add all nodes to graph with their attributes
    for node_id, pos in nodes.items():
        if 'STAND' in node_id:
            ntype = 'gate'
        elif 'RWY' in node_id:
            ntype = 'runway_exit'
        elif 'APRON' in node_id:
            ntype = 'apron'
        else:
            ntype = 'taxiway'
        
        G.add_node(node_id, pos=pos, node_type=ntype)
    
    
    # ================================================================
    # EDGE DEFINITIONS
    # ================================================================
    # Each edge is a taxiway segment: (from, to, distance_m, speed_kts, taxiway_name)
    #
    # Speed guidelines (ICAO):
    #   Straight taxiway segments: 15-20 knots (~8-10 m/s)
    #   Turns and intersections: 8-10 knots (~4-5 m/s)
    #   Apron areas: 5-8 knots (~2.5-4 m/s)
    #   Runway exit (high-speed): 10-15 knots
    #
    # Edges are BIDIRECTIONAL for taxiways (aircraft can go both ways)
    # Runway exits are ONE-WAY (runway -> taxiway only)
    
    edges = []
    
    # --- RUNWAY TO PARALLEL TAXIWAY (one-way: runway exit -> taxiway P) ---
    # These are the cross-taxiways connecting runway to parallel taxiway
    edges.append(('RWY_A',  'P_A',  200, 10, 'A'))
    edges.append(('RWY_B',  'P_B',  200, 10, 'B'))
    edges.append(('RWY_F',  'P_F',  200, 10, 'F'))
    edges.append(('RWY_C',  'P_C',  200, 10, 'C'))
    edges.append(('RWY_K',  'P_K',  200, 10, 'K'))
    edges.append(('RWY_D',  'P_D',  200, 10, 'D'))
    edges.append(('RWY_R4', 'P_R4', 200, 10, 'R4'))
    edges.append(('RWY_R1', 'R_R1', 200, 10, 'R1'))
    
    # --- PARALLEL TAXIWAY P (bidirectional) ---
    # Segments along TWY P from west to east
    p_segments = [
        ('P_A',  'P_B',   36,  15, 'P'),   # A to B (very short)
        ('P_B',  'P_F',  214,  18, 'P'),   # B to F
        ('P_F',  'P_C',  560,  20, 'P'),   # F to C
        ('P_C',  'P_K',  440,  20, 'P'),   # C to K
        ('P_K',  'P_D',  394,  20, 'P'),   # K to D
        ('P_D',  'P_R4',  90,  15, 'P'),   # D to R4 (short)
    ]
    for (u, v, d, s, name) in p_segments:
        edges.append((u, v, d, s, name))
        edges.append((v, u, d, s, name))  # Reverse direction
    
    # --- EXTENDED TAXIWAY R (bidirectional, R4 to R1) ---
    r_segments = [
        ('P_R4', 'R_MID', 556, 18, 'R'),   # R4 to midpoint
        ('R_MID', 'R_R1', 556, 18, 'R'),   # Midpoint to R1
    ]
    for (u, v, d, s, name) in r_segments:
        edges.append((u, v, d, s, name))
        edges.append((v, u, d, s, name))
    
    # --- PARALLEL TAXIWAY TO APRON ENTRIES (bidirectional) ---
    edges.append(('P_F',  'APRON1_WEST',  200, 8, 'F'))
    edges.append(('APRON1_WEST', 'P_F',   200, 8, 'F'))
    
    edges.append(('P_C',  'APRON1_EAST',  200, 8, 'C'))
    edges.append(('APRON1_EAST', 'P_C',   200, 8, 'C'))
    
    edges.append(('P_K',  'APRON2_ENTRY', 200, 8, 'K'))
    edges.append(('APRON2_ENTRY', 'P_K',  200, 8, 'K'))
    
    edges.append(('P_D',  'APRON3_ENTRY', 200, 8, 'D'))
    edges.append(('APRON3_ENTRY', 'P_D',  200, 8, 'D'))
    
    # --- APRON INTERNAL CONNECTIONS (bidirectional, slow speed) ---
    # Apron 1 internal
    edges.append(('APRON1_WEST',   'APRON1_CENTER', 280, 6, 'Apron1'))
    edges.append(('APRON1_CENTER', 'APRON1_WEST',   280, 6, 'Apron1'))
    edges.append(('APRON1_CENTER', 'APRON1_EAST',   280, 6, 'Apron1'))
    edges.append(('APRON1_EAST',   'APRON1_CENTER', 280, 6, 'Apron1'))
    
    # --- APRON TO GATES (bidirectional, very slow) ---
    # Apron 1 gates
    edges.append(('APRON1_WEST',   'STAND_70', 150, 5, 'Apron1'))
    edges.append(('STAND_70', 'APRON1_WEST',   150, 5, 'Apron1'))
    
    edges.append(('APRON1_WEST',   'STAND_75', 140, 5, 'Apron1'))
    edges.append(('STAND_75', 'APRON1_WEST',   140, 5, 'Apron1'))
    
    edges.append(('APRON1_CENTER', 'STAND_80', 120, 5, 'Apron1'))
    edges.append(('STAND_80', 'APRON1_CENTER', 120, 5, 'Apron1'))
    
    edges.append(('APRON1_CENTER', 'STAND_85', 150, 5, 'Apron1'))
    edges.append(('STAND_85', 'APRON1_CENTER', 150, 5, 'Apron1'))
    
    edges.append(('APRON1_EAST',   'STAND_90', 120, 5, 'Apron1'))
    edges.append(('STAND_90', 'APRON1_EAST',   120, 5, 'Apron1'))
    
    # Apron 2 gates
    edges.append(('APRON2_ENTRY', 'STAND_40', 150, 5, 'Apron2'))
    edges.append(('STAND_40', 'APRON2_ENTRY', 150, 5, 'Apron2'))
    
    edges.append(('APRON2_ENTRY', 'STAND_45', 120, 5, 'Apron2'))
    edges.append(('STAND_45', 'APRON2_ENTRY', 120, 5, 'Apron2'))
    
    edges.append(('APRON2_ENTRY', 'STAND_50', 150, 5, 'Apron2'))
    edges.append(('STAND_50', 'APRON2_ENTRY', 150, 5, 'Apron2'))
    
    # Apron 3 gates
    edges.append(('APRON3_ENTRY', 'STAND_25', 150, 5, 'Apron3'))
    edges.append(('STAND_25', 'APRON3_ENTRY', 150, 5, 'Apron3'))
    
    edges.append(('APRON3_ENTRY', 'STAND_28', 120, 5, 'Apron3'))
    edges.append(('STAND_28', 'APRON3_ENTRY', 120, 5, 'Apron3'))
    
    edges.append(('APRON3_ENTRY', 'STAND_30', 150, 5, 'Apron3'))
    edges.append(('STAND_30', 'APRON3_ENTRY', 150, 5, 'Apron3'))
    
    
    # ================================================================
    # ADD EDGES TO GRAPH WITH COMPUTED ATTRIBUTES
    # ================================================================
    
    for (u, v, dist, speed_kts, twy_name) in edges:
        speed_ms = speed_kts * 0.514444   # Convert knots to m/s
        time_s = dist / speed_ms           # Time in seconds
        
        G.add_edge(u, v,
                   distance_m=dist,
                   base_speed_kts=speed_kts,
                   base_speed_ms=round(speed_ms, 2),
                   base_time_s=round(time_s, 1),
                   taxiway=twy_name)
    
    return G


def visualize_airport(G, title="Ahmedabad Airport (VAAH) - Taxiway Network",
                      highlighted_path=None, filename=None):
    """
    Draws the airport taxiway graph.
    
    Args:
        G: The airport DiGraph
        title: Plot title
        highlighted_path: List of node IDs to highlight as a route (optional)
        filename: If provided, saves the figure to this file
    """
    
    fig, ax = plt.subplots(1, 1, figsize=(16, 8))
    
    pos = nx.get_node_attributes(G, 'pos')
    node_types = nx.get_node_attributes(G, 'node_type')
    
    # --- Draw the runway as a thick gray line ---
    rwy_05 = (0, 0)
    rwy_23 = (3505, 0)
    ax.plot([rwy_05[0], rwy_23[0]], [rwy_05[1], rwy_23[1]],
            color='gray', linewidth=12, alpha=0.3, solid_capstyle='round')
    ax.text(0, -40, 'RWY 05', fontsize=9, ha='center', fontweight='bold', color='gray')
    ax.text(3505, -40, 'RWY 23', fontsize=9, ha='center', fontweight='bold', color='gray')
    
    # --- Draw edges (taxiway segments) ---
    # Draw non-highlighted edges first
    for (u, v, data) in G.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        ax.plot([x0, x1], [y0, y1], color='#cccccc', linewidth=1.5,
                alpha=0.6, zorder=1)
    
    # --- Draw highlighted path if provided ---
    if highlighted_path:
        for i in range(len(highlighted_path) - 1):
            u = highlighted_path[i]
            v = highlighted_path[i + 1]
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            ax.plot([x0, x1], [y0, y1], color='#00AA00', linewidth=3.5,
                    alpha=0.9, zorder=2)
    
    # --- Draw nodes by type ---
    color_map = {
        'runway_exit': '#E74C3C',   # Red
        'taxiway':     '#3498DB',   # Blue
        'apron':       '#F39C12',   # Orange
        'gate':        '#2ECC71',   # Green
    }
    size_map = {
        'runway_exit': 80,
        'taxiway':     60,
        'apron':       70,
        'gate':        90,
    }
    
    for ntype in ['taxiway', 'apron', 'runway_exit', 'gate']:
        nodelist = [n for n in G.nodes() if node_types[n] == ntype]
        nx.draw_networkx_nodes(G, pos, nodelist=nodelist,
                               node_color=color_map[ntype],
                               node_size=size_map[ntype],
                               ax=ax)
    
    # --- Node labels ---
    label_pos = {k: (v[0], v[1] + 25) for k, v in pos.items()}
    # Only label important nodes to avoid clutter
    labels = {}
    for n in G.nodes():
        if 'STAND' in n:
            labels[n] = n.replace('STAND_', 'S')
        elif 'RWY' in n:
            labels[n] = n.replace('RWY_', '')
        elif n.startswith('P_'):
            labels[n] = n
        elif 'APRON' in n:
            labels[n] = n.replace('APRON', 'Apr').replace('_ENTRY', '').replace('_', '')
        else:
            labels[n] = n
    
    nx.draw_networkx_labels(G, label_pos, labels, font_size=6,
                            font_color='#333333', ax=ax)
    
    # --- Legend ---
    legend_elements = [
        mpatches.Patch(color='#E74C3C', label='Runway Exit'),
        mpatches.Patch(color='#3498DB', label='Taxiway Junction'),
        mpatches.Patch(color='#F39C12', label='Apron Entry'),
        mpatches.Patch(color='#2ECC71', label='Parking Stand (Gate)'),
    ]
    if highlighted_path:
        from matplotlib.lines import Line2D
        legend_elements.append(Line2D([0], [0], color='#00AA00', linewidth=3,
                                       label='Optimized Route'))
    
    ax.legend(handles=legend_elements, loc='upper right', fontsize=8)
    
    # --- Formatting ---
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Distance along runway axis (meters)', fontsize=10)
    ax.set_ylabel('Distance from runway (meters)', fontsize=10)
    ax.set_xlim(-100, 3600)
    ax.set_ylim(-80, 620)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.15)
    
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Figure saved: {filename}")
    
    plt.show()


def print_graph_summary(G):
    """Prints a summary of the airport graph."""
    
    node_types = nx.get_node_attributes(G, 'node_type')
    
    print("=" * 60)
    print("AHMEDABAD AIRPORT (VAAH) - GRAPH SUMMARY")
    print("=" * 60)
    print(f"Total nodes: {G.number_of_nodes()}")
    print(f"Total edges: {G.number_of_edges()}")
    print()
    
    for ntype in ['runway_exit', 'taxiway', 'apron', 'gate']:
        count = sum(1 for n in G.nodes() if node_types[n] == ntype)
        node_list = [n for n in G.nodes() if node_types[n] == ntype]
        print(f"  {ntype:15s}: {count:3d} nodes  ->  {node_list}")
    
    print()
    print("EDGE STATISTICS:")
    distances = [d['distance_m'] for _, _, d in G.edges(data=True)]
    times = [d['base_time_s'] for _, _, d in G.edges(data=True)]
    print(f"  Distance range: {min(distances):.0f}m - {max(distances):.0f}m")
    print(f"  Time range:     {min(times):.1f}s - {max(times):.1f}s")
    print(f"  Total taxiway length (unique segments): {sum(distances)/2:.0f}m")
    print("=" * 60)


# ================================================================
# MAIN - Run this to test the airport graph
# ================================================================

if __name__ == "__main__":
    # Build the airport
    G = create_ahmedabad_airport()
    
    # Print summary
    print_graph_summary(G)
    
    # Visualize it
    visualize_airport(G, filename="airport_layout.png")
    
    print("\nAirport graph built successfully!")
    print("Nodes:", list(G.nodes()))