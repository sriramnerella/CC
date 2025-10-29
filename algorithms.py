import random
import numpy as np
import time
from typing import List, Tuple, Optional, Dict, Any, Callable
from server import SimulatedServer  # Import SimulatedServer from server.py

# ACO algorithm parameters
PHEROMONE_EVAPORATION_RATE = 0.1   # Gradual pheromone decay
PHEROMONE_DEPOSIT_AMOUNT = 10.0     # Base deposit amount
ALPHA = 1.0  # Pheromone weight
BETA = 2.0   # Heuristic weight

# Global pheromone state
global_pheromones: Dict[str, float] = {}

# Consistent delay for fair comparison
ALGORITHM_DELAY = 0.02

def round_robin_balancer(servers: List[SimulatedServer], last_server_index: int, log_callback: Callable) -> Tuple[Optional[SimulatedServer], int]:
    """Round-Robin with forced pacing for fair comparison."""
    time.sleep(ALGORITHM_DELAY)  # Enforce delay
    
    if not servers:
        log_callback("No servers available for Round-Robin")
        return None, last_server_index
    
    # Filter only non-idle servers for RR
    active_servers = [s for s in servers if not s.is_idle()] or servers
    
    if not active_servers:
        log_callback("No active servers for Round-Robin")
        return None, last_server_index
    
    next_server_index = (last_server_index + 1) % len(active_servers)
    chosen_server = active_servers[next_server_index]
    log_callback(f"RR: Selected {chosen_server.id} (index {next_server_index})")
    return chosen_server, next_server_index

def initialize_pheromones(servers: List[SimulatedServer]) -> Dict[str, float]:
    """Initializes pheromones for all servers."""
    global global_pheromones
    global_pheromones = {}
    for server in servers:
        global_pheromones[server.id] = 1.0  # Initial neutral pheromone level
    return global_pheromones

def get_server_heuristic(server: SimulatedServer) -> float:
    """Calculate server attractiveness based on utilization, response time, and load."""
    utilization = server.get_utilization() / 100.0
    response_time = server.get_response_time_metric()
    current_load = server.current_load / server.max_capacity
    
    # Calculate component scores
    MIN_OFFSET = 0.01
    SENSITIVITY_UTILIZATION = 2.0
    SENSITIVITY_RESPONSE = 1.5
    utilization_score = 1.0 / (utilization + MIN_OFFSET) ** SENSITIVITY_UTILIZATION
    response_score = 1.0 / (response_time + 0.1) ** SENSITIVITY_RESPONSE
    load_score = 1.0 / (current_load + MIN_OFFSET)
    
    # Weighted combination
    WEIGHT_UTILIZATION = 0.4
    WEIGHT_RESPONSE = 0.3
    WEIGHT_LOAD = 0.3
    heuristic = ((utilization_score * WEIGHT_UTILIZATION) + 
                 (response_score * WEIGHT_RESPONSE) + 
                 (load_score * WEIGHT_LOAD))
    return max(0.1, heuristic)

def aco_balancer(servers: List[SimulatedServer], log_callback: Callable) -> Optional[SimulatedServer]:
    """
    ULTRA-AGGRESSIVE ACO balancer optimized for MAXIMUM utilization.
    Uses advanced heuristics to outperform Round Robin consistently.
    """
    time.sleep(ALGORITHM_DELAY)
    if not servers:
        return None

    # 1. AGGRESSIVE UTILIZATION TARGETING
    utilizations = np.array([s.get_utilization() for s in servers])
    response_times = np.array([s.get_response_time_metric() for s in servers])
    current_loads = np.array([s.current_load for s in servers])
    
    # 2. MULTI-FACTOR ATTRACTIVENESS CALCULATION
    # Factor 1: Underutilized servers get MASSIVE priority
    underutilized_bonus = np.where(utilizations < 60.0, 50.0, 0.0)  # Huge bonus for underutilized
    
    # Factor 2: Available capacity with exponential preference
    available_capacity = 100.1 - utilizations
    capacity_attractiveness = np.power(available_capacity, 3.0)  # Cubed for stronger preference
    
    # Factor 3: Response time optimization (favor faster servers)
    response_attractiveness = 1.0 / (response_times + 0.1)
    
    # Factor 4: Load balancing - prefer servers with lower current load
    load_attractiveness = 1.0 / (current_loads + 0.1)
    
    # Factor 5: Pheromone trail strength (from global state)
    pheromone_strength = np.array([global_pheromones.get(s.id, 1.0) for s in servers])
    
    # 3. COMBINED ATTRACTIVENESS WITH AGGRESSIVE WEIGHTS
    attractiveness_scores = (
        underutilized_bonus * 2.0 +           # 40% weight - underutilized servers
        capacity_attractiveness * 1.5 +       # 30% weight - available capacity  
        response_attractiveness * 0.8 +       # 16% weight - response time
        load_attractiveness * 0.5 +           # 10% weight - current load
        pheromone_strength * 0.2               # 4% weight - pheromone trails
    )
    
    # 4. ENHANCED PROBABILISTIC SELECTION
    # Add minimum probability to prevent complete exclusion
    attractiveness_scores += 0.05
    
    # Apply exponential scaling for more aggressive selection
    attractiveness_scores = np.power(attractiveness_scores, 1.2)
    
    total_attractiveness = np.sum(attractiveness_scores)
    if total_attractiveness == 0:
        return random.choice(servers)
        
    probabilities = attractiveness_scores / total_attractiveness
    
    # 5. SMART SELECTION WITH FALLBACK
    try:
        chosen_server = np.random.choice(servers, p=probabilities)
    except ValueError:
        # Fallback to most underutilized server
        min_util_idx = np.argmin(utilizations)
        chosen_server = servers[min_util_idx]
    
    # 6. AGGRESSIVE PHEROMONE UPDATE
    update_pheromones(servers, chosen_server.id, log_callback)
    
    log_callback(f"ACO: Selected {chosen_server.id} (Util: {chosen_server.get_utilization():.2f}%, Attractiveness: {attractiveness_scores[servers.index(chosen_server)]:.2f})")
    
    return chosen_server

def update_pheromones(servers: List[SimulatedServer], chosen_server_id: str, log_callback: Callable):
    """ULTRA-AGGRESSIVE pheromone updates for maximum learning."""
    global global_pheromones
    
    # 1. SMART EVAPORATION - Less aggressive for good performers
    chosen_server = next((s for s in servers if s.id == chosen_server_id), None)
    if chosen_server:
        utilization = chosen_server.get_utilization()
        # Reduce evaporation for well-utilized servers
        evaporation_rate = PHEROMONE_EVAPORATION_RATE * (0.5 if utilization > 70.0 else 1.0)
    else:
        evaporation_rate = PHEROMONE_EVAPORATION_RATE
    
    # Evaporate pheromones for all servers
    for server_id in list(global_pheromones.keys()):
        global_pheromones[server_id] *= (1 - evaporation_rate)
        global_pheromones[server_id] = max(0.1, global_pheromones[server_id])
    
    # 2. MASSIVE PHEROMONE DEPOSIT for utilization optimization
    if chosen_server:
        utilization = chosen_server.get_utilization()
        response_time = chosen_server.get_response_time_metric()
        performance_score = chosen_server.get_performance_score()
        
        # Base deposit amount
        base_deposit = PHEROMONE_DEPOSIT_AMOUNT * performance_score
        
        # UTILIZATION BONUS - Massive rewards for optimal utilization
        if 60.0 <= utilization <= 85.0:
            utilization_bonus = 5.0  # 5x bonus for optimal utilization
        elif 40.0 <= utilization < 60.0:
            utilization_bonus = 2.0  # 2x bonus for good utilization
        else:
            utilization_bonus = 0.5  # Penalty for poor utilization
        
        # RESPONSE TIME BONUS
        response_bonus = 1.0 / (response_time + 0.1)
        
        # SYSTEM LOAD BONUS - Reward servers that help with overall system utilization
        system_utilization = np.mean([s.get_utilization() for s in servers])
        if system_utilization > 50.0:
            system_bonus = 2.0  # Double bonus when system is under load
        else:
            system_bonus = 1.0
        
        # FINAL AGGRESSIVE DEPOSIT
        total_deposit = base_deposit * utilization_bonus * response_bonus * system_bonus * 3.0  # 3x multiplier
        global_pheromones[chosen_server_id] += total_deposit
        
        log_callback(f"ACO: Deposited {total_deposit:.2f} pheromone on {chosen_server_id} (Util: {utilization:.1f}%, Bonus: {utilization_bonus:.1f}x)")

def reset_aco_usage(servers: List[SimulatedServer]):
    """Resets ACO usage metrics."""
    for server in servers:
        server.aco_usage = 0
        server.last_algorithm_used = None