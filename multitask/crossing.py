import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import random

def count_line_curve_intersections(x_values, y_values, num_trials=100):
    """
    Estimate the complexity of a curve by:
    1. Randomly selecting two points on the curve
    2. Drawing a straight line between them
    3. Counting intersections between this line and the curve
    4. Repeating and tracking the maximum count
    
    Args:
        x_values: x-coordinates of the curve points
        y_values: y-coordinates of the curve points
        num_trials: number of random line trials
    
    Returns:
        max_intersections: maximum number of intersections found
    """
    max_intersections = 0
    
    # Convert to numpy arrays if they aren't already
    x_values = np.array(x_values)
    y_values = np.array(y_values)
    n_points = len(x_values)
    
    for _ in range(num_trials):
        # Pick two random indices (ensuring they're different)
        idx1, idx2 = random.sample(range(n_points), 2)
        
        # Get the two points on the curve
        x1, y1 = x_values[idx1], y_values[idx1]
        x2, y2 = x_values[idx2], y_values[idx2]
        
        # Skip if the x-values are the same (vertical line case)
        if x1 == x2:
            continue
        
        # Create a linear function between these two points
        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1
        
        # Generate y-values for the straight line at each x-value
        line_y = slope * x_values + intercept
        
        # Calculate difference between curve and line
        diff = y_values - line_y
        
        # Count sign changes (intersections)
        # We only count where diff is not zero to avoid counting tangent points
        non_zero_diff = diff[diff != 0]
        if len(non_zero_diff) > 0:
            intersections = np.sum(np.diff(np.signbit(non_zero_diff)) != 0)
            
            # We need to add 1 to account for the first intersection
            # (since diff() reduces the length by 1)
            intersections += 1
            
            # Account for the two points we selected (they're not true intersections)
            intersections = max(0, intersections - 2)
            
            max_intersections = max(max_intersections, intersections)
    
    return max_intersections

# Example usage:
def demonstrate_with_example():
    # Generate example data with increasing wiggliness
    x = np.linspace(0, 10, 1000)
    
    # Three curves with different levels of wiggliness
    y1 = np.sin(x)                    # Low wiggliness
    y2 = np.sin(x) + 0.5 * np.sin(3 * x)  # Medium wiggliness
    y3 = np.sin(x) + 0.5 * np.sin(3 * x) + 0.3 * np.sin(7 * x)  # High wiggliness
    
    # Compute wiggliness scores
    score1 = count_line_curve_intersections(x, y1, num_trials=500)
    score2 = count_line_curve_intersections(x, y2, num_trials=500)
    score3 = count_line_curve_intersections(x, y3, num_trials=500)
    
    print(f"Wiggliness scores:")
    print(f"Curve 1 (low): {score1}")
    print(f"Curve 2 (medium): {score2}")
    print(f"Curve 3 (high): {score3}")
    
    # Plot the curves
    plt.figure(figsize=(10, 6))
    plt.plot(x, y1, label=f"Low wiggliness (score={score1})")
    plt.plot(x, y2, label=f"Medium wiggliness (score={score2})")
    plt.plot(x, y3, label=f"High wiggliness (score={score3})")
    plt.legend()
    plt.title("Curves with Different Wiggliness Levels")
    plt.show()

if __name__ == "__main__":
    # Demonstration
    demonstrate_with_example()