import numpy as np

def build_segments(points, closed=False):
    """
    Given an array of points (x, y), build a list of segments.
    If closed=True, connect the last point back to the first.
    """
    segments = []
    n = len(points)
    for i in range(n - 1):
        segments.append((points[i], points[i+1]))
    if closed and n > 2:
        segments.append((points[-1], points[0]))
    return segments


def line_segment_intersection_count(segments, theta, rho, tol=1e-12):
    """
    Count how many segments in `segments` are intersected by
    the line given by x*cos(theta) + y*sin(theta) = rho.

    segments: list of ((x1, y1), (x2, y2))
    theta, rho: line parameters in normal form
    tol: small tolerance for numerical comparisons.
    """
    c, s = np.cos(theta), np.sin(theta)
    count = 0
    for (x1, y1), (x2, y2) in segments:
        # Evaluate line equation for each endpoint: LHS - rho
        v1 = x1*c + y1*s - rho
        v2 = x2*c + y2*s - rho

        # If the signs differ (one positive, one negative), the line intersects
        # Or if either is extremely close to zero (within tol), treat it as an intersection
        if (v1 > tol and v2 < -tol) or (v1 < -tol and v2 > tol):
            count += 1
        else:
            # Optional: consider the case where an endpoint lies exactly on the line
            if abs(v1) < tol or abs(v2) < tol:
                count += 1

    return count


def hough_transform_max_intersections(points, 
                                      n_thetas=180, 
                                      n_rhos=200, 
                                      closed=False):
    """
    Perform a simple Hough-style search over (theta, rho) to find
    a line that intersects the given polygonal chain the most times.

    points: 2D array-like of shape (n, 2).
    n_thetas: number of discrete angle samples (0..pi).
    n_rhos: number of discrete rho samples (negative to positive).
    closed: whether to treat points as a closed loop.

    Returns:
        best_count, best_theta, best_rho
    """
    # Convert points to np.array if not already
    points = np.array(points, dtype=float)
    segments = build_segments(points, closed=closed)
    
    # 1) Determine a bounding radius so that rho covers all possible lines 
    #    that could intersect the bounding box of the points
    xs, ys = points[:,0], points[:,1]
    max_dist = np.sqrt(np.max(xs**2 + ys**2))  # max distance from origin
    # Or you could be even safer by taking a bounding box + margin:
    #   min_x, max_x = np.min(xs), np.max(xs)
    #   min_y, max_y = np.min(ys), np.max(ys)
    #   radius = np.sqrt(max_x**2 + max_y**2)
    # We'll just use max_dist for simplicity here.
    
    # 2) Build the discrete sampling arrays
    #    theta from [0, pi).  (We only need 0..pi because lines repeat after pi.)
    thetas = np.linspace(0, np.pi, n_thetas, endpoint=False)
    #    rho from [-max_dist, +max_dist]
    rhos = np.linspace(-max_dist, max_dist, n_rhos)
    
    best_count = -1
    best_theta = None
    best_rho = None
    
    # 3) Brute force over all sampled (theta, rho)
    for theta in thetas:
        for rho in rhos:
            count = line_segment_intersection_count(segments, theta, rho)
            if count > best_count:
                best_count = count
                best_theta = theta
                best_rho = rho
    
    return best_count, best_theta, best_rho


# ----------- DEMO USAGE -----------
if __name__ == "__main__":
    # Example: create a wiggly polygonal chain of random points
    np.random.seed(0)
    n_points = 50
    # random walk in 2D:
    step_x = np.random.randn(n_points).cumsum()
    step_y = np.random.randn(n_points).cumsum()
    points = np.stack([step_x, step_y], axis=1)
    
    # Three curves with different levels of wiggliness
    x = np.linspace(0, 10, 1000)
    y1 = np.sin(x)                    # Low wiggliness
    y2 = np.sin(x) + 0.5 * np.sin(3 * x)  # Medium wiggliness
    y3 = np.sin(x) + 0.5 * np.sin(3 * x) + 0.3 * np.sin(7 * x)  # High wiggliness
    points = np.stack([x, y3], axis=1)
    
    # Run the Hough transform approach
    max_intersect_count, theta_star, rho_star = hough_transform_max_intersections(
        points, 
        n_thetas=90,  # 180
        n_rhos=100,   # 200
        closed=False
    )
    
    # plot the curve
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 8))
    plt.plot(points[:, 0], points[:, 1], 'b-', label='Curve')
    plt.title('Curve with Maximum Intersections')
    plt.show()
    # plt.xlabel('X-axis')
    # plt.ylabel('Y-axis')
    # plt.axhline(0, color='gray', lw=0.5, ls='--')
    # plt.axvline(0, color='gray', lw=0.5, ls='--')
    # plt.grid()
    # plt.legend()
    
    print("Approximate maximum intersection count:", max_intersect_count)
    print("Line parameters that achieve it (theta, rho) =", (theta_star, rho_star))
