import time
import functools
from collections import defaultdict

class TimeTracker:
    """Class to track time spent in decorated functions."""
    
    def __init__(self):
        self.function_times = defaultdict(float)
        self.function_calls = defaultdict(int)
        self.start_time = time.time()
    
    def __call__(self, func):
        """Make this class usable as a decorator."""
        func_name = func.__name__
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            elapsed = time.time() - start_time
            
            # Update timing statistics
            self.function_times[func_name] += elapsed
            self.function_calls[func_name] += 1
            
            return result
        
        return wrapper
    
    def get_stats(self):
        """Get timing statistics for all tracked functions."""
        total_runtime = time.time() - self.start_time
        
        stats = {}
        for func_name in self.function_times:
            func_time = self.function_times[func_name]
            stats[func_name] = {
                "total_time": round(func_time, 4),
                "call_count": self.function_calls[func_name],
                "avg_time": round(func_time / self.function_calls[func_name], 4) if self.function_calls[func_name] > 0 else 0,
                "fraction": round(func_time / total_runtime, 4) if total_runtime > 0 else 0
            }
        
        return stats
    
    def print_stats(self, sort_by="total_time"):
        """Print a formatted report of function timing statistics."""
        stats = self.get_stats()
        
        # Sort functions by the specified metric
        sorted_funcs = sorted(stats.keys(), key=lambda x: stats[x][sort_by], reverse=True)
        
        print(f"\n{'Function':<30} {'Calls':<10} {'Total Time':<15} {'Avg Time':<15} {'% of Runtime':<15}")
        print("-" * 85)
        
        for func_name in sorted_funcs:
            func_stats = stats[func_name]
            print(f"{func_name:<30} {func_stats['call_count']:<10} "
                  f"{func_stats['total_time']:<15} {func_stats['avg_time']:<15} "
                  f"{func_stats['fraction']*100:<15}")

# Example usage
if __name__ == "__main__":
    # Create a tracker instance
    tracker = TimeTracker()

    # Decorate functions to track
    @tracker
    def example_function():
        time.sleep(0.1)

    @tracker
    def another_function():
        time.sleep(0.2)
        example_function()

    # Call the functions
    example_function()
    another_function()
    example_function()

    # Print stats
    tracker.print_stats()