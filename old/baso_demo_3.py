import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import random
import sys
import os

# Import the adaptive model from the parent module
# Assuming the AdaptiveSchoolTestingModel is defined in another file
from baso_demo_2 import AdaptiveSchoolTestingModel

class PrioritizedSchoolTestingModel(AdaptiveSchoolTestingModel):
    """
    Enhanced model that focuses on efficiently identifying the top schools through
    dynamic prioritization and pruning with the ability to return to schools later.
    """
    
    def __init__(self, num_grades=12, initial_uncertainty=0.5, min_confidence=0.95,
                revisit_threshold=0.2):
        """
        Initialize the prioritized testing model.
        
        Parameters:
        -----------
        num_grades : int
            Total number of grade levels to test
        initial_uncertainty : float
            Initial uncertainty estimate (0-1) for our model before any data
        min_confidence : float
            Minimum confidence level required for making decisions (0-1)
        revisit_threshold : float
            Probability threshold above which a previously pruned school might be revisited
        """
        super().__init__(num_grades, initial_uncertainty, min_confidence)
        self.revisit_threshold = revisit_threshold
        
        # Additional tracking for prioritization
        self.school_rankings = []  # Current rankings of schools
        self.pruned_schools = []   # Schools that were pruned but could be revisited
        self.testing_history = []  # History of testing actions and rankings
        
    def add_grade_result(self, school_id, grade_index, score):
        """
        Add a single grade score result, update model estimates, and recalculate rankings.
        """
        # Call parent method to update distributions and correlations
        school_data = super().add_grade_result(school_id, grade_index, score)
        
        # Update school rankings after adding new data
        self._update_school_rankings()
        
        # Record this testing step in history
        self._record_testing_step(school_id, grade_index, score)
        
        return school_data
    
    def _update_school_rankings(self):
        """
        Recalculate the ranking of all schools based on current information.
        This includes partially tested schools, fully tested schools, and pruned schools
        that might be worth revisiting.
        """
        rankings = []
        
        # Process fully tested schools (we know their true average)
        for i, school in enumerate(self.schools_data):
            rankings.append({
                'school_id': school['school_id'],
                'status': 'COMPLETE',
                'average': school['average'],
                'estimated_average': school['average'],
                'confidence': 1.0,
                'grades_tested': self.num_grades,
                'probability_best': 0.0,  # Will be updated after all schools are processed
                'rank': 0  # Will be updated after sorting
            })
        
        # Process partially tested schools
        for school in self.partial_schools:
            # Need at least 2 grades to make a prediction
            if len(school['sampled_grades']) >= 2:
                # Run evaluation to get estimate
                eval_result = self.evaluate_school(school['school_id'])
                
                # Check if estimated_average exists in the result
                estimated_average = eval_result.get('estimated_average')
                
                # If it doesn't exist, calculate it from the mean of sampled grades
                if estimated_average is None:
                    estimated_average = np.mean([
                        school['grade_scores'][g] for g in school['sampled_grades']
                    ])
                
                rankings.append({
                    'school_id': school['school_id'],
                    'status': 'PARTIAL',
                    'average': None,  # Unknown true average
                    'estimated_average': estimated_average,
                    'confidence': eval_result.get('confidence', 0.0),
                    'grades_tested': len(school['sampled_grades']),
                    'probability_best': 0.0,  # Will be updated
                    'rank': 0  # Will be updated after sorting
                })
            else:
                # Too few grades to estimate properly, assign average rank
                rankings.append({
                    'school_id': school['school_id'],
                    'status': 'PARTIAL',
                    'average': None,
                    'estimated_average': np.mean([
                        school['grade_scores'][g] for g in school['sampled_grades']
                    ]) if school['sampled_grades'] else np.mean(self.grade_means),
                    'confidence': 0.0,
                    'grades_tested': len(school['sampled_grades']),
                    'probability_best': 0.0,
                    'rank': 0
                })
        
        # Process pruned schools that might be worth revisiting
        for school in self.pruned_schools:
            rankings.append({
                'school_id': school['school_id'],
                'status': 'PRUNED',
                'average': None,
                'estimated_average': school['estimated_average'],
                'confidence': school['confidence'],
                'grades_tested': school['grades_tested'],
                'probability_best': 0.0,
                'rank': 0
            })
        
        # First, sort by estimated average
        rankings.sort(key=lambda x: x['estimated_average'], reverse=True)
        
        # Assign initial ranks
        for i, r in enumerate(rankings):
            r['rank'] = i + 1
        
        # Calculate probability of being best school using simulation
        if rankings:
            self._calculate_best_probabilities(rankings)
        
        # Final sort by probability of being best
        rankings.sort(key=lambda x: x['probability_best'], reverse=True)
        
        # Update ranks after final sort
        for i, r in enumerate(rankings):
            r['rank'] = i + 1
        
        self.school_rankings = rankings
        
    def _calculate_best_probabilities(self, rankings, num_simulations=1000):
        """
        Calculate the probability that each school is the best using Monte Carlo simulation.
        
        Parameters:
        -----------
        rankings : list
            List of school ranking dictionaries
        num_simulations : int
            Number of Monte Carlo simulations to run
        """
        # Best known average so far (from fully tested schools)
        best_known_average = max(
            [r['average'] for r in rankings if r['status'] == 'COMPLETE'],
            default=float('-inf')
        )
        
        # Collect all schools that need simulation (partial and pruned)
        schools_to_simulate = [r for r in rankings if r['status'] != 'COMPLETE']
        
        if not schools_to_simulate:
            # All schools are fully tested, assign best school probability=1.0
            best_index = np.argmax([r['average'] for r in rankings])
            rankings[best_index]['probability_best'] = 1.0
            return
        
        # For each school, simulate potential final averages
        simulated_averages = {}
        
        for school in schools_to_simulate:
            school_id = school['school_id']
            
            # Find this school's data
            school_data = None
            confidence = school['confidence']
            
            # Look in partial schools first
            for s in self.partial_schools:
                if s['school_id'] == school_id:
                    school_data = s
                    break
            
            # If not found, check pruned schools
            if school_data is None:
                for s in self.pruned_schools:
                    if s['school_id'] == school_id:
                        school_data = s
                        break
            
            # Skip if school data not found
            if school_data is None:
                simulated_averages[school_id] = np.full(num_simulations, 
                                                      school['estimated_average'])
                continue
            
            # For partial schools, use the evaluation method to simulate
            if isinstance(school_data, dict) and 'sampled_grades' in school_data:
                # Check if we have enough grades
                if len(school_data['sampled_grades']) >= 2:
                    # Only call evaluate_school for schools in partial_schools
                    is_partial = any(s['school_id'] == school_id for s in self.partial_schools)
                    
                    if is_partial:
                        eval_result = self.evaluate_school(school_id, num_simulations=num_simulations)
                        if 'full_averages' in eval_result:
                            simulated_averages[school_id] = eval_result['full_averages']
                        else:
                            # Fallback if simulation failed
                            simulated_averages[school_id] = np.random.normal(
                                school['estimated_average'], 
                                5.0 * (1.0 - confidence),
                                num_simulations
                            )
                    else:
                        # For pruned schools, use saved estimates
                        simulated_averages[school_id] = np.random.normal(
                            school['estimated_average'],
                            5.0 * (1.0 - confidence),
                            num_simulations
                        )
                else:
                    # Too few grades, use wider distribution
                    simulated_averages[school_id] = np.random.normal(
                        school['estimated_average'], 
                        10.0,
                        num_simulations
                    )
            else:
                # For pruned schools, use saved estimates
                simulated_averages[school_id] = np.random.normal(
                    school['estimated_average'],
                    5.0 * (1.0 - confidence),
                    num_simulations
                )
        
        # Now run simulations to determine probability of each school being best
        win_counts = {school_id: 0 for school_id in simulated_averages.keys()}
        
        for sim in range(num_simulations):
            # Collect simulated averages for this iteration
            sim_results = {
                school_id: simulated_averages[school_id][sim] 
                for school_id in simulated_averages.keys()
            }
            
            # Add fully tested schools with known averages
            for r in rankings:
                if r['status'] == 'COMPLETE':
                    sim_results[r['school_id']] = r['average']
            
            # Find the best school in this simulation
            best_school_id = max(sim_results.items(), key=lambda x: x[1])[0]
            win_counts[best_school_id] = win_counts.get(best_school_id, 0) + 1
        
        # Update probabilities in the rankings
        for r in rankings:
            school_id = r['school_id']
            if school_id in win_counts:
                r['probability_best'] = win_counts[school_id] / num_simulations
            elif r['status'] == 'COMPLETE':
                # For fully tested schools not in win_counts
                r['probability_best'] = 0.0
        
    def prune_school(self, school_id):
        """
        Prune a school from active testing, but save its information 
        for possible revisiting later.
        
        Parameters:
        -----------
        school_id : str or int
            Identifier for the school to prune
            
        Returns:
        --------
        dict
            Information about the pruned school
        """
        # Find school in partial schools
        school_index = None
        for i, school in enumerate(self.partial_schools):
            if school['school_id'] == school_id:
                school_index = i
                break
                
        if school_index is None:
            raise ValueError(f"School {school_id} not found in partial schools")
            
        # Get current evaluation
        eval_result = self.evaluate_school(school_id)
        
        # Create pruned school record
        pruned_school = {
            'school_id': school_id,
            'estimated_average': eval_result['estimated_average'],
            'confidence': eval_result['confidence'],
            'grades_tested': len(self.partial_schools[school_index]['sampled_grades']),
            'sampled_grades': self.partial_schools[school_index]['sampled_grades'].copy(),
            'grade_scores': self.partial_schools[school_index]['grade_scores'].copy(),
            'pruned_at': len(self.testing_history),
            'probability_best': next((r['probability_best'] for r in self.school_rankings 
                                   if r['school_id'] == school_id), 0.0)
        }
        
        # Add to pruned schools
        self.pruned_schools.append(pruned_school)
        
        # Remove from partial schools
        self.partial_schools.pop(school_index)
        
        # Update rankings
        self._update_school_rankings()
        
        return pruned_school
        
    def revisit_school(self, school_id):
        """
        Revisit a previously pruned school and add it back to active testing.
        
        Parameters:
        -----------
        school_id : str or int
            Identifier for the school to revisit
            
        Returns:
        --------
        dict
            The reactivated school data
        """
        # Find school in pruned schools
        school_index = None
        for i, school in enumerate(self.pruned_schools):
            if school['school_id'] == school_id:
                school_index = i
                break
                
        if school_index is None:
            raise ValueError(f"School {school_id} not found in pruned schools")
            
        # Create partial school record
        partial_school = {
            'school_id': school_id,
            'grade_scores': self.pruned_schools[school_index]['grade_scores'].copy(),
            'sampled_grades': self.pruned_schools[school_index]['sampled_grades'].copy()
        }
        
        # Add to partial schools
        self.partial_schools.append(partial_school)
        
        # Remove from pruned schools
        self.pruned_schools.pop(school_index)
        
        # Update rankings
        self._update_school_rankings()
        
        return partial_school
    
    def get_next_school_to_test(self):
        """
        Determine the next school to test based on current rankings and status.
        
        Returns:
        --------
        dict
            Information about the next school to test, including the next grade
            to test for that school
        """
        if not self.school_rankings:
            return None
            
        # Prioritize schools with higher probability of being best
        for ranking in self.school_rankings:
            if ranking['status'] == 'PARTIAL':
                school_id = ranking['school_id']
                # Find school in partial schools
                for school in self.partial_schools:
                    if school['school_id'] == school_id:
                        next_grade = self.get_next_grade_to_sample(school_id, school['sampled_grades'])
                        return {
                            'school_id': school_id,
                            'next_grade': next_grade,
                            'current_rank': ranking['rank'],
                            'probability_best': ranking['probability_best'],
                            'grades_tested': ranking['grades_tested']
                        }
                        
        # If no partial schools, check if any pruned schools should be revisited
        for ranking in self.school_rankings:
            if ranking['status'] == 'PRUNED' and ranking['probability_best'] >= self.revisit_threshold:
                school_id = ranking['school_id']
                # Revisit this school
                self.revisit_school(school_id)
                # Find in partial schools now
                for school in self.partial_schools:
                    if school['school_id'] == school_id:
                        next_grade = self.get_next_grade_to_sample(school_id, school['sampled_grades'])
                        return {
                            'school_id': school_id,
                            'next_grade': next_grade,
                            'current_rank': ranking['rank'],
                            'probability_best': ranking['probability_best'],
                            'grades_tested': ranking['grades_tested'],
                            'revisited': True
                        }
        
        return None
    
    def _record_testing_step(self, school_id, grade_index, score):
        """
        Record a testing step in the history.
        
        Parameters:
        -----------
        school_id : str or int
            Identifier for the school
        grade_index : int
            Index of the grade that was tested
        score : float
            Score that was recorded
        """
        # Get current top schools
        top_schools = self.school_rankings[:min(5, len(self.school_rankings))]
        
        # Create history entry
        entry = {
            'step': len(self.testing_history) + 1,
            'school_id': school_id,
            'grade_tested': grade_index + 1,  # Convert to 1-indexed
            'score': score,
            'top_schools': [
                {
                    'rank': s['rank'],
                    'school_id': s['school_id'],
                    'status': s['status'],
                    'probability_best': s['probability_best'],
                    'estimated_average': s['estimated_average']
                }
                for s in top_schools
            ],
            'total_tests': sum(r['grades_tested'] for r in self.school_rankings),
            'model_state': {
                'distribution_confidence': np.mean(self.distribution_confidence),
                'correlation_confidence': self.correlation_confidence
            }
        }
        
        self.testing_history.append(entry)
        
    def get_testing_summary(self):
        """
        Get a summary of the current testing state.
        
        Returns:
        --------
        dict
            Summary information
        """
        total_schools = len(self.schools_data) + len(self.partial_schools) + len(self.pruned_schools)
        total_possible_tests = total_schools * self.num_grades
        tests_conducted = sum(s['grades_tested'] for s in self.school_rankings)
        
        # Best school so far
        best_school = self.school_rankings[0] if self.school_rankings else None
        
        summary = {
            'total_schools': total_schools,
            'schools_complete': len(self.schools_data),
            'schools_partial': len(self.partial_schools),
            'schools_pruned': len(self.pruned_schools),
            'tests_conducted': tests_conducted,
            'tests_possible': total_possible_tests,
            'efficiency': 1 - (tests_conducted / total_possible_tests) if total_possible_tests > 0 else 0,
            'best_school': best_school,
            'model_confidence': {
                'distribution': np.mean(self.distribution_confidence),
                'correlation': self.correlation_confidence
            },
            'testing_progress': len(self.testing_history)
        }
        
        return summary
    
    def visualize_rankings(self, show_top_n=10):
        """
        Visualize the current school rankings.
        
        Parameters:
        -----------
        show_top_n : int
            Number of top schools to display
            
        Returns:
        --------
        matplotlib.pyplot
            Plot of the current rankings
        """
        if not self.school_rankings:
            print("No schools to visualize yet")
            return None
            
        # Get top N schools
        top_schools = self.school_rankings[:min(show_top_n, len(self.school_rankings))]
        
        # Set up plot
        plt.figure(figsize=(14, 8))
        
        # Create color map for different statuses
        status_colors = {
            'COMPLETE': 'green',
            'PARTIAL': 'blue',
            'PRUNED': 'orange'
        }
        
        # Bar heights = probability of being best
        school_ids = [s['school_id'] for s in top_schools]
        probs = [s['probability_best'] for s in top_schools]
        colors = [status_colors[s['status']] for s in top_schools]
        
        # Create bars
        bars = plt.bar(range(len(top_schools)), probs, color=colors)
        
        # Add text annotations
        for i, (bar, school) in enumerate(zip(bars, top_schools)):
            # Add school ID
            plt.text(i, 0.02, school['school_id'].split('_')[-1], 
                    ha='center', color='black', fontweight='bold', fontsize=10)
            
            # Add estimated average
            plt.text(i, bar.get_height() + 0.02, 
                    f"{school['estimated_average']:.1f}", 
                    ha='center', va='bottom', fontsize=9)
            
            # Add grades tested
            plt.text(i, bar.get_height() / 2, 
                    f"{school['grades_tested']}/{self.num_grades}", 
                    ha='center', va='center', color='white', fontweight='bold')
            
        # Add legend
        legend_elements = [
            Patch(facecolor='green', label='Fully Tested'),
            Patch(facecolor='blue', label='Partially Tested'),
            Patch(facecolor='orange', label='Pruned')
        ]
        plt.legend(handles=legend_elements, loc='upper right')
        
        # Set labels and title
        plt.xlabel('School Rank')
        plt.ylabel('Probability of Being Best School')
        plt.title('Current School Rankings (Top contenders for best school)')
        plt.xticks(range(len(top_schools)), [str(s['rank']) for s in top_schools])
        plt.ylim(0, 1)
        plt.grid(axis='y', alpha=0.3)
        
        return plt
    
    def visualize_testing_progress(self):
        """
        Visualize how the testing has progressed over time.
        
        Returns:
        --------
        matplotlib.pyplot
            Plot of the testing progress
        """
        if not self.testing_history:
            print("No testing history to visualize yet")
            return None
            
        # Set up plot
        plt.figure(figsize=(15, 10))
        
        # Extract data from history
        steps = [h['step'] for h in self.testing_history]
        
        # Track top school changes
        top_schools = []
        top_school_probs = []
        
        for h in self.testing_history:
            if h['top_schools']:
                top_schools.append(h['top_schools'][0]['school_id'])
                top_school_probs.append(h['top_schools'][0]['probability_best'])
            else:
                top_schools.append(None)
                top_school_probs.append(0)
        
        # Plot 1: Top school probability over time
        plt.subplot(2, 2, 1)
        plt.plot(steps, top_school_probs, 'b-', linewidth=2)
        plt.xlabel('Testing Step')
        plt.ylabel('Probability')
        plt.title('Probability of Top-Ranked School Being Best')
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Number of tests per grade
        plt.subplot(2, 2, 2)
        plt.bar(range(1, self.num_grades + 1), self.grade_sample_counts)
        plt.xlabel('Grade')
        plt.ylabel('Number of Tests')
        plt.title('Tests Conducted per Grade')
        plt.xticks(range(1, self.num_grades + 1))
        plt.grid(True, alpha=0.3)
        
        # Plot 3: School status progression
        testing_counts = [h['total_tests'] for h in self.testing_history]
        model_confidence = [h['model_state']['distribution_confidence'] for h in self.testing_history]
        
        plt.subplot(2, 2, 3)
        plt.plot(steps, testing_counts, 'g-', linewidth=2)
        plt.xlabel('Testing Step')
        plt.ylabel('Total Tests')
        plt.title('Testing Progress')
        plt.grid(True, alpha=0.3)
        
        # Plot 4: Top school changes
        changes = [i for i in range(1, len(top_schools)) 
                  if top_schools[i] != top_schools[i-1]]
        plt.subplot(2, 2, 4)
        
        if changes:
            change_x = [steps[i] for i in changes]
            change_y = [top_school_probs[i] for i in changes]
            plt.scatter(change_x, change_y, color='red', s=50, zorder=3, 
                       label='Leader Changed')
            
        plt.plot(steps, top_school_probs, 'b-', linewidth=1, alpha=0.7)
        plt.xlabel('Testing Step')
        plt.ylabel('Probability')
        plt.title('Top School Changes')
        if changes:
            plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return plt

def demonstrate_prioritized_testing():
    """
    Run a demonstration of the prioritized testing approach and visualize the results.
    """
    print("Starting prioritized testing simulation...")
    
    # Create model
    model = PrioritizedSchoolTestingModel(
        num_grades=12,
        initial_uncertainty=0.5,
        min_confidence=0.85,
        revisit_threshold=0.15
    )
    
    # Generate true scores for schools
    np.random.seed(42)
    num_schools = 50
    
    # Define true grade means and standard deviations
    true_grade_means = np.linspace(70, 90, model.num_grades)  # Increasing means by grade
    true_grade_stds = np.full(model.num_grades, 10)  # Equal std devs
    true_correlation = 0.7
    
    # Generate true scores for all schools
    true_scores = []
    for s in range(num_schools):
        # Generate correlated scores for this school
        shared_factor = np.random.normal(0, 1)  # School-specific factor
        unique_factors = np.random.normal(0, 1, model.num_grades)  # Grade-specific factors
        
        # Mix factors based on desired correlation
        z_scores = shared_factor * np.sqrt(true_correlation) + \
                  unique_factors * np.sqrt(1 - true_correlation)
        
        # Scale to actual scores
        scores = true_grade_means + z_scores * true_grade_stds
        
        # Calculate true average
        average = np.mean(scores)
        
        true_scores.append({
            'school_id': f"School_{s+1}",
            'grade_scores': scores,
            'average': average
        })
    
    # Sort by true average (descending)
    true_scores.sort(key=lambda x: x['average'], reverse=True)
    
    # Store true rankings for later comparison
    true_rankings = {s['school_id']: i+1 for i, s in enumerate(true_scores)}
    
    # Tracking variables
    max_tests = 250  # Limit total number of tests for demonstration
    tests_conducted = 0
    
    # Start with a few random schools to build initial model
    initial_schools = 5
    initial_grades_per_school = 2
    
    print(f"Starting with {initial_schools} schools, {initial_grades_per_school} grades each...")
    
    # Test initial schools to build model
    for i in range(initial_schools):
        school = true_scores[i]
        school_id = school['school_id']
        
        for j in range(initial_grades_per_school):
            # Randomly select a grade for initial testing
            grade_index = np.random.randint(0, model.num_grades)
            score = school['grade_scores'][grade_index]
            
            # Add result to model
            model.add_grade_result(school_id, grade_index, score)
            tests_conducted += 1
    
    # Update rankings after initial tests
    model._update_school_rankings()
    
    # Visualization points - show progress at these steps
    viz_points = [25, 50, 100, 150, 200, max_tests]
    
    # Track discovered best school at each viz point
    discovered_best = {}
    
    # Main testing loop - use prioritization
    print(f"Beginning prioritized testing (max {max_tests} tests)...")
    
    pruning_threshold = 0.03  # Probability threshold for pruning
    
    while tests_conducted < max_tests:
        # Get summary and show progress every 25 tests
        if tests_conducted in viz_points or tests_conducted % 25 == 0:
            summary = model.get_testing_summary()
            print(f"\nTests conducted: {tests_conducted}")
            print(f"Schools: {summary['schools_complete']} complete, " 
                 f"{summary['schools_partial']} partial, "
                 f"{summary['schools_pruned']} pruned")
            print(f"Best school so far: {summary['best_school']['school_id']} "
                 f"(P(best)={summary['best_school']['probability_best']:.2f}, "
                 f"Estimated avg={summary['best_school']['estimated_average']:.1f})")
            
            # Store discovered best at this point
            discovered_best[tests_conducted] = {
                'school_id': summary['best_school']['school_id'],
                'probability': summary['best_school']['probability_best'],
                'true_rank': true_rankings[summary['best_school']['school_id']]
            }
            
            # Generate and save visualization
            if tests_conducted in viz_points:
                model.visualize_rankings()
                plt.savefig(f"ranking_after_{tests_conducted}_tests.png")
                plt.close()
        
        # Determine which school to test next
        next_school = model.get_next_school_to_test()
        
        if next_school is None:
            # If no schools to test, add a new one
            new_index = len(model.schools_data) + len(model.partial_schools) + len(model.pruned_schools)
            if new_index < len(true_scores):
                school_id = true_scores[new_index]['school_id']
                # Pick a random grade to start with
                grade_index = np.random.randint(0, model.num_grades)
                score = true_scores[new_index]['grade_scores'][grade_index]
                model.add_grade_result(school_id, grade_index, score)
                tests_conducted += 1
            else:
                # No more schools to add
                print("All schools have been considered")
                break
        else:
            # Test next grade for the selected school
            school_id = next_school['school_id']
            grade_index = next_school['next_grade']
            
            # Find this school in true scores
            school_data = None
            for s in true_scores:
                if s['school_id'] == school_id:
                    school_data = s
                    break
            
            if school_data:
                score = school_data['grade_scores'][grade_index]
                model.add_grade_result(school_id, grade_index, score)
                tests_conducted += 1
                
                # Check if we should prune this school
                current_ranking = next((r for r in model.school_rankings 
                                     if r['school_id'] == school_id), None)
                                     
                if (current_ranking and 
                    current_ranking['status'] == 'PARTIAL' and
                    current_ranking['probability_best'] < pruning_threshold and
                    current_ranking['confidence'] >= 0.7 and
                    current_ranking['grades_tested'] >= 3):
                    
                    model.prune_school(school_id)
            else:
                print(f"Error: Could not find school {school_id} in true data")
    
    # Final visualizations
    print("\nTesting complete.")
    final_summary = model.get_testing_summary()
    print(f"Final status: {tests_conducted} tests used")
    print(f"Schools: {final_summary['schools_complete']} complete, " 
         f"{final_summary['schools_partial']} partial, "
         f"{final_summary['schools_pruned']} pruned")
    
    # Compare discovered rankings with true rankings
    print("\nEvaluation of results:")
    best_school_id = model.school_rankings[0]['school_id']
    true_best = true_scores[0]['school_id']
    
    print(f"True best school: {true_best} (avg: {true_scores[0]['average']:.1f})")
    print(f"Discovered best school: {best_school_id} (avg: {model.school_rankings[0]['estimated_average']:.1f})")
    
    # Show final rankings
    model.visualize_rankings()
    plt.show()
    
    # Show testing progress
    model.visualize_testing_progress()
    plt.show()
    
    return model, discovered_best

if __name__ == "__main__":
    demonstrate_prioritized_testing()