'''
BASO : Bayesian Adaptive Sampling for Optimization
'''

import numpy as np
import matplotlib.pyplot as plt
import random
# import pandas as pd
# from scipy import stats
# from scipy.stats import multivariate_normal
# from sklearn.preprocessing import StandardScaler

class AdaptiveSchoolTestingModel:
    def __init__(self, num_grades=12, initial_uncertainty=0.5, min_confidence=0.95):
        """
        Initialize the adaptive statistical model for school testing optimization.
        
        Parameters:
        -----------
        num_grades : int
            Total number of grade levels to test
        initial_uncertainty : float
            Initial uncertainty estimate (0-1) for our model before any data
        min_confidence : float
            Minimum confidence level required for making decisions (0-1)
        """
        self.num_grades = num_grades
        self.initial_uncertainty = initial_uncertainty
        self.min_confidence = min_confidence
        
        # Data structures for collected information
        self.schools_data = []  # Complete data for fully tested schools
        self.partial_schools = []  # Data for schools we haven't fully tested
        self.best_school_index = None
        self.best_school_average = float('-inf')
        
        # Tracking grade-specific information
        self.grade_means = np.zeros(num_grades)
        self.grade_stds = np.ones(num_grades) * initial_uncertainty * 100  # Initial wide uncertainty
        self.grade_sample_counts = np.zeros(num_grades, dtype=int)
        
        # Correlation matrix starts with moderate assumed correlations
        self.correlation_matrix = np.eye(num_grades)
        for i in range(num_grades):
            for j in range(num_grades):
                if i != j:
                    # Assume moderate correlation initially
                    self.correlation_matrix[i, j] = 0.5
        
        # Confidence-related tracking
        self.distribution_confidence = np.zeros(num_grades)  # Confidence in our estimates per grade
        self.correlation_confidence = 0.0  # Overall confidence in our correlation estimates
    
    #-----------------------------------------------------------------------------------------------------------------
    # NEW CODE    
    def get_next_grade_to_sample(self, school_id, already_sampled_grades=None):
        """
        Determine the optimal next grade to sample based on current uncertainties
        and school-specific information.
        
        Parameters:
        -----------
        school_id : str or int
            Identifier for the school
        already_sampled_grades : list or None
            List of grades already sampled for this school
            
        Returns:
        --------
        int
            The next grade index to sample (0-indexed)
        """
        if already_sampled_grades is None:
            already_sampled_grades = []
        
        if len(already_sampled_grades) == self.num_grades:
            return None  # All grades already sampled
            
        # Calculate sampling priority for each grade
        remaining_grades = [g for g in range(self.num_grades) if g not in already_sampled_grades]
        
        # Find this school in partial schools
        school_data = None
        for s in self.partial_schools:
            if s['school_id'] == school_id:
                school_data = s
                break
        
        # Consider five factors:
        # 1. Global uncertainty (fewer samples overall = higher priority)
        # 2. Potential information gain (higher std = higher priority)
        # 3. Predictive power (grades with higher correlation to already tested grades = higher priority)
        # 4. School-specific performance patterns (if available)
        # 5. Random component to ensure exploration
        
        priorities = []
        for g in remaining_grades:
            # 1. Normalize sample count to 0-1 range (inverse - fewer samples = higher priority)
            sample_priority = 1.0 / (1.0 + self.grade_sample_counts[g])
            
            # 2. Standard deviation represents uncertainty in our estimate
            uncertainty_priority = self.grade_stds[g] / max(1.0, np.max(self.grade_stds))
            
            # 3. Correlation-based priority (if we have sampled grades for this school)
            correlation_priority = 0.0
            if school_data and already_sampled_grades:
                # Calculate average correlation between this grade and already sampled grades
                correlations = [abs(self.correlation_matrix[g, sg]) for sg in already_sampled_grades]
                # High correlation = more predictive value = lower priority (we want diverse information)
                avg_correlation = np.mean(correlations) if correlations else 0
                correlation_priority = 1.0 - avg_correlation  # Invert so lower correlation = higher priority
            
            # 4. School-specific pattern (if we have data from other grades in this school)
            school_pattern_priority = 0.0
            if school_data and already_sampled_grades:
                # Check if this school has a pattern of performance
                # If already-sampled grades perform above/below average, prioritize grades that might
                # confirm/challenge this pattern
                z_scores = []
                for sg in already_sampled_grades:
                    score = school_data['grade_scores'][sg]
                    if score is not None:
                        z = (score - self.grade_means[sg]) / max(1e-6, self.grade_stds[sg])
                        z_scores.append(z)
                
                if z_scores:
                    # If school is performing consistently (all scores in same direction),
                    # prioritize grades that would give us more confidence in our estimate
                    avg_z = np.mean(z_scores)
                    if abs(avg_z) > 0.5:  # If there's a clear pattern
                        # Prioritize grades with higher variance when performance is extreme
                        # This helps confirm/reject our hypothesis more quickly
                        normalized_std = self.grade_stds[g] / max(1.0, np.max(self.grade_stds))
                        school_pattern_priority = normalized_std * min(1.0, abs(avg_z) / 2)
            
            # 5. Small random component (10%) to prevent getting stuck in patterns
            random_component = 0.1 * random.random()
            
            # Combined priority score - weights can be adjusted based on importance
            priority = (
                0.25 * sample_priority + 
                0.30 * uncertainty_priority + 
                0.20 * correlation_priority +
                0.15 * school_pattern_priority +
                random_component
            )
            priorities.append((g, priority))
        
        # Select the grade with highest priority
        priorities.sort(key=lambda x: x[1], reverse=True)
        return priorities[0][0]
    
    #-----------------------------------------------------------------------------------------------------------------
    # ORIGINAL CODE
    # def get_next_grade_to_sample(self, school_id, already_sampled_grades=None):
    #     """
    #     Determine the optimal next grade to sample based on current uncertainties.
        
    #     Parameters:
    #     -----------
    #     school_id : str or int
    #         Identifier for the school
    #     already_sampled_grades : list or None
    #         List of grades already sampled for this school
            
    #     Returns:
    #     --------
    #     int
    #         The next grade index to sample (0-indexed)
    #     """
    #     if already_sampled_grades is None:
    #         already_sampled_grades = []
        
    #     if len(already_sampled_grades) == self.num_grades:
    #         return None  # All grades already sampled
            
    #     # Calculate sampling priority for each grade
    #     remaining_grades = [g for g in range(self.num_grades) if g not in already_sampled_grades]
        
    #     # Consider three factors:
    #     # 1. Global uncertainty (fewer samples overall = higher priority)
    #     # 2. Potential information gain (higher std = higher priority)
    #     # 3. Random component to ensure exploration
        
    #     priorities = []
    #     for g in remaining_grades:
    #         # Normalize sample count to 0-1 range (inverse - fewer samples = higher priority)
    #         sample_priority = 1.0 / (1.0 + self.grade_sample_counts[g])
            
    #         # Standard deviation represents uncertainty in our estimate
    #         uncertainty_priority = self.grade_stds[g] / max(1.0, np.max(self.grade_stds))
            
    #         # Small random component (10%) to prevent getting stuck in patterns
    #         random_component = 0.1 * random.random()
            
    #         # Combined priority score
    #         priority = 0.4 * sample_priority + 0.5 * uncertainty_priority + random_component
    #         priorities.append((g, priority))
        
    #     # Select the grade with highest priority
    #     priorities.sort(key=lambda x: x[1], reverse=True)
    #     return priorities[0][0]
    #-----------------------------------------------------------------------------------------------------------------
        
    def add_grade_result(self, school_id, grade_index, score):
        """
        Add a single grade score result and update model estimates.
        
        Parameters:
        -----------
        school_id : str or int
            Identifier for the school
        grade_index : int
            Index of the grade (0-indexed, where 0 = 1st grade)
        score : float
            Test score for this grade
            
        Returns:
        --------
        dict
            Updated school data including sampled grades
        """
        # Find or create school record
        school_data = None
        for s in self.partial_schools:
            if s['school_id'] == school_id:
                school_data = s
                break
                
        if school_data is None:
            # New school
            school_data = {
                'school_id': school_id,
                'grade_scores': [None] * self.num_grades,
                'sampled_grades': []
            }
            self.partial_schools.append(school_data)
        
        # Add the new score
        school_data['grade_scores'][grade_index] = score
        school_data['sampled_grades'].append(grade_index)
        
        # Update our distribution estimates
        self._update_grade_distribution(grade_index, score)
        
        # If we have at least 2 grades from this school, update correlation
        if len(school_data['sampled_grades']) >= 2:
            self._update_correlations(school_data)
            
        # If we've sampled all grades for this school, move to completed schools
        if len(school_data['sampled_grades']) == self.num_grades:
            average = np.mean(school_data['grade_scores'])
            school_data['average'] = average
            
            # Check if this is the new best school
            if average > self.best_school_average:
                self.best_school_average = average
                self.best_school_index = len(self.schools_data)
                
            # Move from partial to complete
            self.schools_data.append(school_data)
            self.partial_schools.remove(school_data)
            
        # Recalculate confidence levels after updates
        self._update_confidence_levels()
            
        return school_data
        
    def _update_grade_distribution(self, grade_index, new_score):
        """
        Update distribution estimates for a specific grade using online update formulas.
        """
        # Increment sample count
        n = self.grade_sample_counts[grade_index]
        self.grade_sample_counts[grade_index] += 1
        n_new = n + 1
        
        # Online update for mean
        old_mean = self.grade_means[grade_index]
        self.grade_means[grade_index] = old_mean + (new_score - old_mean) / n_new
        
        # Online update for variance/std (Welford's algorithm)
        if n > 0:  # If we have at least one previous sample
            old_s = self.grade_stds[grade_index] ** 2 * n  # Previous sum of squared deviations
            new_s = old_s + (new_score - old_mean) * (new_score - self.grade_means[grade_index])
            self.grade_stds[grade_index] = np.sqrt(new_s / n_new)
        else:
            # First sample, can't compute std yet, use initial uncertainty scaled to the score
            self.grade_stds[grade_index] = self.initial_uncertainty * abs(new_score)
            
    def _update_correlations(self, school_data):
        """
        Update correlation estimates based on new multi-grade data from a school.
        Uses adaptive weights based on confidence.
        """
        # Extract valid pairs of grades and scores for this school
        pairs = []
        for i in range(self.num_grades):
            for j in range(i+1, self.num_grades):
                if school_data['grade_scores'][i] is not None and school_data['grade_scores'][j] is not None:
                    pairs.append((i, j, school_data['grade_scores'][i], school_data['grade_scores'][j]))
        
        if not pairs:
            return  # No valid pairs
            
        # Standardize the scores (z-scores) for correlation calculation
        for i, j, score_i, score_j in pairs:
            # Calculate z-scores
            if self.grade_stds[i] > 0 and self.grade_stds[j] > 0:
                z_i = (score_i - self.grade_means[i]) / self.grade_stds[i]
                z_j = (score_j - self.grade_means[j]) / self.grade_stds[j]
                
                # Weight for this update - based on our confidence in the grade distributions
                conf_i = self.distribution_confidence[i]
                conf_j = self.distribution_confidence[j]
                weight = np.sqrt(conf_i * conf_j)  # Geometric mean of confidences
                
                # Update correlation with weighted average
                old_corr = self.correlation_matrix[i, j]
                # Adaptive weight based on confidence and correlation confidence
                adapt_weight = min(0.1, 1.0 / (1.0 + len(self.schools_data) + len(self.partial_schools)))
                new_corr = (1 - adapt_weight) * old_corr + adapt_weight * (z_i * z_j)
                
                # Ensure correlation is in [-1, 1]
                new_corr = max(-1.0, min(1.0, new_corr))
                
                # Update both entries (symmetric matrix)
                self.correlation_matrix[i, j] = new_corr
                self.correlation_matrix[j, i] = new_corr
                
    def _update_confidence_levels(self):
        """
        Update confidence levels in our distribution and correlation estimates.
        """
        # Distribution confidence increases with sample size
        # Use an inverse exponential approach (diminishing returns)
        for g in range(self.num_grades):
            n = self.grade_sample_counts[g]
            # Formula: 1 - e^(-λn) gives confidence that approaches 1 as n increases
            # λ controls how quickly confidence rises with samples
            self.distribution_confidence[g] = 1.0 - np.exp(-0.3 * n)
            
        # Correlation confidence depends on:
        # 1. Number of schools with multiple grades tested
        # 2. Average distribution confidence across grades
        
        # Count schools with at least 2 grades tested
        multi_grade_count = sum(1 for s in self.partial_schools if len(s['sampled_grades']) >= 2)
        multi_grade_count += len(self.schools_data)  # Fully tested schools
        
        # Average distribution confidence
        avg_dist_conf = np.mean(self.distribution_confidence)
        
        # Use similar formula for correlation confidence
        self.correlation_confidence = (1.0 - np.exp(-0.2 * multi_grade_count)) * avg_dist_conf
        
    def evaluate_school(self, school_id, stopping_threshold=0.05, num_simulations=10000):
        """
        Evaluate a partially tested school to decide whether to continue testing.
        
        Parameters:
        -----------
        school_id : str or int
            Identifier for the school to evaluate
        stopping_threshold : float
            Probability threshold below which to stop testing
        num_simulations : int
            Number of Monte Carlo simulations to run
            
        Returns:
        --------
        dict
            Evaluation results including the decision to continue or stop,
            and confidence in that decision
        """
        # Find the school data
        school_data = None
        for s in self.partial_schools:
            if s['school_id'] == school_id:
                school_data = s
                break
                
        if school_data is None:
            raise ValueError(f"School {school_id} not found in partial schools")
            
        # Extract data from the school
        sampled_grades = school_data['sampled_grades']
        observed_scores = [school_data['grade_scores'][i] for i in sampled_grades]
        
        # Require minimum 2 grades before making predictions
        if len(sampled_grades) < 2:
            return {
                'decision': 'CONTINUE',
                'confidence': 0.0,
                'reason': 'Insufficient grades sampled for prediction (need at least 2)'
            }
            
        # Check if we have sufficient confidence in our model
        if self.correlation_confidence < 0.5:
            return {
                'decision': 'CONTINUE',
                'confidence': 0.0,
                'reason': f'Insufficient confidence in correlation model ({self.correlation_confidence:.2f})'
            }
            
        # Get unsampled grades
        unsampled_grades = [g for g in range(self.num_grades) if g not in sampled_grades]
        
        # Convert observed scores to z-scores
        observed_z = [(observed_scores[i] - self.grade_means[sampled_grades[i]]) / 
                      max(self.grade_stds[sampled_grades[i]], 1e-6) 
                      for i in range(len(sampled_grades))]
        
        # Calculate conditional distribution for unsampled grades
        # Extract relevant correlation submatrices
        obs_indices = np.array(sampled_grades)
        unobs_indices = np.array(unsampled_grades)
        
        # Need at least one unsampled grade
        if len(unobs_indices) == 0:
            # All grades sampled, calculate average
            avg_score = np.mean(observed_scores)
            return {
                'decision': 'COMPLETE',
                'confidence': 1.0,
                'average': avg_score,
                'best_average': self.best_school_average,
                'is_best': avg_score > self.best_school_average
            }
        
        # Extract correlation submatrices
        C_aa = self.correlation_matrix[np.ix_(obs_indices, obs_indices)]
        C_ab = self.correlation_matrix[np.ix_(obs_indices, unobs_indices)]
        C_bb = self.correlation_matrix[np.ix_(unobs_indices, unobs_indices)]
        
        # Ensure numerical stability
        C_aa_stable = C_aa + np.eye(len(obs_indices)) * 1e-6
        
        try:
            # Calculate conditional mean (in z-space)
            C_aa_inv = np.linalg.inv(C_aa_stable)
            cond_mean_z = C_ab.T @ C_aa_inv @ observed_z
            
            # Calculate conditional covariance (in z-space)
            cond_cov_z = C_bb - C_ab.T @ C_aa_inv @ C_ab
            
            # Ensure positive definiteness of conditional covariance
            cond_cov_z = (cond_cov_z + cond_cov_z.T) / 2  # Ensure symmetry
            eigvals = np.linalg.eigvalsh(cond_cov_z)
            if np.min(eigvals) < 1e-6:
                # If not positive definite, add small value to diagonal
                cond_cov_z += np.eye(len(unobs_indices)) * (abs(np.min(eigvals)) + 1e-6)
        except np.linalg.LinAlgError:
            # Fallback to simpler approach if matrix inversion fails
            # Use mean z-score of observed values for prediction
            mean_z = np.mean(observed_z)
            cond_mean_z = np.full(len(unobs_indices), mean_z)
            cond_cov_z = np.eye(len(unobs_indices)) * 0.5  # Default covariance
        
        # Run simulations to estimate probability of being the best school
        try:
            # Generate samples in z-space
            unobs_z_samples = np.random.multivariate_normal(
                cond_mean_z, cond_cov_z, size=num_simulations)
        except np.linalg.LinAlgError:
            # Fallback to independent sampling if multivariate fails
            unobs_z_samples = np.random.normal(
                loc=cond_mean_z.reshape(-1, 1), 
                scale=np.sqrt(np.diag(cond_cov_z)).reshape(-1, 1),
                size=(len(unobs_indices), num_simulations)).T
        
        # Convert z-samples back to score space
        unobs_score_samples = np.zeros_like(unobs_z_samples)
        for i, g in enumerate(unobs_indices):
            unobs_score_samples[:, i] = (
                self.grade_means[g] + unobs_z_samples[:, i] * self.grade_stds[g])
        
        # Combine with observed scores to get full average
        full_averages = np.zeros(num_simulations)
        for sim in range(num_simulations):
            all_scores = np.zeros(self.num_grades)
            # Add observed scores
            for i, g in enumerate(sampled_grades):
                all_scores[g] = observed_scores[i]
            # Add simulated scores
            for i, g in enumerate(unsampled_grades):
                all_scores[g] = unobs_score_samples[sim, i]
            # Calculate average
            full_averages[sim] = np.mean(all_scores)
        
        # Calculate probability of exceeding best school
        prob_exceed = np.mean(full_averages > self.best_school_average)
        
        # Calculate confidence in this estimate based on:
        # 1. Confidence in our distributions
        # 2. Confidence in our correlations
        # 3. Number of observed grades
        observed_dist_conf = np.mean([self.distribution_confidence[g] for g in sampled_grades])
        frac_observed = len(sampled_grades) / self.num_grades
        decision_confidence = observed_dist_conf * self.correlation_confidence * frac_observed
        
        # Get point estimate of school average
        mean_full_average = np.mean(full_averages)
        
        # Decision rule
        if prob_exceed < stopping_threshold:
            decision = "STOP"
            reason = f"Low probability ({prob_exceed:.1%}) of exceeding best school"
        else:
            decision = "CONTINUE"
            reason = f"School has {prob_exceed:.1%} chance of being the best"
            
        # If confidence is too low, continue testing regardless
        if decision == "STOP" and decision_confidence < self.min_confidence:
            decision = "CONTINUE"
            reason = f"Confidence in prediction too low ({decision_confidence:.1%})"
        
        return {
            'decision': decision,
            'confidence': decision_confidence,
            'probability_exceed': prob_exceed,
            'estimated_average': mean_full_average,
            'best_average': self.best_school_average,
            'reason': reason,
            'observed_grades': len(sampled_grades),
            'observed_scores': observed_scores,
            'full_averages': full_averages,  # For visualization
            'sampled_grades': sampled_grades
        }
        
    def visualize_evaluation(self, school_id, eval_result):
        """
        Create a visualization of the school evaluation results.
        
        Parameters:
        -----------
        school_id : str or int
            Identifier for the school
        eval_result : dict
            Results from evaluate_school
        """
        if 'full_averages' not in eval_result:
            print("Cannot visualize - no simulation data available")
            return None
            
        plt.figure(figsize=(12, 7))
        
        # Plot distribution of simulated final averages
        plt.hist(eval_result['full_averages'], bins=30, alpha=0.7, color='skyblue',
                label=f'Simulated outcomes (mean={np.mean(eval_result["full_averages"]):.2f})')
        
        # Add line for current best school
        plt.axvline(self.best_school_average, color='red', linestyle='--', linewidth=2,
                   label=f'Current best: {self.best_school_average:.2f}')
        
        # Add line for estimated average
        plt.axvline(eval_result['estimated_average'], color='green', linestyle='-', linewidth=2,
                   label=f'Estimated average: {eval_result["estimated_average"]:.2f}')
        
        # Probability annotation
        plt.annotate(f'P(exceed best) = {eval_result["probability_exceed"]:.1%}',
                    xy=(self.best_school_average, plt.ylim()[1]*0.9),
                    xytext=(self.best_school_average + 5, plt.ylim()[1]*0.9),
                    arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                    fontsize=12, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black"))
        
        # Confidence annotation
        plt.annotate(f'Confidence: {eval_result["confidence"]:.1%}',
                    xy=(eval_result['estimated_average'], plt.ylim()[1]*0.8),
                    xytext=(eval_result['estimated_average'] + 5, plt.ylim()[1]*0.8),
                    fontsize=12, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black"))
                    
        # Sampled grades info
        sampled_str = f"Tested grades: {sorted([g+1 for g in eval_result['sampled_grades']])}"
        plt.annotate(sampled_str,
                    xy=(plt.xlim()[0] + 2, plt.ylim()[1]*0.7),
                    fontsize=11, bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="orange"))
        
        # Decision
        decision_color = {
            'CONTINUE': 'blue',
            'STOP': 'red',
            'COMPLETE': 'green'
        }.get(eval_result['decision'], 'black')
        
        title = f"School {school_id} Evaluation - Decision: {eval_result['decision']}"
        plt.title(title, fontsize=14, color=decision_color)
        
        # Add reason
        if 'reason' in eval_result:
            plt.figtext(0.5, 0.01, eval_result['reason'], ha='center', fontsize=12, 
                      bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=decision_color))
        
        plt.xlabel('Predicted Final Average Score', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.legend(loc='upper left')
        plt.grid(True, alpha=0.3)
        
        return plt
        
    def model_diagnostics(self):
        """
        Generate a diagnostic report on the current state of the model.
        """
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Grade means and standard deviations
        plt.subplot(2, 2, 1)
        x = np.arange(1, self.num_grades + 1)
        plt.errorbar(x, self.grade_means, yerr=self.grade_stds, fmt='o', capsize=5)
        plt.xlabel('Grade')
        plt.ylabel('Score')
        plt.title('Grade Score Distributions')
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Sample counts per grade
        plt.subplot(2, 2, 2)
        plt.bar(x, self.grade_sample_counts)
        plt.xlabel('Grade')
        plt.ylabel('Number of samples')
        plt.title('Sampling Distribution Across Grades')
        plt.grid(True, alpha=0.3)
        
        # Plot 3: Correlation heatmap
        plt.subplot(2, 2, 3)
        plt.imshow(self.correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        plt.colorbar(label='Correlation')
        plt.title(f'Grade Correlation Matrix (Confidence: {self.correlation_confidence:.2f})')
        plt.xlabel('Grade')
        plt.ylabel('Grade')
        
        # Plot 4: Confidence in grade distributions
        plt.subplot(2, 2, 4)
        plt.bar(x, self.distribution_confidence)
        plt.xlabel('Grade')
        plt.ylabel('Confidence')
        plt.title('Confidence in Grade Distribution Estimates')
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)
        
        plt.tight_layout()
        return plt

    def get_testing_plan(self):
        """
        Generate an optimal adaptive testing plan based on current model state.
        """
        testing_plan = {
            'prioritized_schools': [],
            'grade_sampling_strategy': 'adaptive',
            'model_state': {
                'distribution_confidence': self.distribution_confidence.tolist(),
                'correlation_confidence': self.correlation_confidence,
                'num_complete_schools': len(self.schools_data),
                'num_partial_schools': len(self.partial_schools)
            }
        }
        
        # Prioritize schools for further testing
        for school in self.partial_schools:
            # Skip schools we've already decided to stop testing
            if hasattr(school, 'stop_testing') and school['stop_testing']:
                continue
                
            # Evaluate the school
            eval_result = self.evaluate_school(school['school_id'])
            
            # Calculate priority score based on:
            # 1. Probability of being the best
            # 2. Number of grades already tested (higher = higher priority)
            # 3. Confidence in evaluation
            
            if eval_result['decision'] == 'STOP':
                priority = 0  # Don't prioritize schools we're stopping
            else:
                prob = eval_result.get('probability_exceed', 0)
                grades_tested = len(school['sampled_grades'])
                frac_tested = grades_tested / self.num_grades
                confidence = eval_result.get('confidence', 0)
                
                # Priority formula
                priority = prob * (0.3 + 0.7 * frac_tested) * confidence
            
            testing_plan['prioritized_schools'].append({
                'school_id': school['school_id'],
                'priority': priority,
                'grades_tested': len(school['sampled_grades']),
                'decision': eval_result['decision'],
                'next_grade': self.get_next_grade_to_sample(
                    school['school_id'], school['sampled_grades']) + 1 
                    if eval_result['decision'] == 'CONTINUE' else None
            })
            
        # Sort by priority, highest first
        testing_plan['prioritized_schools'].sort(key=lambda x: x['priority'], reverse=True)
        
        return testing_plan

def simulate_testing_scenario(num_schools=100, num_grades=12, true_correlation=0.6, 
                             stopping_threshold=0.05, min_confidence=0.9):
    """
    Simulate the adaptive testing process on a population of schools.
    
    Parameters:
    -----------
    num_schools : int
        Total number of schools in the population
    num_grades : int
        Number of grades per school
    true_correlation : float
        True correlation between grades within schools
    stopping_threshold : float
        Probability threshold for stopping testing
    min_confidence : float
        Minimum confidence required for stopping decisions
        
    Returns:
    --------
    tuple
        (model, results_summary)
    """
    # Initialize model
    model = AdaptiveSchoolTestingModel(num_grades=num_grades, min_confidence=min_confidence)
    
    # Generate true scores for all schools
    np.random.seed(42)
    
    # Define true grade means and standard deviations
    true_grade_means = np.linspace(70, 90, num_grades)  # Increasing means by grade
    true_grade_stds = np.full(num_grades, 10)  # Equal std devs for simplicity
    
    # Generate true scores for all schools
    true_scores = []
    for s in range(num_schools):
        # Generate correlated scores for this school
        shared_factor = np.random.normal(0, 1)  # School-specific factor
        unique_factors = np.random.normal(0, 1, num_grades)  # Grade-specific factors
        
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
    
    #-----------------------------------------------------------------------------------------------------------------
    # NEW CODE
    # Tracking variables
    total_tests_conducted = 0
    tests_saved = 0
    current_school_index = 0
    correctly_identified_best = False
    all_tested_averages = []
    testing_decisions = []
    
    # Start testing process
    while current_school_index < num_schools:
        school = true_scores[current_school_index]
        school_id = school['school_id']
        
        # Start with no grades tested
        tested_grades = []
        continue_testing = True
        
        while continue_testing and len(tested_grades) < num_grades:
            # Determine next grade to test
            if not tested_grades:
                # First grade - pick adaptively from model
                next_grade = model.get_next_grade_to_sample(school_id)
            else:
                # Subsequent grades - use adaptive strategy
                next_grade = model.get_next_grade_to_sample(school_id, tested_grades)
            
            # Add test result
            score = school['grade_scores'][next_grade]
            model.add_grade_result(school_id, next_grade, score)
            tested_grades.append(next_grade)
            total_tests_conducted += 1
            
            # After at least 2 grades, evaluate whether to continue
            if len(tested_grades) >= 2:
                # Only evaluate if school is still in partial_schools list
                school_exists = any(s['school_id'] == school_id for s in model.partial_schools)
                if not school_exists:
                    continue_testing = False
                    continue
                    
                eval_result = model.evaluate_school(
                    school_id, stopping_threshold=stopping_threshold)
                
                # If decision is confident enough, follow it
                if eval_result['decision'] == 'STOP' and eval_result['confidence'] >= min_confidence:
                    continue_testing = False
                    tests_saved += num_grades - len(tested_grades)
                    
                    # Record decision
                    testing_decisions.append({
                        'school_id': school_id,
                        'decision': 'STOP',
                        'true_average': school['average'],
                        'estimated_average': eval_result['estimated_average'],
                        'grades_tested': len(tested_grades),
                        'max_grade_tested': max(g+1 for g in tested_grades),
                        'probability_exceed': eval_result['probability_exceed'],
                        'confidence': eval_result['confidence']
                    })
                    
    #-----------------------------------------------------------------------------------------------------------------
    # ORIGINAL CODE
    # # Tracking variables
    # total_tests_conducted = 0
    # tests_saved = 0
    # current_school_index = 0
    # correctly_identified_best = False
    # all_tested_averages = []
    # testing_decisions = []
    
    # # Start testing process
    # while current_school_index < num_schools:
    #     school = true_scores[current_school_index]
    #     school_id = school['school_id']
        
    #     # Start with no grades tested
    #     tested_grades = []
    #     continue_testing = True
        
    #     while continue_testing and len(tested_grades) < num_grades:
    #         # Determine next grade to test
    #         if not tested_grades:
    #             # First grade - pick adaptively from model
    #             next_grade = model.get_next_grade_to_sample(school_id)
    #         else:
    #             # Subsequent grades - use adaptive strategy
    #             next_grade = model.get_next_grade_to_sample(school_id, tested_grades)
            
    #         # Add test result
    #         score = school['grade_scores'][next_grade]
    #         model.add_grade_result(school_id, next_grade, score)
    #         tested_grades.append(next_grade)
    #         total_tests_conducted += 1
            
    #         # After at least 2 grades, evaluate whether to continue
    #         if len(tested_grades) >= 2:
    #             eval_result = model.evaluate_school(
    #                 school_id, stopping_threshold=stopping_threshold)
                
    #             # If decision is confident enough, follow it
    #             if eval_result['decision'] == 'STOP' and eval_result['confidence'] >= min_confidence:
    #                 continue_testing = False
    #                 tests_saved += num_grades - len(tested_grades)
                    
    #                 # Record decision
    #                 testing_decisions.append({
    #                     'school_id': school_id,
    #                     'decision': 'STOP',
    #                     'true_average': school['average'],
    #                     'estimated_average': eval_result['estimated_average'],
    #                     'grades_tested': len(tested_grades),
    #                     'max_grade_tested': max(g+1 for g in tested_grades),
    #                     'probability_exceed': eval_result['probability_exceed'],
    #                     'confidence': eval_result['confidence']
    #                 })
    #-----------------------------------------------------------------------------------------------------------------
        
        # If fully tested, record the average
        if len(tested_grades) == num_grades:
            all_tested_averages.append(school['average'])
            
            # Record decision
            testing_decisions.append({
                'school_id': school_id,
                'decision': 'COMPLETE',
                'true_average': school['average'],
                'estimated_average': school['average'],
                'grades_tested': num_grades,
                'max_grade_tested': num_grades,
                'probability_exceed': 1.0 if not all_tested_averages or school['average'] > max(all_tested_averages) else 0.0,
                'confidence': 1.0
            })
        
        # Move to next school
        current_school_index += 1
        
        # Check if we found the true best school
        if current_school_index == num_schools:
            best_tested_index = np.argmax(all_tested_averages)
            best_tested_school_id = testing_decisions[best_tested_index]['school_id']
            correctly_identified_best = (best_tested_school_id == true_scores[0]['school_id'])
    
    # Compile results
    results = {
        'total_schools': num_schools,
        'total_possible_tests': num_schools * num_grades,
        'tests_conducted': total_tests_conducted,
        'tests_saved': tests_saved,
        'efficiency': 1 - (total_tests_conducted / (num_schools * num_grades)),
        'found_true_best': correctly_identified_best,
        'schools_fully_tested': sum(1 for d in testing_decisions if d['decision'] == 'COMPLETE'),
        'schools_partially_tested': sum(1 for d in testing_decisions if d['decision'] == 'STOP'),
        'testing_decisions': testing_decisions
    }
    
    return model, results

def demonstrate_adaptive_testing():
    """
    Run a demonstration of the adaptive testing approach and visualize the results.
    """
    print("Starting adaptive testing simulation...")
    
    # Run simulation
    model, results = simulate_testing_scenario(
        num_schools=100,  # Smaller number for demo
        num_grades=12,
        true_correlation=0.7,
        stopping_threshold=0.05,
        min_confidence=0.85
    )
    
    # Print summary results
    print(f"\nSimulation Results:")
    print(f"Total schools: {results['total_schools']}")
    print(f"Tests conducted: {results['tests_conducted']} out of {results['total_possible_tests']} possible")
    print(f"Tests saved: {results['tests_saved']} ({results['efficiency']*100:.1f}% efficiency)")
    print(f"Found true best school: {results['found_true_best']}")
    print(f"Schools fully tested: {results['schools_fully_tested']}")
    print(f"Schools partially tested: {results['schools_partially_tested']}")
    
    # Create visualization of the testing process
    plt.figure(figsize=(14, 8))
    
    # Plot 1: Testing pattern
    decisions = results['testing_decisions']
    
    # Extract data for visualization
    school_ids = [int(d['school_id'].split('_')[1]) for d in decisions]
    max_grades_tested = [d['max_grade_tested'] for d in decisions]
    true_averages = [d['true_average'] for d in decisions]
    decisions_type = [d['decision'] for d in decisions]
    
    # Sort by true average
    sort_idx = np.argsort(true_averages)[::-1]  # Descending
    school_ids_sorted = [school_ids[i] for i in sort_idx]
    max_grades_sorted = [max_grades_tested[i] for i in sort_idx]
    decisions_sorted = [decisions_type[i] for i in sort_idx]
    
    # Create color map
    colors = ['green' if d == 'COMPLETE' else 'red' for d in decisions_sorted]
    
    plt.subplot(2, 1, 1)
    plt.bar(range(len(school_ids_sorted)), max_grades_sorted, color=colors)
    plt.xlabel('Schools (sorted by true average)')
    plt.ylabel('Maximum Grade Tested')
    plt.title('Testing Pattern by School Quality')
    plt.axhline(y=12, color='black', linestyle='--', alpha=0.5)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', label='Fully Tested'),
        Patch(facecolor='red', label='Early Stopping')
    ]
    plt.legend(handles=legend_elements)
    
    # Plot 2: Model diagnostics
    plt.subplot(2, 1, 2)
    x = np.arange(1, model.num_grades + 1)
    plt.bar(x, model.grade_sample_counts)
    plt.xlabel('Grade')
    plt.ylabel('Number of Tests')
    plt.title('Distribution of Tests Across Grades')
    
    plt.tight_layout()
    plt.show()
    
    # Show model diagnostics
    model.model_diagnostics()
    plt.show()
    
    return model, results

if __name__ == "__main__":
    demonstrate_adaptive_testing()