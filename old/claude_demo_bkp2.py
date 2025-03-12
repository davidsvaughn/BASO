import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from scipy import stats
# from scipy.stats import multivariate_normal

class AdaptiveSchoolTestingModel:
    """
    Base model for adaptive testing of schools across multiple grades.
    Uses Bayesian inference to optimize testing strategy and early stopping.
    """
    def __init__(self, num_grades=12, initial_uncertainty=0.5, min_confidence=0.95, initial_correlation=0.5):
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
        self.eval_memory = {}  # Cache for evaluation results
        
        # Tracking grade-specific information
        self.grade_means = np.zeros(num_grades)
        self.grade_stds = np.ones(num_grades) * initial_uncertainty
        self.grade_sample_counts = np.zeros(num_grades, dtype=int)
        
        # dsv
        self.cov = np.eye(num_grades).astype(float) + initial_uncertainty  * initial_correlation #* initial_uncertainty
        # set the diagonal to all 1s
        np.fill_diagonal(self.cov, 1.0)
        self.cross_terms = np.zeros((num_grades, num_grades), dtype=float)
        self.grade_pair_sample_counts = np.zeros((num_grades, num_grades), dtype=int)
        
        
        # Correlation matrix starts with moderate assumed correlations
        self.correlation_matrix = np.eye(num_grades)
        for i in range(num_grades):
            for j in range(num_grades):
                if i != j:
                    # Assume moderate correlation initially
                    self.correlation_matrix[i, j] = initial_correlation
        
        # Confidence-related tracking
        self.distribution_confidence = np.zeros(num_grades)  # Confidence in our estimates per grade
        self.correlation_confidence = 0.0  # Overall confidence in our correlation estimates

        
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
            random_component = 0.1 * np.random.random()
            
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
        
        # Clear evaluation cache for this school
        self.eval_memory[school_id] = None
        
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
                'sampled_grades': [],
                'sampled_pairs': [],
            }
            self.partial_schools.append(school_data)
        
        # Add the new score
        school_data['grade_scores'][grade_index] = score
        if grade_index not in school_data['sampled_grades']:
            school_data['sampled_grades'].append(grade_index)
        
        # # Update our distribution estimates
        # self._update_grade_distribution(grade_index, score)
        
        # # If we have at least 2 grades from this school, update correlation
        # if len(school_data['sampled_grades']) >= 2:
        #     self._update_correlations(school_data)
        
        # Update our distribution estimates
        self._update_distributions(grade_index, score, school_data)
            
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
        #TODO: loop over sampled_grades instead of all grades
        #TODO: add 'new_grade' to school_data to track which grade was just added, then only loop over (that grade, sampled_grades) pairs! much faster!
        # Extract valid pairs of grades and scores for this school
        pairs = []
        for i in range(self.num_grades):
            for j in range(i+1, self.num_grades):
                if school_data['grade_scores'][i] is not None and school_data['grade_scores'][j] is not None:
                    if (i, j) not in school_data['sampled_pairs']:
                        school_data['sampled_pairs'].append((i, j))
                        pairs.append((i, j, school_data['grade_scores'][i], school_data['grade_scores'][j]))
                        self.grade_pair_sample_counts[(i, j)] += 1
        
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
                conf_factor = np.sqrt(conf_i * conf_j)  # Geometric mean of confidences
                
                # Adaptive weight based on confidence and correlation confidence
                # adapt_weight = min(0.1, 1.0 / (1.0 + len(self.schools_data) + len(self.partial_schools)))
                
                # Something more like this:
                base_step = min(0.1, 1.0 / (1.0 + len(self.schools_data)))# + len(self.partial_schools)))
                adapt_weight = base_step * conf_factor # dsv
                
                # print(f'{adapt_weight:0.4f}') #dsv
                
                # Update correlation with weighted average
                old_corr = self.correlation_matrix[i, j]
                new_corr = (1 - adapt_weight) * old_corr + adapt_weight * (z_i * z_j)
                
                # Ensure correlation is in [-1, 1]
                new_corr = max(-1.0, min(1.0, new_corr))
                
                # Update both entries (symmetric matrix)
                self.correlation_matrix[i, j] = new_corr
                self.correlation_matrix[j, i] = new_corr
        # print(self.correlation_matrix) #dsv
    
    def _update_distributions(self, grade_index, new_score, school_data):
        """
        Update distribution estimates for a specific grade using online update formulas.
        """
        # Increment sample count
        n = self.grade_sample_counts[grade_index]
        self.grade_sample_counts[grade_index] += 1
        n_new = n + 1
        
        # Online update for mean
        old_mean = self.grade_means[grade_index]
        self.grade_means[grade_index] = new_mean = old_mean + (new_score - old_mean) / n_new
        
        # Online update for variance/std (Welford's algorithm)
        if n > 0:  # If we have at least one previous sample
            old_s = self.grade_stds[grade_index] ** 2 * n  # Previous sum of squared deviations
            new_s = old_s + (new_score - old_mean) * (new_score - self.grade_means[grade_index])
            self.grade_stds[grade_index] = np.sqrt(new_s / n_new)
        else:
            # First sample, can't compute std yet, use initial uncertainty scaled to the score
            self.grade_stds[grade_index] = self.initial_uncertainty * abs(new_score)
        
        
        #-----------------------------------------------------------------------
        update = False
        for j in school_data['sampled_grades']:
            if j == grade_index:
                continue
            update = True
            
            # update pair sample counts
            self.grade_pair_sample_counts[grade_index, j] += 1
            self.grade_pair_sample_counts[j, grade_index] += 1
            
            
            # update cross terms
            # self.cov[grade_index, j] = (1 - 1/n_new) * self.cov[grade_index, j] + (new_score - new_mean) * (school_data['grade_scores'][j] - self.grade_means[j]) / n_new
            self.cross_terms[grade_index, j] += (new_score - new_mean) * (school_data['grade_scores'][j] - self.grade_means[j])
            self.cross_terms[j, grade_index] = self.cross_terms[grade_index, j]
            
            
            # # update covariance matrix from cross terms
            # self.cov[grade_index, j] = self.cross_terms[grade_index, j] / self.grade_pair_sample_counts[grade_index, j]
            # # self.cov[grade_index, j] = self.cross_terms[grade_index, j] / n_new # ????
            # self.cov[j, grade_index] = self.cov[grade_index, j]
            
            #-------------------------------
            # alternative covariance update
            self.cov[grade_index, j] = (1 - 1/n_new) * self.cov[grade_index, j] + (new_score - new_mean) * (school_data['grade_scores'][j] - self.grade_means[j]) / n_new
            self.cov[j, grade_index] = self.cov[grade_index, j]
            #--------------------------------
            
            
            # update correlation matrix from covariance matrix
            self.correlation_matrix[grade_index, j] = self.cov[grade_index, j] / (self.grade_stds[grade_index] * self.grade_stds[j])
            self.correlation_matrix[j, grade_index] = self.correlation_matrix[grade_index, j]
    
    
    def print_error(self):
        
        # OLD
        # # compute norm diff between true_means and grade_means
        # err = np.linalg.norm(self.true_means - self.grade_means)
        # print(f'\nmeans_err:\t{err:0.4f}')
        # # compute norm diff between true_stds and grade_stds
        # err = np.linalg.norm(self.true_stds - self.grade_stds)
        # print(f'stds_err:\t{err:0.4f}')
        # # compute norm diff between true_correlation_matrix and correlation_matrix
        # err = np.linalg.norm(self.true_correlation_matrix - self.correlation_matrix)
        # print(f'corr_err:\t{err:0.4f}')
        # print()
        
        # NEW
        rmse = np.sqrt(np.mean((self.true_means - self.grade_means)**2))
        print(f'\nmeans_err:\t{rmse:0.4f}')
        rel_error = np.mean(np.abs(self.true_stds - self.grade_stds) / self.true_stds)
        print(f'stds_err:\t{rel_error:0.4f}')
        fro_norm = np.linalg.norm(self.true_correlation_matrix - self.correlation_matrix, 'fro')
        print(f'corr_err:\t{fro_norm:0.4f}')

        
    # def _update_correlations_orig(self, school_data):
    #     """
    #     Update correlation estimates based on new multi-grade data from a school.
    #     Uses adaptive weights based on confidence.
    #     """
    #     # Extract valid pairs of grades and scores for this school
    #     pairs = []
    #     for i in range(self.num_grades):
    #         for j in range(i+1, self.num_grades):
    #             if school_data['grade_scores'][i] is not None and school_data['grade_scores'][j] is not None:
    #                 pairs.append((i, j, school_data['grade_scores'][i], school_data['grade_scores'][j]))
        
    #     if not pairs:
    #         return  # No valid pairs
            
    #     # Standardize the scores (z-scores) for correlation calculation
    #     for i, j, score_i, score_j in pairs:
    #         # Calculate z-scores
    #         if self.grade_stds[i] > 0 and self.grade_stds[j] > 0:
    #             z_i = (score_i - self.grade_means[i]) / self.grade_stds[i]
    #             z_j = (score_j - self.grade_means[j]) / self.grade_stds[j]
                
    #             # Weight for this update - based on our confidence in the grade distributions
    #             conf_i = self.distribution_confidence[i]
    #             conf_j = self.distribution_confidence[j]
    #             conf_factor = np.sqrt(conf_i * conf_j)  # Geometric mean of confidences
                
    #             # Adaptive weight based on confidence and correlation confidence
    #             # adapt_weight = min(0.1, 1.0 / (1.0 + len(self.schools_data) + len(self.partial_schools)))
                
    #             # Something more like this:
    #             base_step = min(0.1, 1.0 / (1.0 + len(self.schools_data)))# + len(self.partial_schools)))
    #             adapt_weight = base_step * conf_factor * 2.0 # dsv
                
    #             # print(f'{adapt_weight:0.4f}') #dsv
                
    #             # Update correlation with weighted average
    #             old_corr = self.correlation_matrix[i, j]
    #             new_corr = (1 - adapt_weight) * old_corr + adapt_weight * (z_i * z_j)
                
    #             # Ensure correlation is in [-1, 1]
    #             new_corr = max(-1.0, min(1.0, new_corr))
                
    #             # Update both entries (symmetric matrix)
    #             self.correlation_matrix[i, j] = new_corr
    #             self.correlation_matrix[j, i] = new_corr
    #     # print(self.correlation_matrix) #dsv
                 
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
        
        if school_id in self.eval_memory and self.eval_memory[school_id] is not None:
            return self.eval_memory[school_id]
        
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
            result = {
                'decision': 'COMPLETE',
                'confidence': 1.0,
                'average': avg_score,
                'best_average': self.best_school_average,
                'is_best': avg_score > self.best_school_average
            }
            self.eval_memory[school_id] = result
            return result
        
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
        
        result = {
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
        self.eval_memory[school_id] = result
        return result
        
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


class PrioritizedSchoolTestingModel(AdaptiveSchoolTestingModel):
    """
    Enhanced model that focuses on efficiently identifying the top schools through
    dynamic prioritization and pruning with the ability to return to schools later.
    """
    
    def __init__(self, num_grades=12, 
                 initial_uncertainty=0.5,
                 initial_correlation=0.5,
                 min_confidence=0.95,
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
        super().__init__(num_grades, initial_uncertainty, min_confidence, initial_correlation)
        self.revisit_threshold = revisit_threshold
        
        # Additional tracking for prioritization
        self.school_rankings = []  # Current rankings of schools
        self.pruned_schools = []   # Schools that were pruned but could be revisited
        self.testing_history = []  # History of testing actions and rankings
        
    def add_grade_result(self, school_id, grade_index, score, update_rankings=True):
        """
        Add a single grade score result, update model estimates, and recalculate rankings.
        """
        # Call parent method to update distributions and correlations
        school_data = super().add_grade_result(school_id, grade_index, score)
        
        # Update school rankings after adding new data
        if update_rankings:
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
                
                # Check if the key exists, otherwise use a fallback
                estimated_avg = eval_result.get('estimated_average', None)
                
                # If not found, check other possible keys or calculate from raw data
                if estimated_avg is None:
                    if 'mean_full_average' in eval_result:
                        estimated_avg = eval_result['mean_full_average']
                    else:
                        # Fallback to average of available grades
                        estimated_avg = np.mean([
                            school['grade_scores'][g] for g in school['sampled_grades']
                            if school['grade_scores'][g] is not None
                        ])
                
                rankings.append({
                    'school_id': school['school_id'],
                    'status': 'PARTIAL',
                    'average': None,  # Unknown true average
                    'estimated_average': estimated_avg,
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
            # School not found - check if it's already pruned or completed
            for school in self.pruned_schools:
                if school['school_id'] == school_id:
                    return school  # Already pruned, return existing data
                    
            for school in self.schools_data:
                if school['school_id'] == school_id:
                    # Already completed, return data with appropriate status
                    return {
                        'school_id': school_id,
                        'estimated_average': school['average'],
                        'confidence': 1.0,
                        'grades_tested': self.num_grades,
                        'sampled_grades': list(range(self.num_grades)),
                        'grade_scores': school['grade_scores'].copy(),
                        'pruned_at': len(self.testing_history),
                        'probability_best': next((r['probability_best'] for r in self.school_rankings 
                                            if r['school_id'] == school_id), 0.0)
                    }
                
            # If we get here, school truly doesn't exist
            raise ValueError(f"School {school_id} not found in any collection (partial, pruned, or complete)")
            
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
        from matplotlib.patches import Patch
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


# Helper function to get top K schools from the model
def get_top_k_schools(model, k=10):
    """
    Get the current top K ranked schools with their probabilities and confidence levels.
    
    Parameters:
    -----------
    model : PrioritizedSchoolTestingModel
        The model containing the school rankings
    k : int
        Number of top schools to return
        
    Returns:
    --------
    list
        List of dictionaries with information about the top K schools
    """
    # Ensure rankings are up to date if using PrioritizedSchoolTestingModel
    if hasattr(model, '_update_school_rankings'):
        model._update_school_rankings()
        
        # Get top K schools from rankings
        top_schools = model.school_rankings[:min(k, len(model.school_rankings))]
        
        # Format results for reporting
        result = []
        for school in top_schools:
            result.append({
                'rank': school['rank'],
                'school_id': school['school_id'],
                'status': school['status'],
                'estimated_average': school['estimated_average'],
                'probability_best': school['probability_best'],
                'confidence': school['confidence'],
                'grades_tested': school['grades_tested'],
                'recommendation': _get_testing_recommendation(school)
            })
    else:
        # For AdaptiveSchoolTestingModel, create rankings on the fly
        rankings = []
        
        # Process fully tested schools
        for i, school in enumerate(model.schools_data):
            rankings.append({
                'school_id': school['school_id'],
                'status': 'COMPLETE',
                'average': school['average'],
                'estimated_average': school['average'],
                'confidence': 1.0,
                'grades_tested': model.num_grades,
                'probability_best': 0.0  # Will be calculated below
            })
        
        # Process partially tested schools
        for school in model.partial_schools:
            if len(school['sampled_grades']) >= 2:
                # Run evaluation to get estimate
                eval_result = model.evaluate_school(school['school_id'])
                
                rankings.append({
                    'school_id': school['school_id'],
                    'status': 'PARTIAL',
                    'average': None,
                    'estimated_average': eval_result['estimated_average'],
                    'confidence': eval_result['confidence'],
                    'grades_tested': len(school['sampled_grades']),
                    'probability_best': eval_result.get('probability_exceed', 0.0)
                })
        
        # Sort by estimated average
        rankings.sort(key=lambda x: x['estimated_average'], reverse=True)
        
        # Get top K schools
        top_schools = rankings[:min(k, len(rankings))]
        
        # Format results for reporting
        result = []
        for i, school in enumerate(top_schools):
            result.append({
                'rank': i + 1,
                'school_id': school['school_id'],
                'status': school['status'],
                'estimated_average': school['estimated_average'],
                'probability_best': school['probability_best'],
                'confidence': school['confidence'],
                'grades_tested': school['grades_tested'],
                'recommendation': _get_testing_recommendation(school)
            })
    
    return result

def _get_testing_recommendation(school):
    """Generate a testing recommendation based on school status and probabilities."""
    if school['status'] == 'COMPLETE':
        return "Fully tested"
    
    if school['probability_best'] > 0.3:
        return "High priority - continue testing"
    elif school['probability_best'] > 0.1:
        return "Medium priority"
    elif school['probability_best'] > 0.03:
        return "Low priority"
    else:
        return "Consider pruning"



def generate_synthetic_data(num_schools=100, num_grades=12, mean_correlation=0.6):

    # Define true grade means and standard deviations
    true_grade_means = np.linspace(70, 90, num_grades)  # Increasing means by grade
    true_grade_stds = np.full(num_grades, 5)  # Equal std devs for simplicity
    
    # Generate true scores for all schools
    true_scores = []
    for s in range(num_schools):
        # Generate correlated scores for this school
        shared_factor = np.random.normal(0, 1)  # School-specific factor
        unique_factors = np.random.normal(0, 1, num_grades)  # Grade-specific factors
        
        # Mix factors based on desired correlation
        z_scores = shared_factor * np.sqrt(mean_correlation) + \
                  unique_factors * np.sqrt(1 - mean_correlation)
        
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
    # true_scores.sort(key=lambda x: x['average'], reverse=True)
    
    return true_scores

def estimate_data_stats(X):
    mu_hat = X.mean(axis=0)
    sigma_hat = np.cov(X, rowvar=False)
    return mu_hat, sigma_hat

def package_school_data(X):
    true_scores = []
    for i, row in enumerate(X):
        true_scores.append({
            'school_id': f"School_{i+1}",
            'grade_scores': row,
            'average': np.mean(row)
        })
    return true_scores

def simulate_school_data(X, n=None):
    if n is None:
        n,m = X.shape
    mu_hat, sigma_hat = estimate_data_stats(X)
    X_new = np.random.multivariate_normal(mu_hat, sigma_hat, size=n)
    
    return package_school_data(X_new), X_new
    

def import_tab_data(filename):
    """
    Import tabular data from a tab-delimited file into a 2D numpy array.
    
    Args:
        filename (str): Path to the tab-delimited file
        
    Returns:
        np.ndarray: 2D array containing the imported data
    """
    try:
        # Load data using numpy's loadtxt function with tab delimiter
        data = np.loadtxt(filename, delimiter='\t')
        return data
    except Exception as e:
        print(f"Error importing data: {e}")
        return None
    
def load_school_data(file_path='llama-siam-1.txt'):
    X = import_tab_data(file_path)
    
    X *= 100  # Convert QWK to percentage scale
    
    true_scores = []
    for i, row in enumerate(X):
        true_scores.append({
            'school_id': f"School_{i+1}",
            'grade_scores': row,
            'average': np.mean(row)
        })
    return true_scores, X


# Example simulation function
def demonstrate_top_k_rankings():
    """
    Demonstrate how to use the prioritized testing model with top-K rankings.
    """
    
    K = 1      # top K
    N = 0.15    # initial grades to test for each school # 0.15
    check = 25  # check every N tests
    revisit_threshold   = 0.1 # .15
    
    # initial_uncertainty, initial_correlation = 0.01, 0.5 # ***** BEST *****
    
    initial_uncertainty, initial_correlation = 0.01, 0.5 
    # initial_uncertainty, initial_correlation = 0.1, 0.1
    
    
    # random seed
    rand_seed = np.random.randint(100, 1000)
    # rand_seed = 407
    print(f"\nRANDOM SEED ---> {rand_seed}\n")
    np.random.seed(rand_seed)
    
    # # Generate true scores for schools
    # num_schools, num_grades = 50, 30
    # true_scores = generate_synthetic_data(num_schools=num_schools, num_grades=num_grades, true_correlation=0.25)
    
    # or...
    # Load school data
    # fn = 'data/llama-siam-1.txt'
    fn = 'data/phi4-siam.txt'
    
    true_scores, X = load_school_data(fn)
    
    # generate synthetic data similar to the loaded data
    # true_scores, X = simulate_school_data(X, n=100)#, n=100)
    
    # reset dimensions
    num_schools = len(true_scores)
    num_grades = len(true_scores[0]['grade_scores'])
    
    print(f"Loaded data for {num_schools} schools with {num_grades} grades each")
    if N<1: N = int(N*num_grades)
    if K<=1: K = int(K*num_schools)
    print(f"Prioritizing TOP-{K} schools")
    print(f"Testing {N} grades initially for each school\n")
    
    # Sort by true average (descending)
    true_scores.sort(key=lambda x: x['average'], reverse=True)
    true_rankings = {s['school_id']: i+1 for i, s in enumerate(true_scores)}
    
    # store the true best school and top 5
    true_best = true_scores[0]['school_id']
    true_top_5 = [s['school_id'] for s in true_scores[:5]]
    
    # randomly shuffle the schools
    np.random.shuffle(true_scores)
    
    # Create model
    model = PrioritizedSchoolTestingModel(
        num_grades=num_grades,
        initial_uncertainty=initial_uncertainty,
        initial_correlation=initial_correlation,
        min_confidence=0.85,
        revisit_threshold=revisit_threshold
    )
    
    # compute true correlation
    true_correlation_matrix = np.corrcoef(X, rowvar=False)
    true_means = np.mean(X, axis=0)
    true_stds = np.std(X, axis=0)
    model.true_correlation_matrix = true_correlation_matrix
    model.true_means = true_means
    model.true_stds = true_stds
    
    # Start testing
    max_tests = tot_tests = num_schools * model.num_grades
    tests_conducted = 0
    top_k_history = {}
    
    #-----------------------------------------------------------------------------
    # Function to show top K rankings nicely
    def print_top_k(top_k, true_rankings, k=10):
        top_k = top_k[:k]
        print("\nCurrent Top School Rankings:")
        print("{:<5} {:<12} {:<10} {:<12} {:<10} {:<12}".format(
            "Rank", "School ID", "Status", "Est. Avg", "Prob Best", "True Rank"))
        print("-" * 70)
        
        for school in top_k:
            true_rank = true_rankings[school['school_id']]
            school_num = school['school_id'].split('_')[1]
            
            print("{:<5} {:<12} {:<10} {:<12.1f} {:<10.3f} {:<12}".format(
                school['rank'], 
                f"School {school_num}", 
                # school['status'][:4], 
                f"{school['grades_tested']/model.num_grades:.2f}", 
                school['estimated_average'], 
                school['probability_best'], 
                true_rank))
            
        # check if true_best is ranked 1
        top_school = top_k[0]
        top_1_correct = top_school['school_id'] == true_best
        if top_1_correct:
            r_tested = top_school['grades_tested']/model.num_grades
            prob_best = top_school['probability_best']
            if r_tested == 1.0 and prob_best > 0.9:
                print("\nTRUE BEST SCHOOL FOUND!")
                return True
        return False
        
    #-----------------------------------------------------------------------------
        
    
    # Test initial schools randomly to build model
    print("Starting initial testing to build model...")
    
    # select N random grades to test    
    for i in range(num_schools):
        school = true_scores[i]
        # select N random grades to test
        grade_indices = np.random.choice(model.num_grades, N, replace=False)
        
        for grade_index in grade_indices:
            score = school['grade_scores'][grade_index]
            model.add_grade_result(school['school_id'], grade_index, score, update_rankings=False)
            tests_conducted += 1
    
    # Show initial rankings
    top_k = get_top_k_schools(model, k=K)
    print_top_k(top_k, true_rankings)
    top_k_history[tests_conducted] = top_k
    
    # Main testing loop - prioritize based on rankings
    print("\nBeginning prioritized testing...")
    
    while tests_conducted < max_tests:

        # Get top ranked schools
        top_k = get_top_k_schools(model, k=K)
        
        # Find highest priority partially tested school
        next_school = None
        for school in top_k:
            if school['status'] == 'PARTIAL':
                next_school = school
                break
        
        if next_school is None:
            # Add a new school if no partial schools in top K
            new_index = len(model.schools_data) + len(model.partial_schools) + len(model.pruned_schools)
            if new_index < len(true_scores):
                next_school_id = true_scores[new_index]['school_id']
                # Test a random grade
                grade_index = np.random.randint(0, model.num_grades)
                score = true_scores[new_index]['grade_scores'][grade_index]
                model.add_grade_result(next_school_id, grade_index, score, update_rankings=False)
                tests_conducted += 1
            else:
                # No more schools to add
                break
        else:
            # Test next grade for the selected school
            school_id = next_school['school_id']
            
            # Find the school in partial_schools
            school_data = None
            for s in model.partial_schools:
                if s['school_id'] == school_id:
                    school_data = s
                    break
            
            if school_data:
                # Get next grade to test
                next_grade = model.get_next_grade_to_sample(school_id, school_data['sampled_grades'])
                
                # Find true score
                true_school = next((s for s in true_scores if s['school_id'] == school_id), None)
                if true_school:
                    score = true_school['grade_scores'][next_grade]
                    model.add_grade_result(school_id, next_grade, score, update_rankings=False)
                    tests_conducted += 1
                    
                    # Consider pruning low probability schools after some testing
                    if (tests_conducted > 50 and 
                        next_school['probability_best'] < 0.02 and 
                        next_school['confidence'] > 0.8 and
                        next_school['grades_tested']/model.num_grades > 0.25):
                        
                        # Check if the school is still in partial_schools before pruning
                        school_in_partial = any(s['school_id'] == school_id for s in model.partial_schools)
                        if school_in_partial:
                            model.prune_school(school_id)
                        else:
                            print(f"Skipping pruning for {school_id} as it's no longer in partial_schools")
        
        # Save rankings at checkpoints
        if tests_conducted % check == 0:
            top_k = get_top_k_schools(model, k=K)
            top_k_history[tests_conducted] = top_k
            print(f"\nAfter {tests_conducted}/{tot_tests} tests ({tests_conducted/tot_tests:0.2f}):")
            stop = print_top_k(top_k, true_rankings)
            
            if stop:
                break
            
            # Calculate accuracy metrics
            top_1_correct = top_k[0]['school_id'] == true_best
            top_5_overlap = len([s for s in top_k[:5] if true_rankings[s['school_id']] <= 5])
            print(f"\nTop-1 Accuracy: {'CORRECT' if top_1_correct else 'INCORRECT'}")
            print(f"Top-5 Overlap: {top_5_overlap}/5 schools")
            model.print_error()
            print("-" * 100)
    
    # Final evaluation
    print("\n----- FINAL RESULTS -----")
    top_k = get_top_k_schools(model, k=K)
    print_top_k(top_k, true_rankings)
    
    # Evaluate final accuracy
    # true_best = true_scores[0]['school_id']
    model_best = top_k[0]['school_id']
    print(f"\nFound true best school: {'YES' if model_best == true_best else 'NO'}")
    
    # Calculate top-5 overlap
    # true_top_5 = [s['school_id'] for s in true_scores[:5]]
    model_top_5 = [s['school_id'] for s in top_k[:5]]
    overlap = len(set(true_top_5) & set(model_top_5))
    print(f"Top-5 overlap: {overlap}/5 schools")
    
    # Calculate mean reciprocal rank
    mrr = 0
    for i, school in enumerate(top_k[:5]):
        true_rank = true_rankings[school['school_id']]
        if true_rank <= 5:
            mrr += 1 / (i + 1)
    mrr /= 5
    print(f"Mean Reciprocal Rank (MRR): {mrr:.3f}")
    
    # compare to the true correlation matrix from X
    cdiff = np.abs(model.correlation_matrix - true_correlation_matrix)
    corr_values = cdiff[np.triu_indices(X.shape[1], k=1)]
    plt.hist(corr_values, bins=20, color='skyblue', edgecolor='black')
    plt.title(f'Correlation diffs')
    plt.xlabel('cc abs(diff)')
    plt.ylabel('freq')
    plt.show()
    
    sys.exit()
    
    # Visualize final rankings
    model.visualize_rankings(show_top_n=10)
    plt.savefig("final_rankings.png")
    plt.show()
    
    # Visualize testing progress
    model.visualize_testing_progress()
    plt.savefig("testing_progress.png")
    plt.show()
    
    # Return the model and true scores for further analysis
    # print("\nEstimated Model parameters:")
    # print(model.correlation_matrix.round(4))
    # print(model.grade_means.round(4))
    # print(model.grade_stds.round(4))
    
    return model, true_scores, top_k_history

if __name__ == "__main__":
    demonstrate_top_k_rankings()