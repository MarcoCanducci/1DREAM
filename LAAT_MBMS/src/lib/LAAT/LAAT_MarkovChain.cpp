#include "LAAT.h"
#include "Eigen/Sparse"

/**
 * Markov Chain approximation of the LAAT algorithm.
 * 
 * This function approximates LAAT by constructing a sparse transition matrix
 * based on the neighborhood structure and computing the stationary distribution
 * as the dominant eigenvector using power iteration.
 *
 * The transition probability from point i to neighbor j is computed as:
 *   P(i -> j) = exp(beta * (lambda2 * pref_ij + lambda3 * w_j)) / Z_i
 * 
 * where:
 *   - lambda2 = kappa * (1 - gamma)   [preference/alignment weight]
 *   - lambda3 = gamma                  [external weight]
 *   - pref_ij = normalized alignment preference from preprocessing
 *   - w_j = normalized external weight for neighbor j
 *   - Z_i = normalization constant (sum over all neighbors)
 *   - beta = beta_antmovement (inverse temperature)
 *
 * Note: This version does NOT include pheromone feedback (lambda1 term).
 * The stationary distribution approximates the expected pheromone distribution.
 *
 * Reference:
 * Taghribi, A., Bunte, K., Smith, R., Shin, J., Mastropietro, M., Peletier,
 * R. F., & Tino, P. (2020). LAAT: Locally Aligned Ant Technique for detecting
 * manifolds of varying density. arXiv preprint arXiv:2009.08326.
 *
 * @param data vector containing all the data points
 * @param th_neighb minimum number of nearest neighbours needed
 * @param neighbdradii Neighborhood radius for nearest neighbour search
 * @param kappa tuning parameter for preference importance
 * @param gamma tuning parameter for external weights influence
 * @param external_weights per-point external scalar weights
 * @param tolerance convergence tolerance for power iteration
 * @param max_iterations maximum iterations for power iteration
 * @param numberofthreads number of OpenMP threads
 * @return vector containing the stationary distribution (pheromone approximation)
 */
std::vector<float> LocallyAlignedAntTechnique_MarkovChain(
    std::vector<std::vector<float>> const &data,
    size_t th_neighb,
    float neighbdradii,
    float kappa,
    float gamma,
    std::vector<float> const &external_weights,
    float tolerance,
    size_t max_iterations,
    size_t numberofthreads)
{
    cout << endl << endl << "Running LAAT Markov Chain Approximation, version 1.0.0 (09-12-2025)" << endl << endl;

    printf("Parameters: th_neighb = %zu, radius = %.4f, kappa = %.4f, gamma = %.4f\n",
           th_neighb, neighbdradii, kappa, gamma);
    printf("Power iteration: tolerance = %.2e, max_iterations = %zu\n", tolerance, max_iterations);

    // TIME VARIABLES
    struct timespec GL_start, GL_finish;
    std::vector<float> GL_times(10, 0.0f);

    omp_set_num_threads(numberofthreads);
    Eigen::initParallel();

    // ============================================================================
    // STEP 1: PREPROCESSING - Reuse existing preprocessing infrastructure
    // ============================================================================
    cout << "Preprocessing...\n";
    clock_gettime(CLOCK_REALTIME, &GL_start);

    std::vector<std::vector<size_t>> neighbourhoods;
    std::vector<std::vector<float>> neighbourhoods_distances;
    std::vector<std::vector<std::vector<float>>> eigenVectors;
    std::vector<std::vector<float>> eigenValues;
    std::vector<std::vector<float>> preferences;
    std::vector<std::vector<float>> quality_pheromone;
    std::vector<pair<float, size_t>> probability_std(data.size());
    std::vector<size_t> interesting_particle;

    // Variables for dynamic radius (not used here, but needed for preprocess signature)
    std::vector<std::vector<size_t>> pso_neigbourhoods_number;
    std::vector<std::vector<std::vector<std::vector<float>>>> pso_eigenVectors;
    std::vector<std::vector<std::vector<float>>> pso_eigenValues;
    std::vector<std::vector<float>> pso_radii_accumulated_probabilities;
    std::vector<std::vector<float>> pso_radii_probabilities;

    // Call preprocess with dynamic_radius_actived = 0 (static radius mode)
    preprocess(data, th_neighb, neighbdradii, neighbdradii, 1, pso_neigbourhoods_number,
               neighbourhoods, neighbourhoods_distances, eigenVectors, eigenValues,
               pso_eigenVectors, pso_eigenValues, probability_std, interesting_particle,
               preferences, pso_radii_accumulated_probabilities, pso_radii_probabilities,
               0, quality_pheromone);

    clock_gettime(CLOCK_REALTIME, &GL_finish);
    GL_times[0] = (GL_finish.tv_sec - GL_start.tv_sec) + 
                  (GL_finish.tv_nsec - GL_start.tv_nsec) / 1000000000.0;

    // Count interesting particles
    size_t num_interesting = 0;
    for (size_t i = 0; i < data.size(); i++) {
        if (interesting_particle[i] == 1) {
            num_interesting++;
        }
    }
    printf("\nNumber of 'interesting' particles: %zu out of %zu (%.2f%%)\n",
           num_interesting, data.size(), 100.0f * num_interesting / data.size());

    // ============================================================================
    // STEP 2: BUILD SPARSE TRANSITION MATRIX
    // ============================================================================
    cout << "\nBuilding sparse transition matrix...\n";
    clock_gettime(CLOCK_REALTIME, &GL_start);

    size_t n = data.size();
    
    // Weight coefficients (no pheromone term)
    float lambda2 = kappa * (1.0f - gamma);       // preference weight
    float lambda3 = gamma;                         // external weight

    printf("Weight coefficients: lambda2 (preference) = %.4f, lambda3 (external) = %.4f\n",
           lambda2, lambda3);

    // Count total non-zeros
    size_t total_nnz = 0;
    for (size_t i = 0; i < n; i++) {
        total_nnz += neighbourhoods[i].size();
        if (neighbourhoods[i].size() == 0) total_nnz++;  // self-loop
    }

    // Build sparse matrix using triplets
    std::vector<Eigen::Triplet<float>> triplets;
    triplets.reserve(total_nnz);

    #pragma omp parallel
    {
        std::vector<Eigen::Triplet<float>> local_triplets;
        local_triplets.reserve(total_nnz / numberofthreads + 1000);

        #pragma omp for schedule(dynamic, 100)
        for (long long i = 0; i < static_cast<long long>(n); i++) {
            std::vector<size_t> const &neighbourhood = neighbourhoods[i];
            size_t num_neighbors = neighbourhood.size();

            if (num_neighbors == 0) {
                // Self-loop for isolated points
                local_triplets.push_back(Eigen::Triplet<float>(i, i, 1.0f));
                continue;
            }

            if (interesting_particle[i] == 1) {
                // Compute normalized external weights for this neighborhood
                float ext_sum = 0.0f;
                for (size_t k = 0; k < num_neighbors; k++) {
                    ext_sum += external_weights[neighbourhood[k]];
                }

                // Compute unnormalized transition probabilities
                std::vector<float> probs(num_neighbors);
                float prob_sum = 0.0f;

                for (size_t k = 0; k < num_neighbors; k++) {
                    size_t j = neighbourhood[k];
                    float normalized_ext = (ext_sum > 0.0f) ? external_weights[j] / ext_sum : 1.0f / num_neighbors;
                    
                    float score = lambda2 * preferences[i][k] + lambda3 * normalized_ext;
                    probs[k] = expf(beta_antmovement * score);
                    prob_sum += probs[k];
                }

                // Normalize and add triplets
                for (size_t k = 0; k < num_neighbors; k++) {
                    size_t j = neighbourhood[k];
                    float p_ij = probs[k] / prob_sum;
                    if (p_ij > 1e-10f) {
                        local_triplets.push_back(Eigen::Triplet<float>(i, j, p_ij));
                    }
                }
            } else {
                // Non-interesting particle: uniform distribution
                float uniform_prob = 1.0f / num_neighbors;
                for (size_t k = 0; k < num_neighbors; k++) {
                    size_t j = neighbourhood[k];
                    local_triplets.push_back(Eigen::Triplet<float>(i, j, uniform_prob));
                }
            }
        }

        #pragma omp critical
        {
            triplets.insert(triplets.end(), local_triplets.begin(), local_triplets.end());
        }
    }

    // Create sparse matrix (row-stochastic)
    Eigen::SparseMatrix<float, Eigen::RowMajor> P(n, n);
    P.setFromTriplets(triplets.begin(), triplets.end());

    clock_gettime(CLOCK_REALTIME, &GL_finish);
    GL_times[1] = (GL_finish.tv_sec - GL_start.tv_sec) + 
                  (GL_finish.tv_nsec - GL_start.tv_nsec) / 1000000000.0;

    printf("Transition matrix: %zu x %zu, %ld non-zeros (%.4f%% density)\n",
           n, n, P.nonZeros(), 100.0 * P.nonZeros() / (double)(n * n));

    std::vector<Eigen::Triplet<float>>().swap(triplets);

    // ============================================================================
    // STEP 3: COMPUTE STATIONARY DISTRIBUTION VIA POWER ITERATION
    // ============================================================================
    cout << "\nComputing stationary distribution via power iteration...\n";
    clock_gettime(CLOCK_REALTIME, &GL_start);

    // Transpose for power iteration (π^T P = π^T => P^T π = π)
    Eigen::SparseMatrix<float, Eigen::ColMajor> PT = P.transpose();

    // Initialize with uniform distribution
    Eigen::VectorXf pi(n);
    pi.setConstant(1.0f / n);

    Eigen::VectorXf pi_new(n);
    float diff = 1.0f;
    size_t iter = 0;

    while (diff > tolerance && iter < max_iterations) {
        pi_new = PT * pi;

        // Normalize
        float sum = pi_new.sum();
        if (sum > 0.0f) {
            pi_new /= sum;
        }

        // L1 convergence criterion
        diff = (pi_new - pi).cwiseAbs().sum();
        pi = pi_new;
        iter++;

        if (iter % 100 == 0) {
            printf("  Iteration %zu: L1 diff = %.6e\n", iter, diff);
        }
    }

    clock_gettime(CLOCK_REALTIME, &GL_finish);
    GL_times[2] = (GL_finish.tv_sec - GL_start.tv_sec) + 
                  (GL_finish.tv_nsec - GL_start.tv_nsec) / 1000000000.0;

    if (diff <= tolerance) {
        printf("\nPower iteration converged after %zu iterations (diff = %.6e)\n", iter, diff);
    } else {
        printf("\nWARNING: Power iteration did not converge after %zu iterations (diff = %.6e)\n",
               iter, diff);
    }

    // ============================================================================
    // STEP 4: CONVERT TO OUTPUT FORMAT
    // ============================================================================
    std::vector<float> stationary_distribution(n);
    
    #pragma omp parallel for
    for (long long i = 0; i < static_cast<long long>(n); i++) {
        stationary_distribution[i] = pi(i);
    }

    // Statistics
    float min_val = *std::min_element(stationary_distribution.begin(), stationary_distribution.end());
    float max_val = *std::max_element(stationary_distribution.begin(), stationary_distribution.end());
    float mean_val = std::accumulate(stationary_distribution.begin(), stationary_distribution.end(), 0.0f) / n;

    printf("\nStationary distribution statistics:\n");
    printf("  Min: %.6e, Max: %.6e, Mean: %.6e\n", min_val, max_val, mean_val);

    // Timing
    float total_time = std::accumulate(GL_times.begin(), GL_times.end(), 0.0f);

    printf("\n\nLAAT MARKOV CHAIN EXECUTION TIMES ...\n\n");
    printf("TOTAL TIME = %.6f s\n\n", total_time);
    printf("Preprocessing = %.6f s = %.2f%%\n", GL_times[0], 100.0f * GL_times[0] / total_time);
    printf("Build transition matrix = %.6f s = %.2f%%\n", GL_times[1], 100.0f * GL_times[1] / total_time);
    printf("Power iteration = %.6f s = %.2f%%\n", GL_times[2], 100.0f * GL_times[2] / total_time);

    cout << "\nLocally Aligned Ant Technique (Markov Chain) completed\n\n";

    return stationary_distribution;
}
