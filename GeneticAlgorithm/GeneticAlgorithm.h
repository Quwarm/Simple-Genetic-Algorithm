#include <cmath>
#include <algorithm>
#include <functional>
#include <vector>
#include <random>
#include <stack>

template <class T>
class GeneticAlgorithm {
 public:
    // Initialization of the main variables in the constructor
    GeneticAlgorithm(int number_of_population_members,
                     double percent_of_best_ones_to_live,
                     std::pair<T, T> section,
                     std::function<T(T, T)> function,
                     double probability)
        : number_of_population_members_(number_of_population_members),
          percent_of_best_ones_to_live_(percent_of_best_ones_to_live),
          section_(std::move(section)),
          function_(std::move(function)),
          probability_(probability) {
        std::random_device rd;
        random_gen_.seed(rd());
    }

    // Initialize a population
    void generateInitialPopulation() {
        population_x_ = createNewPopulation();
        population_y_ = createNewPopulation();
    }

    // Create and return a new population
    std::vector<T> createNewPopulation() {
        T minimum = section_.first;
        T maximum = section_.second;
        std::vector<T> result(number_of_population_members_);
        std::uniform_real_distribution<> real_dist(minimum, maximum);
        for (auto &k : result) {
            k = real_dist(random_gen_);
        }
        return result;
    }

    // Fitness function
    // Return pairs of vectors of the best populations X and Y (selection by percentage of survival)
    std::pair<std::vector<T>, std::vector<T>> getBestMembers() {
        std::vector<T> function_values(number_of_population_members_);
        auto tempX = population_x_.begin();
        auto tempY = population_y_.begin();
        for (auto &k : function_values) {
            k = function_(*( tempX++ ), *( tempY++ ));
        }
        Sort(function_values, population_x_, population_y_); // Hoare sort in class
        auto amount_of_best_values = static_cast<int>(function_values.size() * percent_of_best_ones_to_live_);
        return {std::vector<T>(population_x_.begin(), population_x_.begin() + amount_of_best_values),
            std::vector<T>(population_y_.begin(), population_y_.begin() + amount_of_best_values)};
    }

    // Population mutation
    void mutate() {
        auto minimal_population_x = *std::min(population_x_.begin(), population_x_.end());
        auto minimal_population_y = *std::min(population_y_.begin(), population_y_.end());
        std::normal_distribution<> normal_dist {0, std::min(probability_ * 1000, 0.001)};
        for (auto &elem : population_x_) {
            elem += minimal_population_x * normal_dist(random_gen_);
        }
        for (auto &elem : population_y_) {
            elem += minimal_population_y * normal_dist(random_gen_);
        }
    }

    // Recombination (reproduction)
    void crossover() {
        int population_x_length = population_x_.size();
        std::uniform_int_distribution<>
            uniform_dist(0, population_x_length - 1);
        population_x_.resize(number_of_population_members_); // Increase
        population_y_.resize(number_of_population_members_); // Increase
        for (int i = population_x_length; i < number_of_population_members_; ++i) {
            population_x_[i] =
                ( population_x_[uniform_dist(random_gen_)] + population_x_[uniform_dist(random_gen_)] ) / 2.0;
            population_y_[i] =
                ( population_y_[uniform_dist(random_gen_)] + population_y_[uniform_dist(random_gen_)] ) / 2.0;
        }
    }

    // Search for the minimum of the function (the number of iterations in the arguments)
    T searchMinimum(int iterations) {
        generateInitialPopulation();
        for (int i = 0; i < iterations; ++i) {
            auto temp_population = getBestMembers();
            population_x_ = temp_population.first;
            population_y_ = temp_population.second;
            crossover();
            mutate();
        }
        auto minimumValueIndex = getMinimalValueIndex();
        return function_(population_x_[minimumValueIndex], population_y_[minimumValueIndex]);
    }

    // Get the index of an element with the minimum value of the function f (x, y)
    int getMinimalValueIndex() {
        std::vector<T> function_values(number_of_population_members_);
        auto tempX = population_x_.begin(), tempY = population_y_.begin();
        for (auto &k : function_values) {
            k = function_(*( tempX++ ), *( tempY++ ));
        }
        return std::min(function_values.begin(), function_values.end()) - function_values.begin();
    }

    // Getting the X and Y coordinates of the minimum
    std::pair<T, T> getArgumentsOfMinimumValue() {
        auto minimum_value_index = getMinimalValueIndex();
        return {population_x_[minimum_value_index], population_y_[minimum_value_index]};
    }

    GeneticAlgorithm(const GeneticAlgorithm &arg) = delete;
    GeneticAlgorithm &operator=(const GeneticAlgorithm &arg) = delete;
    GeneticAlgorithm &operator=(GeneticAlgorithm &arg) = delete;
    GeneticAlgorithm(GeneticAlgorithm &&arg) = delete;

 private:
    int number_of_population_members_; // Number of populations
    double percent_of_best_ones_to_live_; // Survival percentage (the best survive, the rest die)
    std::pair<T, T> section_; // Limitations of scopes
    std::function<T(T, T)> function_; // Function to be minimized
    double probability_; // Probability
    std::vector<T> population_x_; // Population X
    std::vector<T> population_y_; // Population Y
    std::mt19937 random_gen_; // Random number generator

    // Iterative quick Hoar sort for three-dimensional array by first vector functionValues
    static void Sort(std::vector<T> &functionValues, std::vector<T> &populationX, std::vector<T> &populationY) {
        int Left = 0, Right = functionValues.size() - 1, L2, R2;
        T PivotValue, Temp;
        std::stack<T> Lows, Highs;
        Lows.push(Left);
        Highs.push(Right);
        while (!Lows.empty()) {
            Left = Lows.top();
            Lows.pop();
            Right = Highs.top();
            Highs.pop();
            L2 = Left;
            R2 = Right;
            PivotValue = functionValues[( Left + Right ) / 2];
            do {
                while (functionValues[L2] < PivotValue) { ++L2; }
                while (functionValues[R2] > PivotValue) { --R2; }
                if (L2 <= R2) {
                    if (functionValues[L2] > functionValues[R2]) {
                        std::swap(functionValues[L2], functionValues[R2]);
                        std::swap(populationX[L2], populationX[R2]);
                        std::swap(populationY[L2], populationY[R2]);
                    }
                    ++L2;
                    if (R2 > 0) {
                        --R2;
                    }
                }
            } while (L2 <= R2);
            if (L2 < Right) {
                Lows.push(L2);
                Highs.push(Right);
            }
            if (R2 > Left) {
                Lows.push(Left);
                Highs.push(R2);
            }
        }
    }
};
