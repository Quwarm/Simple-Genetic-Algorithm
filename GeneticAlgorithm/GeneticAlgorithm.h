#include <cmath>
#include <algorithm>
#include <functional>
#include <vector>
#include <random>
#include <stack>

template <class T>
class GeneticAlgorithm {
 public:
    // Инициализация основных переменных в конструкторе
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

    // Инициализирует популяцию
    void generateInitialPopulation() {
        population_x_ = createNewPopulation();
        population_y_ = createNewPopulation();
    }

    // Создает и возвращает новую популяцию
    std::vector<T> createNewPopulation() {
        T minimum = section_.first;
        T maximum = section_.second;
        std::vector<T> result(number_of_population_members_);
        std::uniform_real_distribution<> real_dist(minimum, maximum); // Равномерное непрерывное распределение
        for (auto &k : result) {
            k = real_dist(random_gen_);
        }
        return result;
    }

    // Фитнес-функция (или функция пригодности)
    // Возвращает пары векторов лучших популяций X и Y (отбор по проценту выживаемости)
    std::pair<std::vector<T>, std::vector<T>> getBestMembers() {
        std::vector<T> function_values(number_of_population_members_);
        auto tempX = population_x_.begin();
        auto tempY = population_y_.begin();
        for (auto &k : function_values) {
            k = function_(*( tempX++ ), *( tempY++ ));
        }
        Sort(function_values, population_x_, population_y_); // Сортировка Хоара в классе
        auto amount_of_best_values = static_cast<int>(function_values.size() * percent_of_best_ones_to_live_);
        return {std::vector<T>(population_x_.begin(), population_x_.begin() + amount_of_best_values),
            std::vector<T>(population_y_.begin(), population_y_.begin() + amount_of_best_values)};
    }

    // Мутация популяций
    void mutate() {
        auto minimal_population_x = *std::min(population_x_.begin(), population_x_.end());
        auto minimal_population_y = *std::min(population_y_.begin(), population_y_.end());
        std::normal_distribution<> normal_dist {0, std::min(probability_ * 1000, 0.001)}; // нормальное распределение
        for (auto &elem : population_x_) {
            elem += minimal_population_x * normal_dist(random_gen_);
        }
        for (auto &elem : population_y_) {
            elem += minimal_population_y * normal_dist(random_gen_);
        }
    }

    // Рекомбинация (размножение)
    void crossover() {
        int population_x_length = population_x_.size();
        std::uniform_int_distribution<>
            uniform_dist(0, population_x_length - 1); // Равномерное дискретное распределение
        population_x_.resize(number_of_population_members_); // Увеличение
        population_y_.resize(number_of_population_members_); // Увеличение
        for (int i = population_x_length; i < number_of_population_members_; ++i) {
            population_x_[i] =
                ( population_x_[uniform_dist(random_gen_)] + population_x_[uniform_dist(random_gen_)] ) / 2.0;
            population_y_[i] =
                ( population_y_[uniform_dist(random_gen_)] + population_y_[uniform_dist(random_gen_)] ) / 2.0;
        }
    }

    // Поиск минимума функции (количество итераций в аргументах)
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

    // Получение индекса элемента с минимальным значением функции f(x, y)
    int getMinimalValueIndex() {
        std::vector<T> function_values(number_of_population_members_);
        auto tempX = population_x_.begin(), tempY = population_y_.begin();
        for (auto &k : function_values) {
            k = function_(*( tempX++ ), *( tempY++ ));
        }
        return std::min(function_values.begin(), function_values.end()) - function_values.begin();
    }

    // Получение X и Y координаты минимума
    std::pair<T, T> getArgumentsOfMinimumValue() {
        auto minimum_value_index = getMinimalValueIndex();
        return {population_x_[minimum_value_index], population_y_[minimum_value_index]};
    }

    GeneticAlgorithm(const GeneticAlgorithm &arg) = delete;
    GeneticAlgorithm &operator=(const GeneticAlgorithm &arg) = delete;
    GeneticAlgorithm &operator=(GeneticAlgorithm &arg) = delete;
    GeneticAlgorithm(GeneticAlgorithm &&arg) = delete;

 private:
    int number_of_population_members_; // Количество популяций
    double percent_of_best_ones_to_live_; // Процент выживаемости (лучшие выживают, остальные погибают)
    std::pair<T, T> section_; // Ограничения областей определения
    std::function<T(T, T)> function_; // Функция, которую необходимо минимизировать
    double probability_; // Точность
    std::vector<T> population_x_; // Популяция X
    std::vector<T> population_y_; // Популяция Y
    std::mt19937 random_gen_; // Генератор рандомных чисел

    // Итеративная быстрая сортировка Хоара для трехмерного массива по первому вектору functionValues
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
