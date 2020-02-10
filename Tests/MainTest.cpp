#define _USE_MATH_DEFINES
#include <iostream>
#include <iomanip>
#include <cmath>
#include "../GeneticAlgorithm/GeneticAlgorithm.h"

// Функция Экли
double AckleyFunction(double x, double y) {
    return -20 * exp(-0.2 * sqrt(0.5 * ( x * x + y * y ))) - exp(0.5 * ( cos(2 * M_PI * x) + cos(2 * M_PI * y) )) + M_E + 20;
}

// Функция Била
double BealFunction(double x, double y) {
    return pow(1.5 - x + x * y, 2) + pow(2.25 - x + x * y * y, 2) + pow(2.625 - x + x * y * y * y, 2);
}

// Функция Изома
double IzomFunction(double x, double y) {
    return -cos(x) * cos(y) * exp(-pow(x - M_PI, 2) - pow(y - M_PI, 2));
}

// Функция Розенброка
double RosenbrokFunction(double x, double y) {
    return ( 1.0 - x ) * ( 1.0 - x ) + 100.0 * ( y - x * x ) * ( y - x * x );
}

// Функция Шаффера-N2
double ShafferN2Function(double x, double y) {
    return 0.5 + ( pow(sin(x * x - y * y), 2) - 0.5 ) / pow(1 + 0.001 * ( x * x + y * y ), 2);
}

int main() {
    std::cout << std::setprecision(10);
    const double EPS = 1e-5;
    int numberOfPopulationMembers = 7500;
    int iterations = 1000;
    double percentOfBestOnesToLive = 0.7;
    std::pair<double, double> searchingSection = {-1, 4};
    GeneticAlgorithm<double>
        GA_A(numberOfPopulationMembers, percentOfBestOnesToLive, searchingSection, AckleyFunction, EPS);
    {
        std::cout << "Ackley Function" << std::endl;
        auto minimumValue = GA_A.searchMinimum(iterations);
        auto minimumPoint = GA_A.getArgumentsOfMinimumValue();
        std::cout << "Found Minimum: f(" << minimumPoint.first << ", " << minimumPoint.second << ") = "
                  << minimumValue << std::endl;
        std::cout << "Expected: f(0, 0) = 0" << std::endl;
        std::cout << "Max error: "
                  << std::max(fabs(minimumValue), std::max(fabs(minimumPoint.first), fabs(minimumPoint.second)))
                  << std::endl << std::endl;
    }
    GeneticAlgorithm<double>
        GA_B(numberOfPopulationMembers, percentOfBestOnesToLive, searchingSection, BealFunction, EPS);
    {
        std::cout << "Beal Function" << std::endl;
        auto minimumValue = GA_B.searchMinimum(iterations);
        auto minimumPoint = GA_B.getArgumentsOfMinimumValue();
        std::cout << "Found Minimum: f(" << minimumPoint.first << ", " << minimumPoint.second << ") = "
                  << minimumValue << std::endl;
        std::cout << "Expected: f(3, 0.5) = 0" << std::endl;
        std::cout << "Max error: " << std::max(fabs(minimumValue),
                                               std::max(fabs(3 - minimumPoint.first), fabs(0.5 - minimumPoint.second)))
                  << std::endl << std::endl;
    }
    GeneticAlgorithm<double>
        GA_I(numberOfPopulationMembers, percentOfBestOnesToLive, searchingSection, IzomFunction, EPS);
    {
        std::cout << "Izom Function" << std::endl;
        auto minimumValue = GA_I.searchMinimum(iterations);
        auto minimumPoint = GA_I.getArgumentsOfMinimumValue();
        std::cout << "Found Minimum: f(" << minimumPoint.first << ", " << minimumPoint.second << ") = "
                  << minimumValue << std::endl;
        std::cout << "Expected: f(" << M_PI << ", " << M_PI << ") = -1" << std::endl;
        std::cout << "Max error: " << std::max(fabs(-1 - minimumValue),
                                               std::max(fabs(M_PI - minimumPoint.first),
                                                        fabs(M_PI - minimumPoint.second))) << std::endl << std::endl;
    }
    GeneticAlgorithm<double>
        GA_R(numberOfPopulationMembers, percentOfBestOnesToLive, searchingSection, RosenbrokFunction, EPS);
    {
        std::cout << "Rosenbrok Function" << std::endl;
        auto minimumValue = GA_R.searchMinimum(iterations);
        auto minimumPoint = GA_R.getArgumentsOfMinimumValue();
        std::cout << "Found Minimum: f(" << minimumPoint.first << ", " << minimumPoint.second << ") = "
                  << minimumValue << std::endl;
        std::cout << "Expected: f(1, 1) = 0" << std::endl;
        std::cout << "Max error: "
                  << std::max(fabs(minimumValue), std::max(fabs(1 - minimumPoint.first), fabs(1 - minimumPoint.second)))
                  << std::endl << std::endl;
    }
    GeneticAlgorithm<double>
        GA_S(numberOfPopulationMembers, percentOfBestOnesToLive, searchingSection, ShafferN2Function, EPS);
    {
        std::cout << "Shaffer-N2 Function" << std::endl;
        auto minimumValue = GA_S.searchMinimum(iterations);
        auto minimumPoint = GA_S.getArgumentsOfMinimumValue();
        std::cout << "Found Minimum: f(" << minimumPoint.first << ", " << minimumPoint.second << ") = "
                  << minimumValue << std::endl;
        std::cout << "Expected: f(0, 0) = 0" << std::endl;
        std::cout << "Max error: "
                  << std::max(fabs(minimumValue), std::max(fabs(minimumPoint.first), fabs(minimumPoint.second)))
                  << std::endl << std::endl;
    }
}
