#ifndef PYTORCH_MOTION_PLANNER_BENCHMARK_BENCHMARK_ADAPTER_H
#define PYTORCH_MOTION_PLANNER_BENCHMARK_BENCHMARK_ADAPTER_H

#include <base/Environment.h>
#include <base/PlannerSettings.h>
#include <base/PathStatistics.hpp>
#include "Position.h"


class BenchmarkAdapter {
    std::shared_ptr<Environment> mBenchmarkEnvironment;

    static void loadMovingAIScenarios(PlannerSettings::GlobalSettings &settings);

    static void loadOtherScenarios(PlannerSettings::GlobalSettings &settings);

    static ompl::geometric::PathGeometric omplPathFromPositions(const std::vector<Position> &resultPath);

public:
    explicit BenchmarkAdapter(PlannerSettings::GlobalSettings &settings);

    explicit BenchmarkAdapter(const std::string &settingFile);

    bool collides(double x, double y) { return mBenchmarkEnvironment->collides(x, y); }

    std::vector<bool> collides_positions(const std::vector<Position> &positions);

    std::tuple<double, double, double, double> bounds();

    static PathStatistics evaluate(const ompl::geometric::PathGeometric &path, const std::string &name);

    void evaluateAndSaveResult(const std::vector<Position> &resultPath, const std::string &name);

    Position start();

    Position goal();
};


#endif //PYTORCH_MOTION_PLANNER_BENCHMARK_BENCHMARK_ADAPTER_H
