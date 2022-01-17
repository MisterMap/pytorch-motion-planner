#ifndef PYTORCH_MOTION_PLANNER_BENCHMARK_BENCHMARK_ADAPTER_H
#define PYTORCH_MOTION_PLANNER_BENCHMARK_BENCHMARK_ADAPTER_H
#include <base/Environment.h>
#include <base/PlannerSettings.h>

class BenchmarkAdapter {
    std::shared_ptr<Environment> mBenchmarkEnvironment;
    static void loadMovingAIScenarios(PlannerSettings::GlobalSettings &settings);
    static void loadOtherScenarios(PlannerSettings::GlobalSettings &settings);
public:
    explicit BenchmarkAdapter(PlannerSettings::GlobalSettings &settings);
    bool collides(double x, double y) { return mBenchmarkEnvironment->collides(x, y); }
};


#endif //PYTORCH_MOTION_PLANNER_BENCHMARK_BENCHMARK_ADAPTER_H
