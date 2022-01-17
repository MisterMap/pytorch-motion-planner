#include "BenchmarkAdapter.h"

#include <base/environments/GridMaze.h>
#include <utils/ScenarioLoader.h>


BenchmarkAdapter::BenchmarkAdapter(PlannerSettings::GlobalSettings &settings) {
    if (settings.benchmark.moving_ai.active) {
        loadMovingAIScenarios(settings);
    } else {
        loadOtherScenarios(settings);
    }
    mBenchmarkEnvironment = settings.environment;
}

void BenchmarkAdapter::loadMovingAIScenarios(PlannerSettings::GlobalSettings &settings) {
    ScenarioLoader scenarioLoader;
    scenarioLoader.load(settings.benchmark.moving_ai.scenario);
    auto &scenario = scenarioLoader.scenarios()[0];
    settings.environment = GridMaze::createFromMovingAiScenario(scenario);
    settings.env.collision.initializeCollisionModel();
}

void BenchmarkAdapter::loadOtherScenarios(PlannerSettings::GlobalSettings &settings) {
    settings.env.createEnvironment();
    settings.env.grid.seed += 1;
}
