// Replays tests/robot/copa_golden.json (written by tools/export_robot_policies.py from the
// Python implementations) against robot/Src/Tools/CopaPenalty/CopaPolicies.h, the header
// that runs on the NAO. Exit code 0 only if every step matches.
#include "Tools/CopaPenalty/CopaPolicies.h"

#include "json.hpp"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>

using nlohmann::json;
using namespace CopaPenalty;

namespace
{
  int failures = 0;
  int checks = 0;

  void expect(bool condition, const std::string& what)
  {
    ++checks;
    if(!condition && ++failures <= 25)
      std::cerr << "MISMATCH " << what << "\n";
  }

  void near(double actual, double expected, double tolerance, const std::string& what)
  {
    const double scale = std::max(1., std::abs(expected));
    expect(std::abs(actual - expected) <= tolerance * scale,
           what + " actual=" + std::to_string(actual) + " expected=" + std::to_string(expected));
  }

  DenseLayer layer(const json& data)
  {
    DenseLayer result;
    const json& weights = data.at("weights");
    result.rows = weights.size();
    result.cols = weights.at(0).size();
    for(const json& row : weights)
      for(const json& value : row)
        result.weights.push_back(value.get<double>());
    for(const json& value : data.at("bias"))
      result.bias.push_back(value.get<double>());
    return result;
  }

  void strikers(const json& golden)
  {
    for(const auto& [name, entry] : golden.at("strikers").items())
    {
      const json& p = entry.at("policy");
      MlpPolicyParameters parameters;
      parameters.name = name;
      parameters.history = p.at("history").get<std::size_t>();
      parameters.obsMean = p.at("obsMean").get<std::vector<double>>();
      parameters.obsVar = p.at("obsVar").get<std::vector<double>>();
      parameters.clipObs = p.at("clipObs").get<double>();
      parameters.epsilon = p.at("epsilon").get<double>();
      for(const json& hidden : p.at("hidden"))
        parameters.hidden.push_back(layer(hidden));
      parameters.action = layer(p.at("action"));
      expect(parameters.valid(), name + " parameters valid");
      MlpPolicy policy(parameters);
      int episode = 0;
      for(const json& steps : entry.at("cases"))
      {
        policy.reset();
        int step = 0;
        for(const json& item : steps)
        {
          const std::string where = name + " episode " + std::to_string(episode) + " step " + std::to_string(step++);
          Observation raw{};
          const std::vector<double> values = item.at("observation").get<std::vector<double>>();
          std::copy(values.begin(), values.end(), raw.begin());
          const std::vector<double> normalized = policy(clipObservation(raw));
          const std::vector<double> expected = item.at("normalized").get<std::vector<double>>();
          for(std::size_t i = 0; i < 4; ++i)
            near(normalized[i], expected[i], 2e-5, where + " a" + std::to_string(i));
          const StrikerAction action = StrikerAction::fromNormalized(normalized);
          near(action.targetY, item.at("target_y").get<double>(), 2e-5, where + " target_y");
          expect(action.rightFoot == item.at("right_foot").get<bool>(), where + " foot");
          near(action.power, item.at("power").get<double>(), 2e-5, where + " power");
          near(action.trigger, item.at("trigger").get<double>(), 2e-5, where + " trigger");
          expect(action.fires() == (item.at("trigger").get<double>() >= .5), where + " fires");
        }
        ++episode;
      }
    }
  }

  void observations(const json& golden)
  {
    const char* names[] = {"time_s", "ball_x", "ball_y", "ball_vx", "ball_vy", "ball_age", "opponent_x",
                           "opponent_y", "opponent_yaw", "goal_center_bearing", "left_post_bearing",
                           "right_post_bearing", "torso_roll", "torso_pitch", "gyro_x", "gyro_y", "gyro_z",
                           "remaining_time", "ball_visible"};
    int index = 0;
    for(const json& steps : golden.at("observations"))
    {
      ObservationBuilder builder;
      int step = 0;
      for(const json& item : steps)
      {
        const json& s = item.at("state");
        WorldState state;
        state.time = s.at("time").get<double>();
        state.remainingTime = s.at("remaining").get<double>();
        state.ball = {s.at("ball")[0].get<double>(), s.at("ball")[1].get<double>()};
        state.robot = {s.at("robot")[0].get<double>(), s.at("robot")[1].get<double>(), s.at("robot")[2].get<double>()};
        state.opponent = {s.at("opponent")[0].get<double>(), s.at("opponent")[1].get<double>(), s.at("opponent")[2].get<double>()};
        state.torsoRoll = s.at("torso_roll").get<double>();
        state.torsoPitch = s.at("torso_pitch").get<double>();
        for(int i = 0; i < 3; ++i)
          state.gyro[i] = s.at("gyro")[i].get<double>();
        state.ballVisible = s.at("ball_visible").get<bool>();
        const Observation obs = builder.build(state);
        for(std::size_t i = 0; i < observationSize; ++i)
          // Python stores the observation as float32.
          near(static_cast<double>(static_cast<float>(obs[i])), item.at("observation").at(names[i]).get<double>(), 1e-6,
               "observation case " + std::to_string(index) + " step " + std::to_string(step) + " " + names[i]);
        ++step;
      }
      ++index;
    }
  }

  void keepers(const json& golden)
  {
    int index = 0;
    for(const json& entry : golden.at("keepers"))
    {
      const std::string modeName = entry.at("mode").get<std::string>();
      const LibraryMode mode = modeName == "balanced" ? LibraryMode::balanced : LibraryMode::original;
      LibraryKeeper keeper(mode);
      BallTracker tracker;
      int step = 0;
      for(const json& item : entry.at("steps"))
      {
        const std::string where = "keeper case " + std::to_string(index) + " (" + modeName + ") step " + std::to_string(step++);
        const double capture = item.at("capture").get<double>();
        const Pose odometry{item.at("odometry")[0].get<double>(), item.at("odometry")[1].get<double>(),
                            item.at("odometry")[2].get<double>()};
        std::optional<BallMeasurement> measurement;
        if(!item.at("measurement").is_null())
        {
          const json& m = item.at("measurement");
          BallMeasurement value;
          value.position = {m.at("position")[0].get<double>(), m.at("position")[1].get<double>()};
          for(int i = 0; i < 2; ++i)
            for(int j = 0; j < 2; ++j)
              value.covariance[i][j] = m.at("covariance")[i][j].get<double>();
          measurement = value;
        }
        tracker.update(capture, capture, measurement, odometry);
        const BallEstimate ball = tracker.estimate(capture, odometry);
        const json& e = item.at("estimate");
        near(ball.position.x, e.at("position")[0].get<double>(), 1e-9, where + " position x");
        near(ball.position.y, e.at("position")[1].get<double>(), 1e-9, where + " position y");
        near(ball.velocity.x, e.at("velocity")[0].get<double>(), 1e-9, where + " velocity x");
        near(ball.velocity.y, e.at("velocity")[1].get<double>(), 1e-9, where + " velocity y");
        near(ball.velocityCovariance[0][0], e.at("velocity_variance_x").get<double>(), 1e-9, where + " velocity variance");
        near(ball.ageS, e.at("age_s").get<double>(), 1e-9, where + " age");
        expect(ball.positionValid == e.at("position_valid").get<bool>(), where + " position_valid");
        expect(ball.velocityValid == e.at("velocity_valid").get<bool>(), where + " velocity_valid");
        expect(ball.visible == e.at("visible").get<bool>(), where + " visible");
        expect(ball.reason == e.at("reason").get<std::string>(), where + " tracker reason " + ball.reason);
        const LibraryOutput output = keeper(capture, capture, std::max(0., 28. - capture), ball);
        expect(defenseName(output.defense) == item.at("defense").get<std::string>(),
               where + " defense " + defenseName(output.defense) + " vs " + item.at("defense").get<std::string>());
        near(output.trigger, item.at("trigger").get<double>(), 1e-12, where + " trigger");
        expect(keeper.getDecision().reason == item.at("reason").get<std::string>(),
               where + " reason " + keeper.getDecision().reason + " vs " + item.at("reason").get<std::string>());
        if(!item.at("crossing_y").is_null())
        {
          expect(keeper.getDecision().crossingY.has_value(), where + " crossing present");
          if(keeper.getDecision().crossingY)
            near(*keeper.getDecision().crossingY, item.at("crossing_y").get<double>(), 1e-9, where + " crossing_y");
        }
      }
      ++index;
    }
  }
}

int main(int argc, char** argv)
{
  if(argc != 2)
  {
    std::cerr << "usage: copa_policy_tests <copa_golden.json>\n";
    return 2;
  }
  std::ifstream stream(argv[1]);
  if(!stream)
  {
    std::cerr << "cannot open " << argv[1] << "\n";
    return 2;
  }
  const json golden = json::parse(stream);
  strikers(golden);
  observations(golden);
  keepers(golden);
  std::cout << "checks=" << checks << " failures=" << failures << "\n";
  return failures == 0 ? 0 : 1;
}
