/**
 * @file CopaPolicies.h
 *
 * Portable C++ versions of the PenaltyStack tactical layer (Copa NAO CCM MX 2026,
 * Tiro Penal). The code depends only on the C++ standard library, so the very
 * same header runs on the NAO (inside the B-Human modules) and in the parity
 * tests that compare it with the Python implementations it was ported from:
 *
 *  - Field, ObservationBuilder, RLObservation  <- penalty/rl/{geometry,observation,contracts}.py
 *  - MlpPolicy (deterministic SB3 actor)           <- penalty/rl/actors.py NumpyPolicy + obs_norm
 *  - StrikerAction                                 <- penalty/rl/contracts.py, penalty/skills/kick.py
 *  - BallTracker                                   <- penalty/perception.py BallTracker
 *  - VisionKeeper (vision_keeper_v2)               <- penalty/keeper_vision.py
 *  - LibraryKeeper (vision_keeper_v6 modes)        <- penalty/keeper_library.py
 *
 * Units: metres, seconds, radians. Robot frame: x forward, y left.
 * The field frame is the one of sim/scene.xml: the goal line is at x = -1.2 m
 * and the striker starts at (0.3, 0) looking towards -x.
 */

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <deque>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace CopaPenalty
{
  constexpr double pi = 3.14159265358979323846;

  /** Python's (angle + pi) % (2 pi) - pi, i.e. a result in [-pi, pi). */
  inline double wrap(double angle)
  {
    double result = std::fmod(angle + pi, 2. * pi);
    if(result < 0.)
      result += 2. * pi;
    return result - pi;
  }

  inline bool finite(double value) { return std::isfinite(value); }

  struct Vec2
  {
    double x = 0.;
    double y = 0.;
  };

  struct Pose
  {
    double x = 0.;
    double y = 0.;
    double yaw = 0.;

    Vec2 translation() const { return {x, y}; }
  };

  /** pose (+) relative: apply a robot-relative displacement to a field pose. */
  inline Pose compose(const Pose& pose, const Pose& relative)
  {
    const double c = std::cos(pose.yaw), s = std::sin(pose.yaw);
    return {pose.x + c * relative.x - s * relative.y, pose.y + s * relative.x + c * relative.y,
            wrap(pose.yaw + relative.yaw)};
  }

  /** The displacement that leads from `from` to `to`, expressed in `from`'s frame. */
  inline Pose between(const Pose& from, const Pose& to)
  {
    const double dx = to.x - from.x, dy = to.y - from.y;
    const double c = std::cos(from.yaw), s = std::sin(from.yaw);
    return {c * dx + s * dy, -s * dx + c * dy, wrap(to.yaw - from.yaw)};
  }

  inline Vec2 toRobot(const Vec2& point, const Vec2& origin, double yaw)
  {
    const double dx = point.x - origin.x, dy = point.y - origin.y;
    const double c = std::cos(yaw), s = std::sin(yaw);
    return {c * dx + s * dy, -s * dx + c * dy};
  }

  inline Vec2 toField(const Vec2& point, const Pose& pose)
  {
    const double c = std::cos(pose.yaw), s = std::sin(pose.yaw);
    return {pose.x + c * point.x - s * point.y, pose.y + s * point.x + c * point.y};
  }

  inline Vec2 rotateToRobot(const Vec2& vector, double yaw)
  {
    const double c = std::cos(yaw), s = std::sin(yaw);
    return {c * vector.x + s * vector.y, -s * vector.x + c * vector.y};
  }

  inline double bearing(const Vec2& point, const Vec2& origin, double yaw)
  {
    const Vec2 local = toRobot(point, origin, yaw);
    return std::atan2(local.y, local.x);
  }

  /** Field of the Copa NAO CCM MX 2026 penalty category (rules 3.3.4), sim/scene.xml frame. */
  struct Field
  {
    double goalLineX = -1.2;        /**< Goal line; the penalty mark is 1.30 m in front of it. */
    double postY = .75;             /**< Half of the 1.50 m goal width. */
    double ballRadius = .0325;      /**< 65 mm official ball. */
    double penaltyMarkX = .1;
    Pose strikerStart{.3, 0., pi};  /**< 20 cm behind the penalty mark, facing the goal. */
    Pose keeperStart{-1.08, 0., 0.};/**< Feet on the goal line, facing the striker. */
    double penaltySeconds = 28.;    /**< Active budget (rules: 30 s after the whistle). */

    Vec2 goalCenter() const { return {goalLineX, 0.}; }
    /** Posts as named by a striker facing the goal (its left is field -y). */
    Vec2 leftPost() const { return {goalLineX, -postY}; }
    Vec2 rightPost() const { return {goalLineX, postY}; }
  };

  // ---------------------------------------------------------------------------
  // rl_obs_v1
  // ---------------------------------------------------------------------------

  constexpr std::size_t observationSize = 19;
  using Observation = std::array<double, observationSize>;

  /** Field order of penalty.rl.contracts.RLObservation. */
  enum ObservationField : std::size_t
  {
    timeS, ballX, ballY, ballVx, ballVy, ballAge, opponentX, opponentY, opponentYaw,
    goalCenterBearing, leftPostBearing, rightPostBearing, torsoRoll, torsoPitch,
    gyroX, gyroY, gyroZ, remainingTime, ballVisible
  };

  /** OBS_LOW/OBS_HIGH of penalty.rl.contracts. */
  inline const Observation& observationLow()
  {
    static const Observation low = {0., -6., -6., -12., -12., 0., -6., -6., -pi, -pi, -pi, -pi, -pi, -pi,
                                     -30., -30., -30., 0., 0.};
    return low;
  }

  inline const Observation& observationHigh()
  {
    static const Observation high = {40., 6., 6., 12., 12., 40., 6., 6., pi, pi, pi, pi, pi, pi,
                                      30., 30., 30., 28., 1.};
    return high;
  }

  /** clip_observation(), evaluated in float32 like numpy. */
  inline Observation clipObservation(const Observation& values)
  {
    Observation result{};
    for(std::size_t i = 0; i < observationSize; ++i)
    {
      const float value = static_cast<float>(values[i]);
      result[i] = std::clamp(value, static_cast<float>(observationLow()[i]), static_cast<float>(observationHigh()[i]));
    }
    return result;
  }

  /** Ground-frame snapshot (penalty.rl.observation.WorldState). */
  struct WorldState
  {
    double time = 0.;
    double remainingTime = 28.;
    Vec2 ball;
    Pose robot;
    Pose opponent;
    double torsoRoll = 0.;
    double torsoPitch = 0.;
    std::array<double, 3> gyro{};
    bool ballVisible = true;
  };

  /** penalty.rl.observation.ObservationBuilder: ball velocity is a finite difference of visible estimates. */
  class ObservationBuilder
  {
  public:
    void reset()
    {
      previous.reset();
      velocity = Vec2();
      lastSeen.reset();
      ballXY.reset();
    }

    Observation build(const WorldState& state, const Field& geometry = Field())
    {
      if(state.ballVisible)
      {
        if(previous && state.time > previous->first)
        {
          const double dt = state.time - previous->first;
          velocity = {(state.ball.x - previous->second.x) / dt, (state.ball.y - previous->second.y) / dt};
        }
        previous = std::make_pair(state.time, state.ball);
        lastSeen = state.time;
        ballXY = state.ball;
      }
      const Vec2 ball = ballXY ? *ballXY : state.ball;
      const double age = lastSeen ? std::max(0., state.time - *lastSeen) : 0.;
      const Vec2 origin = state.robot.translation();
      const double yaw = state.robot.yaw;
      const Vec2 b = toRobot(ball, origin, yaw);
      const Vec2 v = rotateToRobot(velocity, yaw);
      const Vec2 o = toRobot(state.opponent.translation(), origin, yaw);
      Observation obs{};
      obs[timeS] = state.time;
      obs[ballX] = b.x;
      obs[ballY] = b.y;
      obs[ballVx] = v.x;
      obs[ballVy] = v.y;
      obs[ballAge] = age;
      obs[opponentX] = o.x;
      obs[opponentY] = o.y;
      obs[opponentYaw] = wrap(state.opponent.yaw - yaw);
      obs[goalCenterBearing] = bearing(geometry.goalCenter(), origin, yaw);
      obs[leftPostBearing] = bearing(geometry.leftPost(), origin, yaw);
      obs[rightPostBearing] = bearing(geometry.rightPost(), origin, yaw);
      obs[torsoRoll] = state.torsoRoll;
      obs[torsoPitch] = state.torsoPitch;
      obs[gyroX] = state.gyro[0];
      obs[gyroY] = state.gyro[1];
      obs[gyroZ] = state.gyro[2];
      obs[remainingTime] = std::max(0., state.remainingTime);
      obs[ballVisible] = state.ballVisible ? 1. : 0.;
      return obs;
    }

  private:
    std::optional<std::pair<double, Vec2>> previous;
    Vec2 velocity;
    std::optional<double> lastSeen;
    std::optional<Vec2> ballXY;
  };

  // ---------------------------------------------------------------------------
  // Deterministic SB3 actor (tanh MLP + linear head, clip(mean) to [-1, 1])
  // ---------------------------------------------------------------------------

  struct DenseLayer
  {
    std::size_t rows = 0;          /**< Outputs. */
    std::size_t cols = 0;          /**< Inputs. */
    std::vector<double> weights;   /**< Row-major rows x cols. */
    std::vector<double> bias;      /**< rows values. */

    bool valid() const { return rows > 0 && cols > 0 && weights.size() == rows * cols && bias.size() == rows; }

    std::vector<double> apply(const std::vector<double>& input, bool activation) const
    {
      std::vector<double> output(rows);
      for(std::size_t r = 0; r < rows; ++r)
      {
        double sum = bias[r];
        const double* row = weights.data() + r * cols;
        for(std::size_t c = 0; c < cols; ++c)
          sum += row[c] * input[c];
        output[r] = activation ? std::tanh(sum) : sum;
      }
      return output;
    }
  };

  struct MlpPolicyParameters
  {
    std::string name;
    std::size_t history = 1;
    std::vector<double> obsMean;   /**< history * 19 values (VecNormalize). */
    std::vector<double> obsVar;
    double clipObs = 10.;
    double epsilon = 1e-8;
    std::vector<DenseLayer> hidden;
    DenseLayer action;

    bool valid() const
    {
      const std::size_t inputs = history * observationSize;
      if(history < 1 || obsMean.size() != inputs || obsVar.size() != inputs || hidden.empty() || !action.valid())
        return false;
      std::size_t width = inputs;
      for(const DenseLayer& layer : hidden)
      {
        if(!layer.valid() || layer.cols != width)
          return false;
        width = layer.rows;
      }
      return action.cols == width;
    }
  };

  /** penalty.rl.policies.trained.TrainedPolicy with deterministic=True, computed like NumpyPolicy. */
  class MlpPolicy
  {
  public:
    MlpPolicy() = default;
    explicit MlpPolicy(MlpPolicyParameters parameters) : parameters(std::move(parameters)) {}

    bool valid() const { return parameters.valid(); }
    const MlpPolicyParameters& getParameters() const { return parameters; }
    void reset() { frames.clear(); }

    /** Normalized action in [-1, 1]; the caller decodes it with the role's contract. */
    std::vector<double> operator()(const Observation& observation)
    {
      if(frames.empty())
        frames.assign(parameters.history, observation);
      else
      {
        frames.push_back(observation);
        while(frames.size() > parameters.history)
          frames.pop_front();
      }
      std::vector<double> input;
      input.reserve(parameters.history * observationSize);
      std::size_t index = 0;
      for(const Observation& frame : frames)
        for(const double raw : frame)
        {
          // float32 observation, float64 statistics, float32 result (numpy semantics).
          const double value = static_cast<double>(static_cast<float>(raw));
          const double normalized = (value - parameters.obsMean[index]) / std::sqrt(parameters.obsVar[index] + parameters.epsilon);
          input.push_back(static_cast<double>(static_cast<float>(std::clamp(normalized, -parameters.clipObs, parameters.clipObs))));
          ++index;
        }
      for(const DenseLayer& layer : parameters.hidden)
        input = layer.apply(input, true);
      std::vector<double> mean = parameters.action.apply(input, false);
      for(double& value : mean)
        value = std::clamp(static_cast<double>(static_cast<float>(value)), -1., 1.);
      return mean;
    }

  private:
    MlpPolicyParameters parameters;
    std::deque<Observation> frames;
  };

  // ---------------------------------------------------------------------------
  // striker_action_v1 and the forwardFast kick reach
  // ---------------------------------------------------------------------------

  constexpr double targetYLimit = .65;   /**< Inside the posts minus ball radius and margin. */
  constexpr double kickLengthMinMm = 1500.;
  constexpr double kickLengthMaxMm = 2900.;
  constexpr double nominalKickLengthMm = 2500.;

  struct StrikerAction
  {
    double targetY = 0.;    /**< Goal-line aim point, positive to the striker's left. */
    bool rightFoot = false; /**< KICK_RIGHT (forwardFastRight) if true, else forwardFastLeft. */
    double power = (nominalKickLengthMm - kickLengthMinMm) / (kickLengthMaxMm - kickLengthMinMm);
    double trigger = 1.;

    bool fires() const { return trigger >= .5; }
    double kickLengthMm() const { return kickLengthMinMm + (kickLengthMaxMm - kickLengthMinMm) * power; }

    static StrikerAction fromNormalized(const std::vector<double>& values)
    {
      StrikerAction action;
      if(values.size() != 4)
        return action;
      const auto clipped = [](double value) { return std::clamp(value, -1., 1.); };
      action.targetY = clipped(values[0]) * targetYLimit;
      action.rightFoot = clipped(values[1]) > 0.;
      action.power = (clipped(values[2]) + 1.) / 2.;
      action.trigger = (clipped(values[3]) + 1.) / 2.;
      return action;
    }

    /** The project's scripted striker: centre, left foot, nominal reach, immediate. */
    static StrikerAction scripted() { return StrikerAction(); }
  };

  // ---------------------------------------------------------------------------
  // Ball tracking (penalty.perception.BallTracker)
  // ---------------------------------------------------------------------------

  using Covariance = std::array<std::array<double, 2>, 2>;

  inline Covariance rotateCovariance(const Covariance& cov, double yaw)
  {
    const double c = std::cos(yaw), s = std::sin(yaw);
    const double a = cov[0][0], b = cov[0][1], d = cov[1][1];
    const double offDiagonal = c * s * (a - d) + (c * c - s * s) * b;
    return {{{c * c * a - 2. * c * s * b + s * s * d, offDiagonal},
             {offDiagonal, s * s * a + 2. * c * s * b + c * c * d}}};
  }

  struct BallMeasurement
  {
    Vec2 position;          /**< Robot-relative ball centre on the ground (m). */
    Covariance covariance{};/**< Robot-relative measurement covariance (m^2). */
  };

  struct BallEstimate
  {
    Vec2 position;               /**< Robot-relative, advanced within the freshness window. */
    Vec2 velocity;               /**< Robot-relative, m/s. */
    bool positionValid = false;
    bool velocityValid = false;
    bool visible = false;
    double ageS = 40.;
    Covariance positionCovariance{};
    Covariance velocityCovariance{};
    std::string reason = "unseen";
  };

  class BallTracker
  {
  public:
    explicit BallTracker(double maxAge = .25, double maxSpeed = 8., double maxGap = .25) :
      maxAge(maxAge), maxSpeed(maxSpeed), maxGap(maxGap) {}

    void reset()
    {
      lastCapture.reset();
      lastSeen.reset();
      point.reset();
      pointCov = Covariance{};
      velocity = Vec2();
      velocityCov = Covariance{};
      velocityValid = false;
      reason = "unseen";
    }

    /** @return Whether the measurement was accepted. `odometry` is the robot's ground pose. */
    bool update(double captureS, double receivedS, const std::optional<BallMeasurement>& measurement, const Pose& odometry)
    {
      if(!finite(captureS) || !finite(receivedS) || captureS > receivedS + 1e-6)
      {
        reason = "invalid_time";
        return false;
      }
      if(lastCapture && captureS <= *lastCapture)
      {
        reason = captureS == *lastCapture ? "duplicate" : "out_of_order";
        return false;
      }
      lastCapture = captureS;
      if(receivedS - captureS > maxAge)
      {
        reason = "delayed_frame";
        return false;
      }
      if(!measurement)
      {
        reason = "not_seen";
        return false;
      }
      const Covariance& m = measurement->covariance;
      if(!finite(measurement->position.x) || !finite(measurement->position.y) || m[0][0] < 0. || m[1][1] < 0. ||
         std::abs(m[0][1] - m[1][0]) > 1e-6 || m[0][0] * m[1][1] < m[0][1] * m[1][0] - 1e-12)
      {
        reason = "invalid_measurement";
        return false;
      }
      const Vec2 ground = toField(measurement->position, odometry);
      const Covariance cov = rotateCovariance(m, odometry.yaw);
      velocityValid = false;
      if(lastSeen)
      {
        const double dt = captureS - *lastSeen;
        if(dt >= .025 && dt <= maxGap)
        {
          const Vec2 candidate = {(ground.x - point->x) / dt, (ground.y - point->y) / dt};
          if(std::hypot(candidate.x, candidate.y) <= maxSpeed)
          {
            velocity = candidate;
            for(int i = 0; i < 2; ++i)
              for(int j = 0; j < 2; ++j)
                velocityCov[i][j] = (cov[i][j] + pointCov[i][j]) / (dt * dt);
            velocityValid = true;
          }
          else
          {
            reason = "position_outlier";
            // One rejected candidate must not contaminate the next derivative.
            lastSeen.reset();
            point.reset();
            pointCov = Covariance{};
            return false;
          }
        }
      }
      point = ground;
      pointCov = cov;
      lastSeen = captureS;
      reason = velocityValid ? "tracked" : "position_only";
      return true;
    }

    BallEstimate estimate(double nowS, const Pose& odometry) const
    {
      BallEstimate result;
      result.ageS = lastSeen ? std::max(0., nowS - *lastSeen) : 40.;
      const bool valid = point && 0. <= nowS - *lastSeen && nowS - *lastSeen <= maxAge;
      Vec2 ground = point ? *point : Vec2();
      if(valid && velocityValid)
        ground = {ground.x + velocity.x * result.ageS, ground.y + velocity.y * result.ageS};
      result.position = toRobot(ground, odometry.translation(), odometry.yaw);
      result.velocity = rotateToRobot(velocity, odometry.yaw);
      result.positionValid = valid;
      result.velocityValid = valid && velocityValid;
      result.visible = valid && (reason == "tracked" || reason == "position_only");
      result.positionCovariance = rotateCovariance(pointCov, -odometry.yaw);
      result.velocityCovariance = rotateCovariance(velocityCov, -odometry.yaw);
      result.reason = reason;
      return result;
    }

    const std::string& getReason() const { return reason; }

  private:
    double maxAge, maxSpeed, maxGap;
    std::optional<double> lastCapture;
    std::optional<double> lastSeen;
    std::optional<Vec2> point;
    Covariance pointCov{};
    Vec2 velocity;
    Covariance velocityCov{};
    bool velocityValid = false;
    std::string reason = "unseen";
  };

  // ---------------------------------------------------------------------------
  // vision_keeper_v2 and the six-defense library (vision_keeper_v6)
  // ---------------------------------------------------------------------------

  struct KeeperOutput
  {
    int blockSide = 0;   /**< +1 keeper's left, -1 right, 0 centre. */
    double trigger = 0.;
    double amplitude = .5;
  };

  struct KeeperDecision
  {
    std::string reason = "waiting_for_ball";
    std::optional<double> crossingY;
    std::optional<double> timeToLine;
    std::optional<int> side;
  };

  struct VisionKeeperSettings
  {
    double cornerY = .3;
    double diveLeadS = 1.8;
    double minIncomingSpeed = .12;
    int confirmFrames = 2;
    int history = 3;
    double maxBallAge = .25;
    double sigmaMargin = 1.5;
    double goalLineX = -.12;

    bool valid() const
    {
      const auto in = [](double value, double low, double high) { return finite(value) && low <= value && value <= high; };
      return in(cornerY, .2, .6) && in(diveLeadS, .3, 3.) && in(minIncomingSpeed, .05, .5) && in(maxBallAge, .01, .5) &&
             in(sigmaMargin, 0., 4.) && in(goalLineX, -.5, 0.) && confirmFrames >= 1 && confirmFrames <= 6 &&
             history >= 1 && history <= 6;
    }
  };

  class VisionKeeper
  {
  public:
    explicit VisionKeeper(VisionKeeperSettings settings = VisionKeeperSettings()) : settings(settings) { reset(); }

    void reset()
    {
      lastCapture.reset();
      lastTime.reset();
      committed.reset();
      incomingFrames = 0;
      trajectories.clear();
      decision = KeeperDecision();
    }

    KeeperOutput wait(const std::string& reason)
    {
      incomingFrames = 0;
      trajectories.clear();
      decision = KeeperDecision();
      decision.reason = reason;
      return KeeperOutput();
    }

    /** One vision_state_v1 step: decision time, capture time, remaining time and the tracked ball. */
    KeeperOutput operator()(double timeS, double captureS, double remainingTime, const BallEstimate& ball)
    {
      if(!finite(timeS) || !finite(captureS) || !finite(remainingTime))
        return wait("invalid_observation");
      if(lastTime && timeS < *lastTime)
        reset();
      lastTime = timeS;
      if(remainingTime <= 0. || timeS >= 28.)
        return wait("deadline");
      if(committed)
        return {*committed, 1., .5};
      if(lastCapture && captureS <= *lastCapture)
        return KeeperOutput();
      lastCapture = captureS;
      if(!ball.positionValid || !ball.visible || !(0. <= ball.ageS && ball.ageS <= settings.maxBallAge))
        return wait("stale_ball");
      if(!ball.velocityValid)
        return wait("unmeasured_velocity");
      const double bx = ball.position.x, by = ball.position.y, vx = ball.velocity.x, vy = ball.velocity.y;
      const double variance = ball.velocityCovariance[0][0];
      if(!finite(bx) || !finite(by) || !finite(vx) || !finite(vy) || !finite(variance) || variance < 0.)
        return wait("invalid_ball");
      const double threshold = settings.minIncomingSpeed + settings.sigmaMargin * std::sqrt(variance);
      if(bx <= 0. || vx >= -threshold)
        return wait("waiting_for_shot");
      ++incomingFrames;
      const double arrivalNow = (settings.goalLineX - bx) / vx;
      trajectories.emplace_back(by + vy * arrivalNow, arrivalNow);
      while(static_cast<int>(trajectories.size()) > settings.history)
        trajectories.pop_front();
      if(incomingFrames < settings.confirmFrames)
      {
        decision = KeeperDecision();
        decision.reason = "confirming_trajectory";
        return KeeperOutput();
      }
      std::vector<double> ys, arrivals;
      for(const auto& [y, arrival] : trajectories)
      {
        ys.push_back(y);
        arrivals.push_back(arrival);
      }
      const double y = median(ys), arrival = median(arrivals);
      const int side = y > settings.cornerY ? 1 : y < -settings.cornerY ? -1 : 0;
      decision = KeeperDecision();
      decision.reason = "waiting_for_arrival";
      decision.crossingY = y;
      decision.timeToLine = arrival;
      decision.side = side;
      if(side != 0 && arrival > settings.diveLeadS)
        return {side, 0., .5};
      committed = side;
      decision.reason = "committed";
      return {side, 1., .5};
    }

    const KeeperDecision& getDecision() const { return decision; }
    bool isCommitted() const { return committed.has_value(); }
    const VisionKeeperSettings& getSettings() const { return settings; }

    static double median(std::vector<double> values)
    {
      std::sort(values.begin(), values.end());
      const std::size_t mid = values.size() / 2;
      return values.size() % 2 ? values[mid] : (values[mid - 1] + values[mid]) / 2.;
    }

  private:
    VisionKeeperSettings settings;
    std::optional<double> lastCapture;
    std::optional<double> lastTime;
    std::optional<int> committed;
    int incomingFrames = 0;
    std::deque<std::pair<double, double>> trajectories;
    KeeperDecision decision;
  };

  /** keeper_defenses_v2 names; each maps to one original B-Human dive request. */
  enum class Defense { center, centerRight, wideLeft, wideRight, left, right };

  inline const char* defenseName(Defense defense)
  {
    switch(defense)
    {
      case Defense::center: return "center";
      case Defense::centerRight: return "center_right";
      case Defense::wideLeft: return "wide_left";
      case Defense::wideRight: return "wide_right";
      case Defense::left: return "left";
      case Defense::right: return "right";
    }
    return "center";
  }

  /** Modes of penalty.keeper_library.VisionLibraryKeeper. */
  enum class LibraryMode { original, centerRight, wideLeft, wideRight, balanced };

  struct LibraryOutput
  {
    Defense defense = Defense::center;
    double trigger = 0.;

    bool fires() const { return trigger >= .5; }
  };

  class LibraryKeeper
  {
  public:
    explicit LibraryKeeper(LibraryMode mode = LibraryMode::balanced, VisionKeeperSettings settings = VisionKeeperSettings()) :
      mode(mode), core(settings) {}

    void reset()
    {
      core.reset();
      committedDefense.reset();
    }

    LibraryOutput operator()(double timeS, double captureS, double remainingTime, const BallEstimate& ball)
    {
      const KeeperOutput action = core(timeS, captureS, remainingTime, ball);
      if(!core.isCommitted())
        committedDefense.reset();
      Defense defense = select(action.blockSide, core.getDecision().crossingY);
      if(action.trigger >= .5)
      {
        if(!committedDefense)
          committedDefense = defense;
        defense = *committedDefense;
      }
      return {defense, action.trigger};
    }

    const KeeperDecision& getDecision() const { return core.getDecision(); }
    LibraryMode getMode() const { return mode; }

  private:
    Defense select(int side, const std::optional<double>& crossing) const
    {
      if(side)
        return side > 0 ? Defense::left : Defense::right;
      switch(mode)
      {
        case LibraryMode::balanced: return crossing && *crossing < 0. ? Defense::centerRight : Defense::center;
        case LibraryMode::original: return Defense::center;
        case LibraryMode::centerRight: return Defense::centerRight;
        case LibraryMode::wideLeft: return Defense::wideLeft;
        case LibraryMode::wideRight: return Defense::wideRight;
      }
      return Defense::center;
    }

    LibraryMode mode;
    VisionKeeper core;
    std::optional<Defense> committedDefense;
  };
}
