// Platform host for the original subsystem. No motion computation lives here.
#include "Platform/SystemCall.h"
#include "Framework/Settings.h"
#include "Streaming/Global.h"
#include "Platform/File.h"
#include <iostream>

SystemCall::Mode SystemCall::getMode() { return simulatedRobot; }
int SystemCall::playSound(const char* name, bool) { std::cerr << "B-Human sound: " << name << '\n'; return 0; }
int SystemCall::say(const char* text, bool, float) { std::cerr << "B-Human: " << text << '\n'; return 0; }
bool SystemCall::soundIsPlaying() { return false; }

// Settings' explicit constructor copied verbatim in effect: bypassing its
// unrelated log-file parser removes the full logger's link dependencies.
Settings::Settings(const std::string& headName, const std::string& bodyName, int teamNumber,
                   TeamColor fieldPlayerColor, TeamColor goalkeeperColor, int playerNumber,
                   const std::string& location, const std::string& scenario, unsigned char magicNumber) :
  headName(headName), bodyName(bodyName), teamNumber(teamNumber), fieldPlayerColor(fieldPlayerColor),
  goalkeeperColor(goalkeeperColor), playerNumber(playerNumber), location(location), scenario(scenario), magicNumber(magicNumber) {}
std::list<std::string> Settings::getSearchPath() const
{
  const std::string base = std::string(BHUMAN_BASELINE_DIR) + "/Config/";
  return {base + "Robots/Nao/Nao/", base + "Robots/Default/", base + "Scenarios/Default/", base + "Locations/Default/"};
}
