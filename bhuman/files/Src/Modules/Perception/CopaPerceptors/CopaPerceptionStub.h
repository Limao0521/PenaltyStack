/**
 * @file CopaPerceptionStub.h
 *
 * Empty ("nothing seen") percepts for the Copa penalty scenarios. The HSL field
 * perception (lines, center circle, intersections, penalty mark, scan line
 * regions, ball spots) and the lower camera's obstacle networks (BOP,
 * PlayersDeeptector) are not used by the penalty but took most of the CPU
 * (robot said "CPU temperature", 2026-10-30). Modules that still require these
 * representations (self locator, obstacle model, streamer) get empty ones, so
 * no requirement is left without a provider (that would stop B-Human).
 */

#pragma once

#include "Framework/Module.h"
#include "Representations/Perception/BallPercepts/BallSpots.h"
#include "Representations/Perception/FieldPercepts/CirclePercept.h"
#include "Representations/Perception/FieldPercepts/FieldLineIntersections.h"
#include "Representations/Perception/FieldPercepts/FieldLines.h"
#include "Representations/Perception/FieldPercepts/IntersectionsPercept.h"
#include "Representations/Perception/FieldPercepts/LinesPercept.h"
#include "Representations/Perception/FieldPercepts/PenaltyMarkPercept.h"
#include "Representations/Perception/ImagePreprocessing/ColorScanLineRegions.h"
#include "Representations/Perception/ImagePreprocessing/FieldBoundary.h"
#include "Representations/Perception/ImagePreprocessing/ImageRegions.h"
#include "Representations/Perception/ImagePreprocessing/SegmentedObstacleImage.h"
#include "Representations/Perception/ObstaclesPercepts/ObstaclesFieldPercept.h"
#include "Representations/Perception/ObstaclesPercepts/ObstaclesImagePercept.h"
#include "Representations/Perception/ObstaclesPercepts/ObstaclesPerceptorData.h"

MODULE(CopaPerceptionStub,
{,
  PROVIDES(BallSpots),
  PROVIDES(CirclePercept),
  PROVIDES(ColorScanLineRegionsHorizontal),
  PROVIDES(ColorScanLineRegionsVerticalClipped),
  PROVIDES(FieldBoundary),
  PROVIDES(FieldLineIntersections),
  PROVIDES(FieldLines),
  PROVIDES(IntersectionsPercept),
  PROVIDES(LinesPercept),
  PROVIDES(ObstaclesFieldPercept),
  PROVIDES(ObstaclesImagePercept),
  PROVIDES(ObstaclesPerceptorData),
  PROVIDES(PenaltyMarkPercept),
  PROVIDES(PenaltyMarkRegions),
  PROVIDES(SegmentedObstacleImage),
});

class CopaPerceptionStub : public CopaPerceptionStubBase
{
  void update(BallSpots& spots) override { spots.ballSpots.clear(); spots.firstSpotIsPredicted = false; }
  void update(CirclePercept& percept) override { percept.wasSeen = false; }
  void update(ColorScanLineRegionsHorizontal& regions) override { regions = ColorScanLineRegionsHorizontal(); }
  void update(ColorScanLineRegionsVerticalClipped& regions) override { regions = ColorScanLineRegionsVerticalClipped(); }
  void update(FieldBoundary& boundary) override { boundary = FieldBoundary(); }
  void update(FieldLineIntersections& intersections) override { intersections.intersections.clear(); }
  void update(FieldLines& lines) override { lines.lines.clear(); }
  void update(IntersectionsPercept& percept) override { percept.intersections.clear(); }
  void update(LinesPercept& percept) override { percept.lines.clear(); }
  void update(ObstaclesFieldPercept& percept) override { percept.obstacles.clear(); }
  void update(ObstaclesImagePercept& percept) override { percept.obstacles.clear(); }
  void update(ObstaclesPerceptorData& data) override { data = ObstaclesPerceptorData(); }
  void update(PenaltyMarkPercept& percept) override { percept.wasSeen = false; }
  void update(PenaltyMarkRegions& regions) override { regions = PenaltyMarkRegions(); }
  void update(SegmentedObstacleImage&) override {}
};
