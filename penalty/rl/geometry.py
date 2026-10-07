"""Field geometry of `sim/scene.xml`, robot-relative transforms and the
geometric goal-crossing predictor used by the keeper baseline."""
import math

GOAL_LINE_X = -1.2
POST_Y = 0.75
BALL_RADIUS = 0.0325
# Ball centre must stay inside the posts, as in PhysicalSimulator._events.
GOAL_HALF_WIDTH = POST_Y - BALL_RADIUS
GOAL_CENTER = (GOAL_LINE_X, 0.)
# Posts named as seen by a striker facing the goal (towards -x): its left
# is world -y. The keeper sees the same physical posts with the same names.
LEFT_POST = (GOAL_LINE_X, -POST_Y)
RIGHT_POST = (GOAL_LINE_X, POST_Y)
PENALTY_MARK = (0.1, 0.)
STRIKER_START = ((0.3, 0.), math.pi)
KEEPER_START = ((-1.08, 0.), 0.)
PENALTY_SECONDS = 28.


def wrap(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


def to_robot(point, origin, yaw):
    """World point -> robot frame (x forward, y left)."""
    dx, dy = point[0]-origin[0], point[1]-origin[1]
    c, s = math.cos(yaw), math.sin(yaw)
    return c*dx + s*dy, -s*dx + c*dy


def rotate_to_robot(vector, yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return c*vector[0] + s*vector[1], -s*vector[0] + c*vector[1]


def bearing(point, origin, yaw):
    x, y = to_robot(point, origin, yaw)
    return math.atan2(y, x)


def predict_goal_crossing(ball_position, ball_velocity, goal_line_x=0., deceleration=0.,
                          min_speed=0.05):
    """Constant-deceleration ball model in the keeper frame (x forward).

    `goal_line_x` is the line to defend in the same frame (0 = keeper line).
    Returns {'time_to_goal', 'predicted_goal_y'} or None when the ball does
    not approach the line or stops before it.
    """
    x, y = ball_position
    vx, vy = ball_velocity
    if not all(math.isfinite(v) for v in (x, y, vx, vy)):
        return None
    speed = math.hypot(vx, vy)
    if speed < min_speed or vx >= -1e-6 or x <= goal_line_x:
        return None
    # Path length along the velocity direction until the line.
    path = (goal_line_x - x) / vx * speed
    if deceleration > 0:
        disc = speed*speed - 2*deceleration*path
        if disc < 0:
            return None
        time = (speed - math.sqrt(disc)) / deceleration
    else:
        time = path / speed
    return {'time_to_goal': time, 'predicted_goal_y': y + vy/vx*(goal_line_x - x)}
