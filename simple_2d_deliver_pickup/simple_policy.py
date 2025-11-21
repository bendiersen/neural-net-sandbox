from world import Action

def simple_policy(state):
    ax, ay, sx, sy, dx, dy, has_star = state

    if not has_star:
        # Move toward star
        if ax < sx: return Action.RIGHT
        if ax > sx: return Action.LEFT
        if ay < sy: return Action.DOWN
        if ay > sy: return Action.UP
        return Action.PICK_UP
    else:
        # Move toward drop zone
        if ax < dx: return Action.RIGHT
        if ax > dx: return Action.LEFT
        if ay < dy: return Action.DOWN
        if ay > dy: return Action.UP
        return Action.DROP