from pyglet.window import key

KEYBINDINGS = {
    key.W: 'translate_up',
    key.S: 'translate_down',
    key.A: 'translate_left',
    key.D: 'translate_right',
    key.Z: 'translate_forward',
    key.X: 'translate_backward',
    key.R: 'reset',
    key.LEFT: 'rotate_minus_z',
    key.RIGHT: 'rotate_plus_z',
    key.DOWN: 'rotate_minus_y',
    key.UP: 'rotate_plus_y',
    key.SLASH: 'rotate_minus_x',
    key.PERIOD: 'rotate_plus_x',
    key.MINUS: 'scale_down',
    key.EQUAL: 'scale_up',
    key.NUM_1: 'step_size1',
    key.NUM_2: 'step_size2',
    key.NUM_3: 'step_size3',
    key.PAGEDOWN: 'next_image',
    key.PAGEUP: 'previous_image',
}


OVERLAY_OPACITY = 0.5
TRANSLATION_STEPS = [0.1, 0.01, 0.001]
TRANSLATION_Z_STEPS = [0.1, 0.01, 0.001]
ROTATION_STEPS = [10, 2, 0.1]
SCALE_STEPS = [0.1, 0.05, 0.01]

# seconds
ACTION_DELAY = 10 / 1000
RENDER_RATE = 0.2