import pyautogui, sys


def control_mouse(x_disp, y_disp):
    """

    :param y_disp: y displacement
    :param x_disp: x displacement
    """

    current_x, current_y = pyautogui.position()

    pyautogui.moveTo(current_x, current_y + (y_disp) * 100)
