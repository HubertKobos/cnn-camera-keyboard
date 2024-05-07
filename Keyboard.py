import pyautogui

class Keyboard:
    def pressKey(self, key: str):
        # press this guy only if after 1 second we have 90% prediction for a key from the model
        pyautogui.press(key)