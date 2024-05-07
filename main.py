from CNNModel import CNNModel
from Camera import Camera
from Keyboard import Keyboard

cnn_model = CNNModel("efficient_net_v2_b0_alphabet.h5")
cnn_model.set_classes(["A", "B", "C", "D" , "del", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "space", "T", "U", "V", "W", "X", "Y", "Z"])

camera = Camera()
keyboard = Keyboard()

camera.set_keyboard(keyboard)
camera.start_camera(cnn_model)
