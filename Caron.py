import numpy as np
from matplotlib import cm  # Import colormap functionality
import time
import dearpygui.dearpygui as dpg

frames = np.load('Simulation.npy')

running = False
speed = 10  # seconds per frame
last_update_time = time.time()
frame_index = 0

frame = frames[0]
frame_min = np.min(frame)
frame_max = np.max(frame)
frame_norm = (frame - frame_min) / (frame_max - frame_min)
frame_rgb = np.stack((frame_norm,) * 3, axis=-1)
frame_rgb = frame_rgb.astype(np.float32) / np.max(frame_rgb)
frame_flattened = frame_rgb.flatten()

dpg.create_context()
#with dpg.font_registry():
#    big_font = dpg.add_font("C:/Windows/Fonts/BASKVILL.TTF", 20, tag="big_font")
with dpg.font_registry():
    big_font = dpg.add_font("C:/Windows/Fonts/BKANT.TTF", 20, tag="big_font")
with dpg.texture_registry(show=True):
    dpg.add_raw_texture(1000, 1000, default_value=frame_flattened, format=dpg.mvFormat_Float_rgb, tag="frame_tag")

def start_callback():
    global running
    running = True

def stop_callback():
    global running
    running = False

def speed_callback(sender, app_data):
    global speed
    speed = app_data

def update_frame():
    global frame_index, last_update_time, running
    current_time = time.time()
    if (running and (current_time - last_update_time) >= 1 / speed) or frame_index == 0:
        last_update_time = current_time
        frame = frames[frame_index]
        frame_min = np.min(frame)
        frame_max = np.max(frame)
        frame_norm = (frame - frame_min) / (frame_max - frame_min)
        frame_norm = np.clip(frame_norm, 0, 1)
        colormap = cm.inferno
        frame_colored = colormap(frame_norm)  # Returns an RGBA array
        frame_rgb = (frame_colored[..., :3] * 255).astype(np.uint8)
        frame_flattened = frame_rgb.flatten().astype(np.float32) / 255.0
        dpg.set_value("frame_tag", frame_flattened)
        frame_index = (frame_index + 1) % len(frames)
    with dpg.mutex():
        target_frame = dpg.get_frame_count() + 2
        dpg.set_frame_callback(target_frame, update_frame)

with dpg.window(label="Jet Inspector", width=1100, height=1000, no_close=True, no_move=True, no_resize=False):
    dpg.bind_font("big_font")
    with dpg.group(label="Visualizator", horizontal=True):
        with dpg.group(label="Map and slider"):
            dpg.add_slider_int(label="Speed (FPS)", height=40, default_value=10, min_value=1, max_value=20, callback=speed_callback)
            dpg.add_image("frame_tag")
        with dpg.group(label="Start&Stop"):
            dpg.add_button(label="Start", callback=start_callback, width=80, height=200)
            dpg.add_button(label="Stop", callback=stop_callback, width=80, height=200)

dpg.create_viewport(title="Our lovely Caron", width=1100, height=1000)

dpg.setup_dearpygui()

# Start updating frames
update_frame()
    
dpg.show_viewport()
dpg.start_dearpygui()
dpg.destroy_context()

