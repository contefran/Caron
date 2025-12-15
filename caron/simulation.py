"""Simulation class producing data for the visualization."""
from caron.data import Data
import numpy as np
import time

class Simulation:
    """Produces 2D frames and pushes them into the Data buffer."""


    def __init__(self, data: Data, args, **kwargs) -> None:
        self.args = args
        self.data = data
        self.kwargs = kwargs
        self.sim_size: int = self.args.sim_size


    def run(self) -> None:
        raise NotImplementedError("[Sim] Simulation.run is not implemented yet.")


    def run_mock(self) -> None:
        """Feed frames into the buffer at a fixed rate (sim_fps)."""
        self.sim_fps=self.args.fake_sim_fps #Hz mock simulation fps
        self.sim_file = self.args.sim_file
        self.sim=np.load(self.sim_file)
        self.sim_size: int = self.sim.shape[1]
        print(f"[Sim] Loaded mock simulation from {self.sim_file} with {self.sim.shape[2]} frames of linear size {self.sim[0].shape[1]}.")
        print(f"[Sim] Simulation initialized in fake_injection mode. Injecting the buffer at {self.sim_fps} FPS.")

        dt = 1.0 / self.sim_fps # seconds per frame of the mock simulation injection
        t_next = time.perf_counter() +dt # time of the next frame to push (cumulative)

        start_push_time = None # set at first push completion
        last_push_time = 0.0 # last push completion time
        pushed = 0 # number of pushed frames, for the average fps computation

        # reporting
        report_every_s = 1.0
        last_report_t = time.perf_counter()
        last_report_pushed = 0

        for frame in self.sim:
            now = time.perf_counter() # Wait until next frame time
            if now < t_next:
                time.sleep(t_next - now) # is this appropriate? Maybe we should take a statistical approach as in the visualizer? To be checked

            self.data.push_frame(frame)
 
            t_push = time.perf_counter() # time of push completion
            if start_push_time is None:
                dt_push = 0.0
                start_push_time = t_push
                last_push_time = t_push
            else:
                dt_push = t_push - last_push_time
                last_push_time = t_push
            if pushed % 50 == 0 and dt_push>0:
                print(f"[Sim] dt_push={dt_push:.4f}s ⇒ fps≈{1.0/dt_push:.1f}")
            pushed += 1

            t_next += dt # advance to next frame time

            # apply the cumulative drift correction: if we're behind, catch up
            now2 = time.perf_counter()
            if now2 > t_next + dt:
                t_next = now2 + dt

            # planned reporting (no printing every frame)
            if (t_push - last_report_t) >= report_every_s and start_push_time is not None: # report every report_every_s seconds
                inst_fps = (pushed - last_report_pushed) / (t_push - last_report_t) # instantaneous fps over the last interval
                avg_fps = pushed / (t_push - start_push_time) # average fps since the start
                print(f"[Sim] pushed={pushed}/{self.sim.shape[0]} | buffer={len(self.data)} | inst≈{inst_fps:.2f} Hz | avg≈{avg_fps:.2f} Hz")
                last_report_t = t_push
                last_report_pushed = pushed

        if start_push_time is not None:
            total_time = time.perf_counter() - start_push_time
            avg_fps = pushed / total_time if total_time > 0 else float("inf")
            print(f"[Sim] Finished. Pushed {pushed} frames in {total_time:.3f}s ⇒ avg≈{avg_fps:.2f} Hz")
        else:
            print("[Sim] Finished. No frames were pushed.")
                #print(f"[Sim] added frame {i+1}/{len(self.sim)} in {time.perf_counter()-now:.4f}s | buffer size={len(self.data)}")

        print("[Sim] Simulation finished pushing frames.")
