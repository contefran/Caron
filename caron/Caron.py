# imports 
import argparse
import threading
import time
from caron.data import Data
from caron.simulation import Simulation
from caron.visualization import Visualization

class Main:
    def __init__(self):
        self.args = self.parse_args()
        self.data = Data(buffer_safe_min=self.args.calib_frames, buffer_safe_max=500)
        self.sim = Simulation(
            data=self.data,
            args=self.args,
        )
        self.viz = Visualization(
            data=self.data,
            args=self.args,
        )


    @staticmethod
    def parse_args():
        parser = argparse.ArgumentParser(prog="Caron")
        parser.add_argument("--sim_size",type=int,default=512,help="Linear size of the simulation grid [Default: 512]")
        parser.add_argument("--n_frames",type=int,default=200,help="Number of simulation frames [Default: 200]")
        parser.add_argument("--viz_fps",type=float,default=100,help="Initial visualisation FPS")
        parser.add_argument("--calib_time",type=float,default=3,help="FPS calibration time in seconds [Default: 3s]")
        parser.add_argument("--calib_frames",type=int,default=50,help="Minimum number of frames for FPS calibration [Default: 50 frames]")
        parser.add_argument("--sim_file",type=str,default="../mock_sim.npy",help="Mock simulation file [Default: mock_sim.npy]")
        parser.add_argument("--no_sim",action="store_true",help="Disable simulation (visualise mock simulation only)")
        parser.add_argument("--no_viz",action="store_true",help="Disable visualisation (run simulation only)")
        parser.add_argument("--fake_injection",action="store_true",help="Inject the mock simulation at a fixed fps (does not work when --no_sim is invoked)")
        parser.add_argument("--fake_sim_fps",type=int,default=60,help="Fake simulation injection fps, when --fake_injection is invoked [Default: 60Hz]")
        
        return parser.parse_args()


    def run(self):
        if self.args.no_sim and self.args.no_viz:
            print("Nothing to do: both --no_sim and --no_viz are set.")
            return

        #if self.args.no_viz:
        #    self.run_sim_only()

        # Phase 1: easy easy, just run one after the other
        #if not self.args.no_viz:
        #    self.viz.run() # visualises self.data frames
        #if not self.args.no_sim:
        #    self.sim.run() # fills self.data with simulation frames

        # Phase 2: replace with self.data class buffering
        if self.args.no_sim:
            self.viz.run() # visualises self.data frames
        else:
            if self.args.fake_injection:
                print("[Main] Running mock simulation injection + visualization.")
                sim_thread = threading.Thread(target=self.sim.run_mock, daemon=True)
                sim_thread.start()
                # Wait until buffer reaches safe_min before starting viz
                while len(self.data.buffer) < self.data.buffer_safe_min:
                    time.sleep(0.01)
                print(f"[Main] Buffer reached size {len(self.data)}: starting visualization.")

                self.viz.run() # now stat the visualization
            else:
                raise NotImplementedError("[Main] Main.run: real simulation + visualization not implemented yet.")


if __name__ == "__main__":
    Main().run()
    