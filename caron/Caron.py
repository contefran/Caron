# imports 
import argparse
from caron.data import Data
from caron.simulation import Simulation
from caron.visualization import Visualization

class Main:
    def __init__(self):
        self.args = self.parse_args()
        self.data = Data()
        self.sim = Simulation(
            data=self.data,
            n_frames=self.args.n_frames,
            size=self.args.size,
        )
        self.viz = Visualization(
            data=self.data,
            args=self.args,
        )



    @staticmethod
    def parse_args():
        parser = argparse.ArgumentParser(prog="Caron")
        parser.add_argument("--size",type=int,default=512,help="Linear size of the simulation grid [Default: 512]")
        parser.add_argument("--n_frames",type=int,default=200,help="Number of simulation frames [Default: 200]")
        parser.add_argument("--viz_fps",type=float,default=100,help="Initial visualisation FPS")
        parser.add_argument("--calib_time",type=float,default=3,help="FPS calibration time in seconds [Default: 3s]")
        parser.add_argument("--calib_frames",type=int,default=50,help="Minimum number of frames for FPS calibration [Default: 50 frames]")
        parser.add_argument("--sim_file",type=str,default="./mock_sim.npy",help="Mock simulation file [Default: mock_sim.npy]")
        parser.add_argument("--no_sim",action="store_true",help="Disable simulation (visualise mock simulation only)")
        parser.add_argument("--no_viz",action="store_true",help="Disable visualisation (run simulation only)")
        return parser.parse_args()


    def run(self):
        if self.args.no_sim and self.args.no_viz:
            print("Nothing to do: both --no_sim and --no_viz are set.")
            return

        #if self.args.no_viz:
        #    self.run_sim_only()

        # Phase 1: easy easy, just run one after the other
        if not self.args.no_sim:
            self.sim.run() # fills self.data with simulation frames
        if not self.args.no_viz:
            self.viz.run() # visualises self.data frames
        # Phase 2: replace with self.data class buffering


    #def run_sim_only(self):
    #    self.sim.run_no_viz() # need to define what it does exactly. At this stage, it just fills the data buffer.


if __name__ == "__main__":
    Main().run()
    