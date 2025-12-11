# imports 
import argparse
from data import Data
from simulation import Simulation
from visualization import Visualization

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
            fps=self.args.fps,
        )


    @staticmethod
    def parse_args():
        parser = argparse.ArgumentParser(prog="Caron")
        parser.add_argument("--size",type=int,default=128,help="Linear size of the simulation grid")
        parser.add_argument("--n_frames",type=int,default=200,help="Number of simulation frames")
        parser.add_argument("--viz_fps",type=float,default=100,help="Initial visualisation FPS")
        parser.add_argument("--no_sim",action="store_true",help="Disable simulation (visualise existing buffer only)")
        parser.add_argument("--no_viz",action="store_true",help="Disable visualisation (run simulation only)")
        return parser.parse_args()


    def run(self):
        if self.args.no_sim and self.args.no_viz:
            print("Nothing to do: both --no_sim and --no_viz are set.")
            return

        if self.args.no_viz:
            self.run_sim_only()
        elif self.args.no_sim:
            self.run_viz_only()
        else:
            self.run_all()


    def run_sim_only(self):
        self.sim.run_no_viz() # need to define what it does exactly


    def run_viz_only(self):
        self.viz.run_no_sim() # I guess self.data would use the existing simulation as a buffer, formatted as a list of frames


    def run_all(self):
        # Phase 1: easy easy, just run one after the other
        self.sim.run() # fills self.data with simulation frames
        self.viz.run() # visualises self.data frames
        # Phase 2: replace with self.data class buffering


if __name__ == "__main__":
    Main().run()
    