__author__ = "Kyle Beyer"
__email__ = "beykyle@umich.edu"

from pathlib import Path
import xml.etree.ElementTree as et

"""
This module reads and stores nuclide data from an xml file
"""

from process_data import CrossSection
from process_data import build_xs_directory

nuc_directory = {
    "h1"   : 1001,
    "O16"  : 8016,
    "U238" : 92238
}

def validate_xs_path(xs_path):
    if not xs_path.is_file():
        print("xs_path must point to a JANIS style .csv file for microscopic xs")
        exit(1)

class Nuclide:
    def __init__(self, node, base_path, grid):
        # read basic info
        self.name = node.get("name")
        self.ZAID = float(node.get("ZAID"))
        self.mass_n = float(node.get("mass_ratio_n"))

        # pre-calculate values used in slowing down
        self.alpha = ((self.mass_n - 1) / (self.mass_n + 1))**2
        self.epsilon = np.log(1/self.alpha)

        # read and interpolate xs data to grid
        self.pot_scatterxs_b  = float(node.get("potential_scat_b"))
        relative_xs_path = Path(node.get("xs_path"))
        data_path = base_path / relative_xs_path
        print("Reading cross section data for " + self.name + " at " + str(data_path.absolute()))
        validate_xs_path(data_path)
        self.xs = build_xs_directory(data_path , self.name, grid.max , grid.min, grid.size)
