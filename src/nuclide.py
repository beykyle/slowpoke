__author__ = "Kyle Beyer"
__email__ = "beykyle@umich.edu"


import xml.etree.ElementTree as et
import process_data

"""
This module reads and stores nuclide data from an xml file
"""

nuc_directory = {
    "h1"   : 1001,
    "O16"  : 8016,
    "U238" : 92238
}

class Nuclide:
    def __init__(self, name : str , fpath):
        tree = et.parse(fpath)
        root = tree.getroot()
        node = root.find(name)
        self.ZAID = float(node.get("ZAID"))
        self.mass_n = float(node.get("mass_ratio_n"))
        self.pot_scatterxs_b  = float(node.get("potential_scat_b"))
        self.xs = dict()

    def set_xs(rxn : str , xs : CrossSection):
        self.xs[rxn] = xs
