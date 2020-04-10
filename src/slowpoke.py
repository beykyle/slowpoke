__author__ = "Kyle Beyer"
__email__ = "beykyle@umich.edu"


import process_data
import glob
import sys
import os
import argparse
import xml.etree.ElementTree as et
from pathlib import Path

"""
This module performs numerical neutron slowing down calculations for homogenous mixtures
"""

from nuclide import Nuclide

class Grid:
    def __init__(self, gmax, gmin, sz):
        self.max = gmax
        self.min = gmin
        self.size = sz

def validate_xml_path(input_path):
    if not input_path.is_file():
        print("Input path must point to a valid xml file. Example format:")
        print("<nuclear_data>")
        print("  <nuclide name=\"H1\"    ZAID=\"1001\"   mass_ratio_n=\"0.9992\"  potential_scat_b=\"20.478\" xs_path=\"data/H1.csv\" />")
        print("</nuclear_data>")
        exit(1)

def build_nuclide_data(input_path, grid):
    tree = et.parse(str(input_path))
    root = tree.getroot()
    nuclides = []
    base_path = input_path.parents[0]
    print("Base data path: " + str(base_path))
    for nuclide_node in root.findall('nuclide'):
        nuclides.append(Nuclide(nuclide_node, base_path, grid))

    return nuclides


def parse_args_and_run(argv: list):

    # default args
    current_path = Path(os.getcwd())
    def_out_path = None
    def_max_energy_eV = 20000.0
    def_min_energy_eV = 1.0
    def_gridsize = 600000

    # argument parsing
    parser = argparse.ArgumentParser(
            description='Interpolate pointwise microscopic cross sections to equal lethargy groups')
    parser.add_argument('-i', '--input',
            help='Path to xml file describing maerial composition of system',
                        dest='input_path', required=True)
    parser.add_argument('--max-energy', type=float,
                        help='maximum energy in [eV] for slowing down equations - for defining lethargy. Defalut: 2E4',
            dest='max_energy_eV', default=def_max_energy_eV)
    parser.add_argument('--min-energy', type=float,
                        help='minimum energy in [eV] for slowing down equations - for defining lethargy. Default: 1.0',
            dest='min_energy_eV', default=def_min_energy_eV)
    parser.add_argument('-n', '--num-gridpoints', type=int,
                        help='desired number of points on lethargy grid: Default: 6E5',
            dest='gridsize', default=def_gridsize)
    args = parser.parse_args()

    input_path = Path(args.input_path)
    validate_xml_path(input_path)
    grid = Grid(args.max_energy_eV , args.min_energy_eV, args.gridsize)

    nuclides = build_nuclide_data(input_path, grid)


if __name__ == "__main__":
    parse_args_and_run(sys.argv)
