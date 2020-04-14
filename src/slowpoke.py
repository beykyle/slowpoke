__author__ = "Kyle Beyer"
__email__ = "beykyle@umich.edu"


import process_data
import glob
import sys
import os
import argparse
import numpy as np
import xml.etree.ElementTree as et
from pathlib import Path

"""
This module performs numerical neutron slowing down calculations for homogenous mixtures
"""

from nuclide import Nuclide
from process_data import Reactions as RXN

class Grid:
    def __init__(self, gmax, gmin, sz):
        self.max = gmax
        self.min = gmin
        self.size = sz

class Simulation:
    def __init__(self, nuclides : list, number_densities: list):
        self.nuclides = nuclides
        self.number_densities = number_densities

class Homogenized2SpeciesSimulation():
    def __init__(self, moderator, absorber, ratios):
        self.moderator = moderator
        self.absorber = absorber
        self.ratios = ratios
        self.problems = len(ratios)
        self.fluxes = []

    def get_problem_data(self, idx: int):
        ratio = self.ratios[idx]
        return Simulation([self.absorber, self.moderator] , [1. , ratio])

    def append_result(self, flux):
        self.fluxes.append(flux)


def validate_xml_path(input_path):
    if not input_path.is_file():
        print("Input path must point to a valid xml file, completely specifying a material")
        exit(1)

def build_nuclide_data(xml_root, input_path, grid):
    nuclear_data_node = xml_root.find("nuclear_data")
    nuclides = []
    base_path = input_path.parents[1]
    print("Base data path: " + str(base_path))
    for nuclide_node in nuclear_data_node.findall('nuclide'):
        nuclides.append(Nuclide(nuclide_node, base_path, grid))

    return nuclides

def build_material(root, nuclides):
    mat_node = root.find("material")
    sim_type = mat_node.get("type")

    if sim_type != "homogenized_moderator_absorber":
        print("Unkown simulation type " + sim_type)
        exit(1)

    # get nuclides
    mod_name = mat_node.get("moderator")
    abs_name = mat_node.get("absorber")
    moderator = [nuclide for nuclide in nuclides if nuclide.name == mod_name][0]
    absorber  = [nuclide for nuclide in nuclides if nuclide.name == abs_name][0]

    # get number density ratios
    mod_abs_ratio_str = mat_node.get("mod_to_abs_ratio")
    mod_abs_ratio = []
    if (mod_abs_ratio_str != None):
        mod_abs_ratio = [float(mod_abs_ratio_str)]
    else:
        ratio_node = mat_node.find("mod_to_abs_ratio")
        rmin = float(ratio_node.get("min"))
        rmax = float(ratio_node.get("max"))
        rnum = int(ratio_node.get("points"))
        mod_abs_ratio = np.linspace(rmin, rmax, rnum).tolist()

    return Homogenized2SpeciesSimulation(moderator, absorber, mod_abs_ratio)


def solver(sig_s, sig_t , alpha, u):
    return np.zeros(len(sig_s[0]))

def slow_down(simulation):
    # calculate all cross sections and lethargy grid
    nucs = simulation.nuclides
    nums_dens = simulation.number_densities

    alpha = [nuc.alpha for nuc in nucs]
    sig_s = [ N * nuc.xs[RXN.elastic_sc].xs for N , nuc in zip(nums_dens, nucs)]
    sig_a = [ N * nuc.xs[RXN.rad_cap].xs    for N , nuc in zip(nums_dens, nucs)]
    sig_t = sig_s + sig_a
    u = nucs[0].xs[RXN.elastic_sc].leth_boundaries

    return(solver(sig_s, sig_t, alpha , u))


def run_problem(material):
    for i in range(material.problems):
        simulation = material.get_problem_data(i)
        flux = slow_down(simulation)
        material.append_result(flux)


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
    parser.add_argument('-o', '--ouput',
            help='Path to write output file to',
                        dest='output_path', required=False)
    parser.add_argument('--max-energy', type=float,
                        help='maximum energy in [eV] for slowing down equations - for defining lethargy. Defalut: 2E4',
            dest='max_energy_eV', default=def_max_energy_eV)
    parser.add_argument('--min-energy', type=float,
                        help='minimum energy in [eV] for slowing down equations - for defining lethargy. Default: 1.0',
            dest='min_energy_eV', default=def_min_energy_eV)
    parser.add_argument('-n', '--num-gridpoints', type=int,
                        help='desired number of points on lethargy grid: Default: 6E5',
            dest='gridsize', default=def_gridsize)
    parser.add_argument('-d', '--display', action='store_true',
                        help='if flag present, generates flux plots', dest='display')
    args = parser.parse_args()


    input_path = Path(args.input_path)
    validate_xml_path(input_path)
    grid = Grid(args.max_energy_eV , args.min_energy_eV, args.gridsize)

    tree = et.parse(str(input_path))
    root = tree.getroot()
    nuclides = build_nuclide_data(root, input_path, grid)
    material = build_material(root, nuclides)
    run_problem(material)


if __name__ == "__main__":
    parse_args_and_run(sys.argv)
