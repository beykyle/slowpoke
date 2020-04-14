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
from enum import Enum

"""
This module performs numerical neutron slowing down calculations for homogenous mixtures
"""

from nuclide import Nuclide
from process_data import Reactions as RXN

class BoundaryCondition(Enum):
    asymptotic_scatter_source = 1

class Grid:
    def __init__(self, gmax, gmin, sz):
        self.max = gmax
        self.min = gmin
        self.size = sz

class Simulation:
    def __init__(self, nuclides : list, number_densities: list):
        self.nuclides = nuclides
        self.number_densities = number_densities

class Homogenized2SpeciesMaterial():
    def __init__(self, moderator, absorber, ratios):
        self.moderator = moderator
        self.absorber = absorber
        self.ratios = ratios
        self.problems = len(ratios)
        self.fluxes = []
        key = next(iter(moderator.xs))
        self.egrid = self.moderator.xs[key].E

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

    return Homogenized2SpeciesMaterial(moderator, absorber, mod_abs_ratio)


def solver(sig_s : list, sig_t :list , alpha : list , u , phi1 : float):
    """
    Numerical slowing down solver for arbirary number of nuclides and arbitrary lethargy grid
    @param sig_s  macroscopic scatter xs: list of np arrays, 1 for each nuclide. Size of each array = num groups
    @param sig_t  macroscopic total xs: list of np arrays, 1 for each nuclide. Size of each array = num groups
    @param alpha  max fractional energy loss from elastic scatter: list of floats, 1 for each nuclide
    @ param u     lethargy grid: np array, size = size of cross section arrays + 1
    @ phi1        flux in first lethargy group
    """
    p = np.zeros(len(u) - 1)
    p[0] = phi1
    for i in range(1,len(p)):
        du = u[i] - u[i-1]
        numerator = 0.0
        # calculate in-scatter contribution for each nuclide
        for s , a in zip(sig_s, alpha):
            # calculate lowest group that can scatter into current group for current nuclide
            min_inscatter_group = int(round(i - np.log(1/a) / du))
            n = min_inscatter_group if min_inscatter_group > 0 else 0

            # increment numerator by the in-scatter contribution from each group
            # from n to i - 1
            for l in range(n, i-1):
                numerator = numerator + s[l] * p[l] * (np.exp(u[l]) - np.exp(u[l-1])) * (np.exp(- u[i-1]) - np.exp(- u[i]) )

        # calculate denominator
        # sum group i in scattering cross section over nuclides
        sum_sig_s = sum( [s[i]/(1 - alpha[j]) for j, s in enumerate(sig_s)] )
        # sum group i total cross section over nuclides
        sum_sig_t = sum( [s[i] for s in sig_t] )
        # calculate flux in currrent group
        p[i] = numerator / ( (du) * (sum_sig_t) - (sum_sig_s) * (du - 1 + np.exp(-du)) )


def slow_down(simulation, boundary=BoundaryCondition.asymptotic_scatter_source):
    # get problem data
    nucs = simulation.nuclides
    nums_dens = simulation.number_densities


    # calculate macroscopic xs
    alpha = [nuc.alpha for nuc in nucs]
    sig_s = [ N * nuc.xs[RXN.elastic_sc].xs for N , nuc in zip(nums_dens, nucs)]
    sig_a = [ N * nuc.xs[RXN.rad_cap].xs    for N , nuc in zip(nums_dens, nucs)]
    sig_p = [ N * nuc.pot_scatterxs_b for N , nuc in zip(nums_dens, nucs)  ]
    sig_t = sig_s + sig_a

    # get lethargy grid
    u = nucs[0].xs[RXN.elastic_sc].leth_boundaries

    # calculate group 1 flux
    phi1 = 0
    if ( boundary == BoundaryCondition.asymptotic_scatter_source ):
        du = u[1] - u[0]
        sum_sigp =  sum( [s   /(1 - alpha[i]) for i, s in enumerate(sig_p)] )
        sum_sig_s = sum( [s[0]/(1 - alpha[i]) for i, s in enumerate(sig_s)] )
        sum_sig_t = sum( [s[0] for s in sig_t] )
        ph1 = sum_sigp / ((du) * (sum_sig_t) - (sum_sig_s) * (du - 1 + np.exp(-du)) )

    # run the solver and retun the flux
    return(solver(sig_s, sig_t, alpha , u, phi1))

def plot_flux(energy, flux, name):
    f,a = process_data.fig_setup()
    plt.semilogx(energy, flux)
    plt.xlabel("Energy [eV]", fontsize=20)
    plt.ylabel("Scalar Flux [a.u.]", fontsize=20)
    a.tick_params(size=10, labelsize=20)
    plt.savefig(name + ".png")

def run_problem(material, display=False):
    for i in range(material.problems):
        print("Running problem 1/" + str(material.problems))
        simulation = material.get_problem_data(i)
        flux = slow_down(simulation)
        material.append_result(flux)

    if display:
        for i in range(material.problems):
            name = "problem_" + str(i)
            plot_flux(material.egrid , material.fluxes[i], name)


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
    run_problem(material, display=args.display)


if __name__ == "__main__":
    parse_args_and_run(sys.argv)
