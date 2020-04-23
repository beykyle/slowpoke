__author__ = "Kyle Beyer"
__email__ = "beykyle@umich.edu"


import process_data
import glob
import sys
import os
import argparse
import numpy as np
from numpy.linalg import multi_dot
import xml.etree.ElementTree as et
from pathlib import Path
from enum import Enum

# plottting
import matplotlib
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import MaxNLocator
import matplotlib.pylab as pylab
from matplotlib import rc
from matplotlib.colors import LogNorm
import matplotlib.font_manager

"""
This module performs numerical neutron slowing down calculations for homogenous mixtures
"""


from nuclide import Nuclide
from process_data import Reactions as RXN
#from skernel import solver_eq_leth as ksolver

class BoundaryCondition(Enum):
    asymptotic_scatter_source = 1

class Grid:
    def __init__(self, gmax, gmin, sz):
        self.max = gmax
        self.min = gmin
        self.size = sz

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

def skernel_const_gsize(in_scatter_source, sig_p_red, denom, alpha, du, phi):

    # calculate lowest group that can scatter into current group for each
    # TODO for now, we are assuming constant groupwidth for speed
    max_group_dist = np.array( [int(round(np.log(1/a) / du)) for a in alpha[:] ])

    (num_nuclides, num_groups) = in_scatter_source.shape

    for i in np.arange(1,num_groups):
        phi[i] =  0
        for nuc in np.arange(0,num_nuclides):
            min_g = i - max_group_dist[nuc]
            leftover = 0
            if min_g < 0:
                leftover = -min_g
                min_g = 0

            back_idx = np.arange(i,i+leftover)
            asym_scat_src = sig_p_red[nuc] * np.sum( (np.exp(-(back_idx-1)*du)*(1-np.exp(-du))**2) )
            phi[i] = phi[i] + np.dot( in_scatter_source[nuc][min_g:i] , phi[min_g:i] ) + asym_scat_src

        phi[i] = phi[i] / denom[i]

    return phi

def slow_down(nuclides, ratios, display=False, out=False, outpath=None):

    # get microscopic xs
    alpha = np.array([nuc.alpha for nuc in nuclides])
    sig_s = np.array( [nuc.xs[RXN.elastic_sc].xs for nuc in nuclides ])
    sig_a = np.array( [nuc.xs[RXN.rad_cap].xs for nuc in nuclides ])

    # get lethargy grid
    u = nuclides[0].xs[RXN.elastic_sc].leth_boundaries
    du = u[1:] - u[:-1]
    egrid = nuclides[0].xs[RXN.elastic_sc].E
    num_groups = len(u)-1

    # precompute exponential factors and bin widths
    exp_der = np.exp(-1*u[:-2]) + np.exp(-1*u[2:]) - 2 * np.exp(-1*u[1:-1])

    flx = {}
    save = True
    for i in range(len(ratios)):
        print("Running problem " + str(i))
        # new flux
        p = np.zeros(num_groups)
        sig_t_all = np.zeros(num_groups)

        # compute xs in form needed by solver
        sig_p = np.array( [n*nuc.pot_scatterxs_b/(1-a) for nuc,a,n in zip(nuclides,alpha,ratios[i])] )
        sig_s_red = np.vstack([n*s/(1-a) for s,a,n in zip(sig_s,alpha,ratios[i])])
        sig_p_red = np.vstack([n*s/(1-a) for s,a,n in zip(sig_p,alpha,ratios[i])])
        sig_s_red_all = np.sum(sig_s_red, axis=0)
        for j in range(0,len(alpha)):
            sig_t_all = sig_t_all + ratios[i][j] * (sig_s[j] + sig_a[j])

        # precompute denominator
        denom = np.multiply(sig_t_all,du) - np.multiply(sig_s_red_all, du - 1 + np.exp(-1* du))

        # precompute scattering sources
        in_scatter_source = sig_s_red[:,:-1] * exp_der

        # run solver kernel
        flux = skernel_const_gsize(in_scatter_source, sig_p_red, denom, alpha, du[0], p)

        # display and output
        if display:
            name = "problem_" + str(i)
            plot_flux(egrid , flux, sig_t_all, name)
        if out:
            name = "problem_" + str(i)
            write_problem_data(outpath, egrid, flux, name, ratios[i], nuclides)
        if save:
            flx[ratios[i][0]] = flux

    plot_all(egrid , flx, "all")

def plot_flux(energy, flux, sig_t_all, name):
    f,a = process_data.fig_setup()
    plt.semilogx(energy, flux, label="$\Phi$")
    plt.semilogx(energy, sig_t_all * max(flux)/max(sig_t_all), label=r"$\Sigma_t$ - scaled")
    plt.xlabel("Energy [eV]", fontsize=20)
    plt.ylabel("Scalar Flux [a.u.]", fontsize=20)
    plt.legend(fontsize=18)
    a.tick_params(size=10, labelsize=20)
    plt.savefig(name + ".png")

def plot_all(energy, fluxes, name):
    f,a = process_data.fig_setup()
    for lbl, flx in fluxes.items():
        label = r"$\frac{N_H}{N_{U238}} = $" + str(lbl)
        plt.semilogx(energy, flx, label=label)
    plt.xlabel("Energy [eV]", fontsize=20)
    plt.ylabel("Scalar Flux [a.u.]", fontsize=20)
    plt.legend(fontsize=18)
    a.tick_params(size=10, labelsize=20)
    plt.savefig(name + ".png")


def write_problem_data(path, en, flux, name, ratios, nuclides):
    if path != None:
        path = path + (name + ".csv")
        print("Writing output to " + path)
    with process_data.smart_open(path) as fh:
        print(name , file=fh)

        # iterate through the table and print in csv format
        print("Nuclides:", file=fh)
        print([n.name for n in nuclides], file=fh)
        print("Ratios:", file=fh)
        print(ratios, file=fh)
        print("{}, {}".format("Energy [eV]", "Flux [a.u.]"), file=fh)

        for i in range(len(en)):
            print("{:1.8e}, {:1.8e}".format( en[i], flux[i]), file=fh)


def parse_args_and_run(argv: list):

    # default args
    current_path = Path(os.getcwd())
    def_out_path = None
    def_max_energy_eV = 2.0E4
    def_min_energy_eV = 1.0
    def_gridsize = 6E5

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
    output_path = args.output_path
    if output_path != None:
        output_path = Path(output_path)


    tree = et.parse(str(input_path))
    root = tree.getroot()
    nuclides = build_nuclide_data(root, input_path, grid)
    r = np.array([[1. ,   1.] ,[2.5 ,   1.] ,[5. ,   1.]])
    slow_down(nuclides, r, display=args.display, out=True, outpath=args.output_path)

if __name__ == "__main__":
    parse_args_and_run(sys.argv)
