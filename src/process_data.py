__author__ = "Kyle Beyer"
__email__ = "beykyle@umich.edu"

import pandas as pd
import numpy as np
import csv
import sys
import os
import argparse
from enum import Enum
from pathlib import Path
import contextlib

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
This module reads in JANIS pointwise .csv style cross sections and interpolates them unto an ultra fine equal lethargy grid
"""

class Reactions(Enum):
    elastic_sc = 1
    rad_cap = 2


rxn_str = {
    Reactions.elastic_sc : "elastic scatter",
    Reactions.rad_cap    : "absoprtion"
}

class CrossSection:
    def __init__(self, energy_grid, micro_xs_b, rxn):
        self.E = energy_grid
        self.xs = micro_xs_b
        self.rxn = rxn
        self.leth = np.zeros(len(self.E))

    def indexFromLeth(leth):
        return (np.abs(self.leth - leth)).argmin()

    def indexFromEnergy(energy):
        return  (np.abs(self.E - energy)).argmin()

def read_xs_mesh(fpath):
    # intialize data structures
    energy = []
    xs = dict()
    xs[Reactions.elastic_sc] = []
    xs[Reactions.rad_cap] = []

    # read dat from file
    with open(fpath) as xs_file:
        reader = csv.reader(xs_file, delimiter=";")
        header = next(reader)
        col_names = [x.strip() for x in header]

        # what's in each column
        en_col = None
        c_col = None
        el_col = None
        for i, name in enumerate(col_names):
            if 'Incident energy' in name:
                en_col = i
            elif 'MT=102' in name:
                c_col = i
            elif 'MT=2' in name:
                el_col = i

        # skip 2 lines
        next(reader)
        next(reader)

        # read columns
        for row in reader:
            dat = [float(x.strip().rstrip("\n\r")) for x in row]
            energy.append(dat[en_col])
            xs[Reactions.elastic_sc].append(dat[el_col])
            xs[Reactions.rad_cap].append(dat[c_col])

        cross_sections = {}
        cross_sections[Reactions.elastic_sc] =  \
            CrossSection(np.array(energy), np.array(xs[Reactions.elastic_sc]), Reactions.elastic_sc)
        cross_sections[Reactions.rad_cap] =  \
            CrossSection(np.array(energy), np.array(xs[Reactions.rad_cap]), Reactions.rad_cap)

        return cross_sections


def interpolate_xs_equal_leth(xs, max_en_eV=20000.0, min_en_eV=1.0, num_points=600000):
    # limit energy and xs to specified energy grid
    # and calculate current leth grid
    # flip so lethargy=0 is first
    mask = np.logical_and(xs.E <= max_en_eV , xs.E >= min_en_eV)
    old_leth = np.flip(np.log(max_en_eV / xs.E))
    old_E = np.flip(np.extract(mask, xs.E))
    old_xs = np.flip(np.extract(mask, xs.xs))

    # setup new energy and lethargy grid structures
    new_leth = np.linspace(0,np.log(max_en_eV/min_en_eV), num_points)
    new_E = max_en_eV * np.exp(-1 * new_leth)

    # do the interpolation
    new_xs = np.interp(new_leth, old_leth, old_xs)

    # create and return a new xs object
    new_xs_obj = CrossSection(new_E, new_xs, xs.rxn)
    new_xs_obj.leth = new_leth
    return new_xs_obj

def fig_setup():
    fig = plt.figure(figsize=(12, 10), dpi=300, facecolor='w', edgecolor='k')
    ax = fig.add_subplot(111)
    rc('text', usetex=True)
    ax.tick_params(axis='both', which='major', labelsize=12)
    params = {'legend.fontsize': 20,
             'axes.labelsize': 20,
             'xtick.labelsize':20,
             'ytick.labelsize':20}
    pylab.rcParams.update(params)
    return fig , ax

def plot_all(xs_directory, name):
    f,a = fig_setup()
    for rxn , xs in xs_directory.items():
        plt.loglog(xs.E , xs.xs, label=rxn_str[rxn])
    plt.xlabel("Energy [eV]", fontsize=20)
    plt.ylabel("Cross Section [b]", fontsize=20)
    a.tick_params(size=10, labelsize=20)
    plt.legend()
    plt.savefig(name + "-xs-plot.png")

# for output
@contextlib.contextmanager
def smart_open(filename=None):
    if filename:
        fh = open(filename, 'w')
    else:
        fh = sys.stdout

    try:
        yield fh
    finally:
        if fh is not sys.stdout:
            fh.close()

def build_xs_directory(input_fpath, nuc_name, max_en_eV, min_en_eV, num_points):
    # read xs and run interpolation
    xs_directory = read_xs_mesh(input_fpath)
    for rxn , xs in xs_directory.items():
        xs_directory[rxn] = \
            interpolate_xs_equal_leth(xs, max_en_eV=max_en_eV, min_en_eV=min_en_eV, num_points=num_points)

    return xs_directory

def run(args):
    # file path specification
    input_fpath = Path(args.input_fpath)
    nuc_name = os.path.splitext(os.path.basename(input_fpath))[0]

    # read and intrpolate xs
    xs_directory = build_xs_directory(input_fpath, nuc_name, args.max_en_eV, args.min_en_eV, args.numpoints)

    # display
    if args.display:
        plot_all(xs_directory, nuc_name)

    # output
    output_fpath = args.output_fpath
    if output_fpath:
        output_fpath = Path(output_fpath) / (str(nuc_name) + "-equal-leth-ufgrid.csv")

    with smart_open(output_fpath) as fh:
        # get common grids from arbitrary reacton in directory
        energy_grid = xs_directory[Reactions.elastic_sc].E
        leth_grid = xs_directory[Reactions.elastic_sc].leth
        elastic = xs_directory[Reactions.elastic_sc].xs
        capture = xs_directory[Reactions.rad_cap].xs
        groups = args.numpoints

        # iterate through the table and print in csv format
        print("{}, {}, {}, {}".format(
                                      "Energy [eV]",
                                      "Lethargy",
                                      rxn_str[Reactions.elastic_sc] + " [b]",
                                      rxn_str[Reactions.rad_cap] + " [b]"),
              file=fh)

        for i in range(0,groups):
            print("{:1.8e}, {:1.8e}, {:1.8e}, {:1.8e}".format(
                                          energy_grid[i],
                                          leth_grid[i],
                                          elastic[i],
                                          capture[i]),
                  file=fh)

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
            help='Path to cross section ".csv" file', dest='input_fpath', required=True)
    parser.add_argument('-o', '--output',
            help='Path to write csv output file to - if not included, prints to stdout',
            dest='output_fpath', default=def_out_path)
    parser.add_argument('--max-energy', type=float,
                        help='maximum energy in [eV] for slowing down equations - for defining lethargy. Defalut: 2E4',
            dest='max_en_eV', default=def_max_energy_eV)
    parser.add_argument('--min-energy', type=float,
                        help='minimum energy in [eV] for slowing down equations - for defining lethargy. Default: 1.0',
            dest='min_en_eV', default=def_min_energy_eV)
    parser.add_argument('-n', '--num-gridpoints', type=int,
                        help='desired number of points on lethargy grid: Default: 6E5',
            dest='numpoints', default=def_gridsize)
    parser.add_argument('-d', '--display', action='store_true',
                        help='if flag present, generates cross section plots', dest='display')
    args = parser.parse_args()

    run(args)



if __name__ == "__main__":
    parse_args_and_run(sys.argv)
