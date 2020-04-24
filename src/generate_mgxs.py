__author__ = "Kyle Beyer"
__email__ = "beykyle@umich.edu"


import glob
import sys
import os
import pandas as pd
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
This module condenses ultra-fine group cross sections into coarse group cross sections given a solution for the
flux on the ultra-fine energy grid
"""

import slowpoke
import process_data
from slowpoke import build_nuclide_data
from process_data import Reactions as RXN
from process_data import CrossSection
from nuclide import Nuclide

class MG_Egrid:
    def __init__(self, upper_bounds_eV, IR_factors=None):
        self.upper_bounds_eV = upper_bounds_eV
        self.IR_factors = IR_factors

def ir_flux(N_nuc, nuc, N_bg_nuc, bg_nuc, egrid):
    s_bg = N_bg_nuc / N_nuc * bg_nuc.pot_scatterxs_b
    s_a =  nuc.xs[RXN.rad_cap].xs
    s_t = nuc.xs[RXN.elastic_sc].xs + nuc.xs[RXN.rad_cap].xs
    egrid_uf = nuc.xs[RXN.elastic_sc].E
    e_prev = np.min(egrid_uf)
    l = np.interp(egrid_uf, egrid.upper_bounds_eV, egrid.IR_factors)
    return (nuc.pot_scatterxs_b * l + s_bg ) / (s_a + l * nuc.pot_scatterxs_b + s_bg)

def nrim_flux(N_nuc, nuc, N_bg_nuc, bg_nuc):
    s_bg = N_bg_nuc / N_nuc * bg_nuc.pot_scatterxs_b
    s_a =  nuc.xs[RXN.rad_cap].xs
    return (s_bg) / (s_a + s_bg)

def nr_flux(N_nuc, nuc, N_bg_nuc, bg_nuc):
    s_bg = N_bg_nuc / N_nuc * bg_nuc.pot_scatterxs_b
    s_t = nuc.xs[RXN.elastic_sc].xs + nuc.xs[RXN.rad_cap].xs
    return (nuc.pot_scatterxs_b +  s_bg) / (s_t + s_bg)

def make_mgxs(egrid_coarse, egrid_uf, xs_uf, phi_uf):
    e_min = min(egrid_uf)
    xs_coarse = []
    for e in np.flip(egrid_coarse.upper_bounds_eV):
        mask = np.logical_and(egrid_uf < e , egrid_uf > e_min)
        e_min = e
        group_phi = np.extract(mask, phi_uf)
        group_xs  = np.extract(mask, xs_uf)
        xs_coarse.append(np.trapz( np.multiply(group_phi , group_xs) ) / np.trapz(group_phi))

    return np.array(xs_coarse)

def write_problem_data(path, egrid_coarse, xs_coarse, name):

    if path != None:
        path = path / (name + ".csv")
        print("Writing output to " + str(path))
    else:
        print(name)

    with process_data.smart_open(path) as fh:
        print(name , file=fh)
        print("{}, {}".format("Energy [eV]", "micro xs [b]"), file=fh)

        for i in range(len(egrid_coarse)):
            print("{:1.8e}, {:1.8e}".format( egrid_coarse[i], xs_coarse[i]), file=fh)

def read_mg_egrid(fpath, nuc):
    df = pd.read_csv(fpath)
    return MG_Egrid(df['e'], IR_factors=df['ir_' + nuc.name])

def read_phi_from_table(fpath):
    df = pd.read_csv(fpath)
    return df['Energy'] , df['Flux']

def make_phi_bundle(fpath, egrid, N_nuc, nuc, N_bg_nuc, bg_nuc ):
    e,p = read_phi_from_table(fpath)
    return e , {
            'IR'    : ir_flux(N_nuc, nuc, N_bg_nuc, bg_nuc , egrid)[1:],
            'NR'    : nr_flux(N_nuc, nuc, N_bg_nuc, bg_nuc )[1:],
            'NRIM'  : nrim_flux(N_nuc, nuc, N_bg_nuc, bg_nuc )[1:],
            'NSD'   : p
        }

def run(fpath, egrid_coarse, N_nuc, nuc, N_bg_nuc, bg_nuc, output_path):

    # for each phi in bundle
    e , phi_bundle = make_phi_bundle(fpath, egrid_coarse, N_nuc, nuc , N_bg_nuc, bg_nuc)
    mm = np.sum(phi_bundle['NSD'])
    for k in phi_bundle:
        lm = np.sum(phi_bundle[k])
        phi_bundle[k] = phi_bundle[k] * mm/lm
    plot_all_fluxes(e,phi_bundle)

    for k in phi_bundle:
        mgxs = make_mgxs(egrid_coarse, e, nuc.xs[RXN.rad_cap].xs[1:] , phi_bundle[k])
        write_problem_data(output_path  , egrid_coarse.upper_bounds_eV , mgxs, k)



def plot_all_fluxes(e , phi_bundle):
    f,a = process_data.fig_setup()
    for lbl, f in phi_bundle.items():
        plt.loglog(e, f, label=lbl)

    plt.xlabel("Energy [eV]", fontsize=20)
    plt.ylabel("Scalar Flux [a.u.]", fontsize=20)
    plt.legend(fontsize=18)
    a.tick_params(size=10, labelsize=20)
    plt.savefig("flux-compare.png")



def parse_args_and_run(argv: list):

    # default args
    current_path = Path(os.getcwd())
    def_out_path = None
    def_max_energy_eV = 2.0E4
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
    args = parser.parse_args()

    input_path = Path(args.input_path)
    base_path = input_path.parents[1]


    input_path = Path(args.input_path)
    slowpoke.validate_xml_path(input_path)
    grid = slowpoke.Grid(args.max_energy_eV , args.min_energy_eV, args.gridsize)
    output_path = args.output_path
    if output_path != None:
        output_path = Path(output_path)

    tree = et.parse(str(input_path))
    root = tree.getroot()
    nuclides = build_nuclide_data(root, input_path, grid)
    mg_node = root.find("mg_xs")

    # sort nuclides
    nuc_name = mg_node.get("nuclide")
    nuc_idx = 0
    bg_nucs = []
    for i,nuc in enumerate(nuclides):
        if nuc.name == nuc_name:
            nuc_idx = i
        else:
            bg_nucs.append(nuc)

    #TODO
    #for now assume one bg nuc, get ratio frome file
    ratio = float(root.find("material").get("mod_to_abs_ratio"))
    N_nuc = 1.
    N_bg_nuc = ratio
    bg_nuc = bg_nucs[0]

    # get egrid
    egrid_fpath = Path(mg_node.get("energy_grid_path"))
    egrid_fpath = base_path / egrid_fpath
    egrid_coarse = read_mg_egrid(egrid_fpath, nuc)

    # make_phi_bundle
    flux_fpath = Path(mg_node.get("flux_path"))
    flux_fpath = base_path / flux_fpath

    #run
    run(flux_fpath, egrid_coarse, N_nuc, nuc, N_bg_nuc, bg_nuc, output_path)

if __name__ == "__main__":
    parse_args_and_run(sys.argv)
