__author__ = "Kyle Beyer"
__email__ = "beykyle@umich.edu"


import nuclide
import process_data
import glob

"""
This module performs numerical neutron slowing down calculations
"""


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
            help='Path to cross section files. Reads all ".csv" files in given dir. Assumes filename is nuclide name + ".csv"',
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
    files = [Path(x) for x in glob.glob(input_path + "*.csv")]
    print(files)

    nuc_name, xs_directory = build_xs_directory(input_fpath, args.max_en_eV, args.min_en_eV, args.numpoints)


if __name__ == "__main__":
    parse_args_and_run(sys.argv)
