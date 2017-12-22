"""
A script that interactively removes duplicates from the analysis

.. moduleauthor:: Fabian Ball <fabian.ball@kit.edu>
"""
from __future__ import unicode_literals, print_function
from future import standard_library
standard_library.install_aliases()
from builtins import map, range
import argparse
import os
import sys

import pandas as pd

if sys.version_info < (3,):
    input = raw_input

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('infile', type=str, help='Path to the CSV file containing the analysis results')
    argparser.add_argument('--group_by', nargs='+',
                           default=['n', 'm', 'num_orbits', 'num_generators', 'aut_group_size'])
    argparser.add_argument('--width', type=int, default=120)

    args = argparser.parse_args()

    pd.set_option('line.width', args.width)

    df = pd.read_csv(args.infile, index_col=0)

    groups = df.groupby(args.group_by)

    to_delete = []

    for name, group in groups:
        if len(group) == 1:
            continue

        print('{}: {}'.format(', '.join(args.group_by), name))
        print('Found possible duplicate entries:')
        print(group)
        print('Type each index you want to delete from the results followed by <Return>.')
        print('Type an index a second time to remove it from the delete list.')
        print('Simply <Return> when you are finished')
        inp = ' '
        local_to_delete = set()
        possible_dups = list(group.index)
        while inp:
            inp = input('"{}" will be deleted. Anything else?'.format(", ".join(map(str, local_to_delete))
                                                                      if local_to_delete else 'Nothing'))
            inp = inp.strip()

            try:
                inp = int(inp)
            except ValueError:
                print('Invalid value "{}"'.format(inp))
            else:
                if inp in local_to_delete:
                    local_to_delete.remove(inp)
                elif inp in possible_dups:
                    local_to_delete.add(inp)
                else:
                    print('Invalid value "{}". Not in "{}"'.format(inp, ", ".join(map(str, possible_dups))))

        to_delete.extend(local_to_delete)

    df.drop(to_delete, inplace=True)

    df.index = range(len(df))

    df.to_csv('{}_nodup{}'.format(*os.path.splitext(args.infile)))
