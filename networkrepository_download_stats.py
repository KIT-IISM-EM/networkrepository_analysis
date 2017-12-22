"""
A simple script to get the graph stats from http://networkrepository.com
and save the results to a CSV file

.. moduleauthor:: Fabian Ball <fabian.ball@kit.edu>
"""
from __future__ import print_function, unicode_literals
from future import standard_library
standard_library.install_aliases()
from builtins import zip

import argparse
import bs4
import requests

import pandas as pd


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('out', type=str)
    argparser.add_argument('--source', type=str, default="http://networkrepository.com/networks.php")
    args = argparser.parse_args()

    df = load(args.source)
    print("Loaded dataframe with {} entries".format(len(df)))

    df.to_csv(args.out, encoding='utf8')


def load(url):
    # Load page with the table on it
    r = requests.get(url)
    print("Retrieved {}".format(url))
    # Create soup
    soup = bs4.BeautifulSoup(r.content, "lxml")

    data = []

    # Get column names
    columns = [th.text.strip() for th in soup.table.thead.tr.children]

    # Parse table
    for row in soup.table.tbody.children:
        # Get the text for each column except the 'Download' column needs special treatment
        row_data = [td.text.strip() if td.text.strip() != u"Download" else td.a.attrs["href"]
                    for td in row.children]
        data.append(dict(zip(columns, row_data)))

    # Create pandas dataframe from the records
    return pd.DataFrame.from_records(data)


if __name__ == '__main__':
    main()
