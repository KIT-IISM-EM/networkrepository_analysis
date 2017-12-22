"""
A simple script to download the graph data from http://networkrepository.com
The script parses the table of all graphs to have access to information such as download size and restrict
the downloaded data with this information.

.. moduleauthor:: Fabian Ball <fabian.ball@kit.edu>
"""
from __future__ import print_function, unicode_literals
from future import standard_library
standard_library.install_aliases()
from builtins import zip

import argparse
import bs4
import os
import requests

import pandas as pd


def main():
    argparser = argparse.ArgumentParser()

    argparser.add_argument('out', type=str)
    argparser.add_argument('--min_size', type=int, default=0)
    argparser.add_argument('--max_size', type=int, default=1000 ** 2 * 10)
    argparser.add_argument('--source', type=str, default="http://networkrepository.com/networks.php")

    args = argparser.parse_args()

    df = load(args.source)
    print("Loaded dataframe with {} entries".format(len(df)))

    df = transform(df)

    df = df[(df["Size"] <= args.max_size) & (df["Size"] >= args.min_size)]  # Only networks smaller than XX MB
    print("Filtered dataframe which has {} remaining entries".format(len(df)))

    print("Starting to download the files...")
    download(df, args.out)


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


def transform(df):
    # Perform some transformations
    def normalize_size(value):
        tokens = value.split()

        if len(tokens) == 1:
            return int(value)
        elif len(tokens) == 2:
            size, unit = tokens
            size = int(size)

            if unit == "B":
                return size
            elif unit == "KB":
                return size * 1000
            elif unit == "MB":
                return size * 1000 ** 2
            elif unit == "GB":
                return size * 1000 ** 3

    df['Size'] = df['Size'].apply(normalize_size)

    return df


def download(df, out):
    for index, row in df.iterrows():
        print("Downloading {}".format(row["Graph Name"]))

        url = row["Download"]
        filename = url.split("/")[-1]
        r = requests.get(url, stream=True)
        with open(os.path.join(out, filename), mode="wb") as outfile:
            print("Saving to {}".format(outfile.name))
            for chunk in r.iter_content(1024):
                outfile.write(chunk)
            print("Done")

if __name__ == '__main__':
    main()
