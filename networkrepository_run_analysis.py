"""
A script that analyzes the graph data downloaded from http://networkrepository.com

.. moduleauthor:: Fabian Ball <fabian.ball@kit.edu>
"""
from __future__ import unicode_literals
from future import standard_library
standard_library.install_aliases()
from builtins import object, range
import argparse
import collections
import io
import json
import logging
import os
import re
import shutil
import sys
import zipfile

import numpy as np
import networkx as nx
import pandas as pd
import scipy.io

import pysaucy
import pycggcrg

if sys.version_info < (3,):
    os.linesep = unicode(os.linesep)


class DisconnectedError(Exception):
    pass


class EdgesReader(object):
    """
    A custom parser for the .edges file format
    """
    class States(object):
        BEGIN = 'BEGIN'
        ADDITIONAL_HEADER = 'ADDITIONAL_HEADER'
        EDGES = 'EDGES'
        ERR = 'ERR'
        END = 'END'

    class Error(Exception):
        def __init__(self, msg, line, reader):
            msg = '{} ({}): {} (got: \'{}\')'.format(reader, reader.state, msg, line)
            super(EdgesReader.Error, self).__init__(msg)

    FLOAT_PATTERN = r'([+-])?\d+(\.\d+)?([eE][+-]?\d+)?'
    EDGE_LIST_PATTERN = r'^(?P<from_id>\d+)(?P<delimiter>[\s,;])(?P<to_id>\d+)((?P=delimiter)' \
                        r'(?P<weight>%s)((?P=delimiter)(?P<timestamp>%s))?)?$' % (FLOAT_PATTERN, FLOAT_PATTERN)
    HEADER_PATTERN = r'^(?P<comment_char>[%#]) (?P<graph_type>[a-zA-Z]+) (?P<weightedness>[a-zA-Z]+)$'
    # Comment char must be inserted before usable!
    ADDITIONAL_HEADER_PATTERN = r'^%s (?P<rel_count>\d+) (?P<subj_count>\d+) (?P<obj_count>\d+)$'

    edge_list_pattern = re.compile(EDGE_LIST_PATTERN)
    header_pattern = re.compile(HEADER_PATTERN)

    def __init__(self, valid_graph_types):
        self.state = EdgesReader.States.BEGIN

        self._graph_format = None
        self._weightedness = None
        self._comment_char = None
        self._m = None
        self._subj_count = None
        self._obj_count = None
        self._delimiter = None

        self._weighted = None
        self._timestamps = None

        self._valid_graph_types = valid_graph_types

        self._edges = []
        self.graph = None

    @property
    def graph_format(self):
        return self._graph_format

    @property
    def comment_char(self):
        return self._comment_char or '%'  # Fail safe variant

    @property
    def is_valid(self):
        return self._graph_format in self._valid_graph_types if self._graph_format else None

    @property
    def is_weighted(self):
        # Return the information that was given in the header
        if self._weightedness in ('weighted', ):
            return True
        elif self._weightedness in ('unweighted', ):
            return False
        else:  # Fallback, if no header information was available
            return self._weighted  # May be None => undetermined yet

    @property
    def has_timestamps(self):
        return self._timestamps

    def has_multiple_edges(self):
        """
        Check for multiple edges without taking weight and/or timestamp into account.
        If for _every_ edge (u, v) and (v, u) exists, the graph is not considered to have multiple edges!
        :return:
        """
        # Create a list of edge 2-tuples (a, b)
        edge_tuples = [(e['from_id'], e['to_id']) for e in self._edges]
        if len(edge_tuples) > len(set(edge_tuples)):  # Do 'real' multiple edges exist?
            return True

        # Create a list of edge 2-tuples (a, b) with a <= b
        edge_tuples = [(min(e['from_id'], e['to_id']), max(e['from_id'], e['to_id'])) for e in self._edges]
        edge_tuples_set = set(edge_tuples)

        if len(edge_tuples) == 2 * len(edge_tuples_set):  # This only happens if for each edge (a, b) also (b, a) exists
            return False
        else:
            # The set kicks out duplicate edges => less edges in the set means there were multiple edges
            return len(edge_tuples) > len(edge_tuples_set)

    def number_of_loops(self):
        return len([(e['from_id'], e['to_id']) for e in self._edges if e['from_id'] == e['to_id']])

    def has_loops(self):
        return self.number_of_loops() > 0

    def transform_bipartite(self):
        """
        This method shifts all 'to_id' node ids by the maximum 'from_id' node id.

        The caller must assure that the graph is actually bipartite and this operation makes sense!
        :return:
        """
        if not self.state == EdgesReader.States.EDGES:
            raise Exception('The reader is not in the right state')

        min_from_id = min(self._edges, key=lambda e: e['from_id'])['from_id']
        max_from_id = max(self._edges, key=lambda e: e['from_id'])['from_id']
        min_to_id = min(self._edges, key=lambda e: e['to_id'])['to_id']

        shift_offset = (max_from_id - min_from_id) + 1

        if self._subj_count and self._subj_count != shift_offset:
            raise EdgesReader.Error('The given subj_count "{subj_count}" is unequal '
                                    'the actual number of subjects "{count}""!'.format(subj_count=self._subj_count,
                                                                                       count=shift_offset), '-', self)

        if min_to_id == 0:  # Correct offset
            shift_offset += 1

        for edge in self._edges:
            edge['to_id'] += shift_offset

    def get_graph(self, simplify=False):
        if not self.state == EdgesReader.States.EDGES:
            raise EdgesReader.Error('The reader is not in the right state', '-', self)
        else:
            self.state = EdgesReader.States.END

        if simplify:
            g = SimpleGraph(data=((e['from_id'], e['to_id']) for e in self._edges))
        else:
            g = nx.Graph()
            g.add_edges_from(((e['from_id'], e['to_id']) for e in self._edges))

        return g

    def next(self, line):
        line = line.strip()
        if self.state == EdgesReader.States.BEGIN:
            self.read_header(line)
        elif self.state == EdgesReader.States.ADDITIONAL_HEADER:
            self.read_additional_header(line)
        elif self.state == EdgesReader.States.EDGES:
            self.read_edge(line)
        elif self.state == EdgesReader.States.END:
            pass
        elif self.state == EdgesReader.States.ERR:
            raise EdgesReader.Error('The reader can\'t be used anymore.', line, self)

    def read_header(self, line):
        matches = self.edge_list_pattern.match(line)

        if matches:  # Already an edge
            self.state = EdgesReader.States.EDGES
            self.next(line)
            return

        matches = self.header_pattern.match(line)

        if not matches:
            self.state = EdgesReader.States.ERR
            raise EdgesReader.Error('Invalid first line', line, self)

        match_groups = matches.groupdict()
        self._comment_char = match_groups['comment_char']
        self._graph_format = match_groups['graph_type']
        self._weightedness = match_groups['weightedness']
        self.state = EdgesReader.States.ADDITIONAL_HEADER

    def read_additional_header(self, line):
        additional_info_pattern = self.ADDITIONAL_HEADER_PATTERN % self.comment_char

        matches = re.match(additional_info_pattern, line)

        if not matches:  # No additional header => it must be an edge
            self.state = EdgesReader.States.EDGES
            self.next(line)
            return

        match_groups = matches.groupdict()
        self._m = int(match_groups['rel_count'])
        self._subj_count = int(match_groups['subj_count'])
        self._obj_count = int(match_groups['obj_count'])
        self.state = EdgesReader.States.EDGES

    def read_edge(self, line):
        if line.startswith(self.comment_char):  # Skip comments
            return

        matches = self.edge_list_pattern.match(line)

        if not matches:
            self.state = EdgesReader.States.ERR
            raise EdgesReader.Error('Invalid edge', line, self)

        match_groups = matches.groupdict()

        # Set or check the delimiter
        if self._delimiter is None:
            self._delimiter = match_groups['delimiter']
        elif match_groups['delimiter'] != self._delimiter:
            self.state = EdgesReader.States.ERR
            raise EdgesReader.Error('Found other delimiter than "{}"'.format(self._delimiter), line, self)

        # Get information
        if match_groups['weight'] and match_groups['timestamp']:
            weight = float(match_groups['weight'])
            timestamp = float(match_groups['timestamp'])
        elif match_groups['weight'] and self._is_weight(match_groups['weight']):
            weight = float(match_groups['weight'])
            timestamp = None
        elif match_groups['weight'] and not self._is_weight(match_groups['weight']):
            weight = None
            timestamp = float(match_groups['weight'])
        else:
            weight = None
            timestamp = None

        # Set weightedness and timestamps if this is the first edge
        if not self._edges:
            self._timestamps = timestamp is not None
            self._weighted = weight is not None

        # Check if edge information matches the global information
        if self.is_weighted and weight is None:
            self.state = EdgesReader.States.ERR
            raise EdgesReader.Error('Found an unweighted edge but the graph is weighted', line, self)
        elif not self.is_weighted and weight is not None:
            self.state = EdgesReader.States.ERR
            raise EdgesReader.Error('Found an weighted edge but the graph is unweighted', line, self)

        if self.has_timestamps and timestamp is None:
            self.state = EdgesReader.States.ERR
            raise EdgesReader.Error('Found an edge without timestamp but the graph has timestamps', line, self)
        elif not self.has_timestamps and timestamp is not None:
            self.state = EdgesReader.States.ERR
            raise EdgesReader.Error('Found an timestamped edge but the graph has not timestamps', line, self)

        # Create edge information
        edge = {'from_id': int(match_groups['from_id']), 'to_id': int(match_groups['to_id'])}
        if weight is not None:
            edge['weight'] = weight

        if timestamp is not None:
            edge['timestamp'] = timestamp

        self._edges.append(edge)

    @staticmethod
    def _is_weight(value):
        ts = int(float(value))

        # Heuristic: If the extension converted to int is larger than the threshold, it is assumed to be a timestamp
        # (datetime.datetime.fromtimestamp(10**8) => datetime.datetime(1973, 3, 3, 10, 46, 40)
        if ts >= 10 ** 8:
            return False
        else:
            return True


class EmptyDict(dict):
    def __setitem__(self, key, value):
        pass

    def update(self, other=None, **kwargs):
        pass


class SimpleGraph(nx.Graph):
    edge_attr_dict_factory = EmptyDict  # Disallow any edge attributes

    def __init__(self, data=None, **attr):
        super(SimpleGraph, self).__init__(data, **attr)

        self.remove_edges_from(nx.selfloop_edges(self))


def networkx_to_pysaucy(graph):
    """
    Convert from WeightedGraph to :class:`pysaucy.Graph` datastructure.

    :param graph: A graph
    :type graph: WeightedGraph
    :return: The converted graph
    :rtype: :class:`pysaucy.Graph`
    """
    adjacency_list = [[] for _ in range(graph.number_of_nodes())]
    for edge in graph.edges:
        adjacency_list[int(edge[0])].append(int(edge[1]))
    g = pysaucy.Graph(adjacency_list)

    return g


def PathType(path_str):
    """
    Determine if the given string is an existing path in the file system.

    :param path_str:
    :return:
    """
    orig = path_str
    path_str = os.path.expanduser(path_str)  # Expand user path if necessary
    path_str = os.path.abspath(path_str)

    if os.path.exists(path_str):
        return path_str
    else:
        raise argparse.ArgumentError(message='"{}" is not a valid path'.format(orig))


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('basedir',
                           help='Path to the directory that contains the data sets',
                           type=PathType)
    argparser.add_argument('resultpath',
                           help='Path of a filename without extension where results are saved')
    argparser.add_argument('--simplify', type=bool, default=False, nargs='?', const=True,
                           help='Simplify non simple graphs be removing weights, multiple edges and loops.')

    args = argparser.parse_args()

    # Setup
    setup_logger(args.resultpath + '.log')
    setup_directories(args.basedir)

    # Iterate over the zipped graphs in the specified directory
    for dirpath, _, filenames in os.walk(args.basedir):
        for graph_zip in filenames:
            if not graph_zip.endswith('.zip'):
                continue

            try:
                graph = handle_file(dirpath, graph_zip, args.simplify)  # Parse the zip-file and return a networkx-Graph
            except Exception as e:
                logging.error('Could not read "{}": {}'.format(graph_zip, e))
                continue

            if not graph:  # The graph seems to be invalid for us -> move the file
                shutil.move(os.path.join(dirpath, graph_zip), os.path.join(dirpath, 'invalid', graph_zip))
                continue

            try:
                graph = transform(graph)  # Normalization of node ids
                record = analyze(graph)  # The magic happens here

            except DisconnectedError as e:
                logging.warning('Error processing {}: {}'.format(graph_zip, e))
                shutil.move(os.path.join(dirpath, graph_zip), os.path.join(dirpath, 'disconnected', graph_zip))
            except Exception as e:  # Not very elegant but hopefully quite fail-safe
                logging.warning('Error processing {}: {}'.format(graph_zip, e))
                shutil.move(os.path.join(dirpath, graph_zip), os.path.join(dirpath, 'error', graph_zip))
            else:
                if record:
                    logging.info('Got result: {}'.format(record))
                    # Quick save the result
                    with io.open(args.resultpath + '.results', 'a') as f:
                        f.write(json.dumps(record) + os.linesep)
                    shutil.move(os.path.join(dirpath, graph_zip), os.path.join(dirpath, 'done', graph_zip))
                else:
                    logging.warning('Error processing {}: No record returned'.format(graph_zip))
                    shutil.move(os.path.join(dirpath, graph_zip), os.path.join(dirpath, 'error', graph_zip))

        break  # Do not traverse subdirs!!

    # Load the saved results to create a pandas DataFrame
    if os.path.exists(args.resultpath + '.results'):
        with io.open(args.resultpath + '.results', 'r') as f:
            records = [json.loads(line) for line in f]

        df = pd.DataFrame(records)
        df = df.replace('NA', np.nan)

        df.to_csv(args.resultpath + '.csv')  # Simply save the results


def setup_logger(file_path):
    logging.basicConfig(filename=file_path,
                        level=logging.DEBUG,
                        format='%(asctime)s - %(levelname)s - %(message)s')


def setup_directories(path):
    dirs = ['error', 'invalid', 'done', 'disconnected']
    for d in dirs:
        subdir = os.path.join(path, d)
        if not os.path.exists(subdir):
            os.mkdir(subdir)


def handle_file(path, name, simplify):
    """
    Unzips the graph data and tries to load the graph
    :param path: Path to the graph
    :param name: Filename of the zipfile
    :param simplify: If True, simplify non simple graphs
    :return: A graph or None on error
    :rtype: :class:`networkx.Graph` | None
    """
    filename, _ = os.path.splitext(name)
    target_folder = os.path.join(path, filename)
    if os.path.exists(target_folder):
        logging.info('{} already exists. Skip.'.format(target_folder))
        return

    os.mkdir(target_folder)
    os.chdir(target_folder)

    g = None

    try:
        with zipfile.ZipFile(os.path.join(path, name)) as z:
            z.extractall()
    except zipfile.BadZipfile as e:
        logging.error('{}: {}'.format(name, e))

        return

    for dirpath, dirnames, filenames in os.walk(target_folder):
        # Remove all readme files
        for filename in filenames[:]:
            if filename.lower().startswith('readme'):
                filenames.remove(filename)

        # if len(dirnames) != 0:
        #     logging.info('{}: There are subdirs ({}). Not implemented.'.format(name, dirnames))
        #     break

        if len(filenames) == 1:
            filename, extension = os.path.splitext(filenames[0])

            if extension == '.mtx':
                g = handle_mtx(dirpath, filenames[0], simplify)
            elif extension in ('.edges', '.txt'):
                g = handle_edges(dirpath, filenames[0], simplify)
            else:
                logging.info('{}: Extension {} not implemented'.format(name, extension))
        elif len(filenames) > 1:
            filename, _ = os.path.splitext(name)
            if '{}.edges'.format(filename) in filenames:
                g = handle_edges(dirpath, '{}.edges'.format(filename))
            else:
                logging.info('{}: Not implemented: {}'.format(name, filenames))

    shutil.rmtree(target_folder)

    return g


def _fix_mtx(path):
    with io.open(path, mode='r+') as f:
        # Read the first line...
        first_line = f.readline()
        # ... check for the missing character.
        if first_line.startswith('%MatrixMarket'):
            f.seek(0)  # Rewind

            # Quick and dirty: read the whole file into memory...
            lines = f.readlines()

            # ... add the missing character ...
            lines[0] = '%' + lines[0]
            # ... rewind ...
            f.seek(0)
            # ... and write everything back.
            f.writelines(lines)

            del lines


def handle_mtx(path, name, simplify=False):
    """
    Handle a file that is in Matrix Market matrix format (http://math.nist.gov/MatrixMarket/formats.html#MMformat)

    We only load graphs that have the format '%%MatrixMarket matrix coordinate pattern symmetric'
    (first line in the file). I.e. unweighted and symmetric/undirected

    :param path: Path to the unzipped graph data
    :param name: Filename of the .mtx-file
    :param simplify: (Optional) Simplify the graph
    :return: A graph or None on error
    :rtype: :class:`networkx.Graph` | None
    """
    filepath = os.path.join(path, name)

    # Some data has wrong format: The first line starts with '%MatrixMarket' instead of '%%MatrixMarket'
    # -> Fix this in advance
    _fix_mtx(filepath)

    try:
        rows, cols, entries, matrix_format, field, symm = scipy.io.mminfo(filepath)
    except ValueError as e:
        logging.error('{}: {}'.format(name, e))
        return None

    if not simplify and matrix_format == 'coordinate' and field == 'pattern' and symm == 'symmetric':
        matrix = scipy.io.mmread(filepath)

        g = nx.Graph(matrix, name=name)
        del matrix
        if nx.number_of_selfloops(g):
            logging.error('{}: Graph has loops'.format(name))
            return None

        return g
    elif simplify and matrix_format == 'coordinate' and field in ('pattern', 'real', 'integer', 'complex'):
        logging.info('{} has format ({}, {}, {}). Simplify.'.format(name, matrix_format, field, symm))
        matrix = scipy.io.mmread(filepath)  # Scipy (sparse) matrix

        if matrix.shape[0] != matrix.shape[1]:
            logging.error('{}: Data matrix is not symmetric {})'.format(name, matrix.shape))
            return None

        # SimpleGraph is undirected, neglects edge weights and removes loops
        g = SimpleGraph(matrix, name=name)
        del matrix

        return g
    else:
        logging.info('{} is of the wrong format ({}, {}, {}). Skip.'.format(name, matrix_format, field, symm))
        return None


def _handle_nodes(path, name, graph):
    filepath = os.path.join(path, name)
    labels = {}

    line_pattern = re.compile(r'^(?P<n_id>\d+)[ ]+(?P<n_label>\d+)\n$')

    with io.open(filepath) as f:
        for line in f:
            matches = line_pattern.match(line)

            if not matches:
                logging.error('{}: Invalid node label line "{}"'.format(name, line))
                return None

            m_dict = matches.groupdict()
            n_id = int(m_dict['n_id'])
            n_label = int(m_dict['n_label'])
            labels[n_id] = n_label

    if len(labels) != graph.number_of_nodes():
        logging.error('{}: Invalid number of node labels detected'.format(name))
        return None
    else:
        logging.info('{}: Found node label information'.format(name))
        return labels


def _handle_edges_strict(path, name):
    filepath = os.path.join(path, name)

    edges_reader = EdgesReader(['bip', 'bipartite', 'sym', 'symmetric', 'undirected'])

    with io.open(filepath) as f:
        for line in f:
            try:
                edges_reader.next(line)
            except EdgesReader.Error as e:
                logging.error('{}: {}'.format(name, e))
                return None

    if edges_reader.is_valid is False:
        logging.error('{}: Invalid graph format "{}"'.format(name, edges_reader.graph_format))
        return None

    if edges_reader.is_weighted is True:
        logging.error('{}: Graph is weighted'.format(name))
        return None

    if edges_reader.graph_format in ('bip', 'bipartite'):
        try:
            edges_reader.transform_bipartite()
        except EdgesReader.Error as e:
            logging.error('{}: {}'.format(name, e))
            return None

        logging.info('{}: Graph is bipartite and to_node ids were shifted!'.format(name))

    if edges_reader.has_multiple_edges():
        logging.error('{}: Graph has multiple edges'.format(name))
        return None

    if edges_reader.has_loops():
        logging.error('{}: Graph has loops'.format(name))
        return None

    g = edges_reader.get_graph()
    g.name = name

    # Check if node information exists and set the node labels if possible
    n, ext = os.path.splitext(name)
    nodes_name = "{}.nodes".format(n)
    if os.path.exists(os.path.join(path, nodes_name)):
        label_dict = _handle_nodes(path, nodes_name, g)

        if label_dict:
            for n_id, label in label_dict.items():
                if n_id not in g.nodes:
                    logging.error('{}: Invalid node id {} detected'.format(nodes_name, n_id))
                    return None

                g.nodes[n_id]['label'] = label

    return g


def _handle_edges_simple(path, name):
    filepath = os.path.join(path, name)

    edges_reader = EdgesReader(['bip', 'bipartite', 'sym', 'symmetric', 'asym', 'undirected', 'directed'])

    with io.open(filepath) as f:
        for line in f:
            try:
                edges_reader.next(line)
            except EdgesReader.Error as e:
                logging.error('{}: {}'.format(name, e))
                return None

    if edges_reader.is_valid is False:
        logging.error('{}: Invalid graph format "{}"'.format(name, edges_reader.graph_format))
        return None

    if edges_reader.graph_format in ('bip', 'bipartite'):
        try:
            edges_reader.transform_bipartite()
        except EdgesReader.Error as e:
            logging.error('{}: {}'.format(name, e))
            return None

        logging.info('{}: Graph is bipartite and to_node ids were shifted!'.format(name))

    g = edges_reader.get_graph(simplify=True)
    g.name = name

    # Check if node information exists and set the node labels if possible
    n, ext = os.path.splitext(name)
    nodes_name = "{}.nodes".format(n)
    if os.path.exists(os.path.join(path, nodes_name)):
        label_dict = _handle_nodes(path, nodes_name, g)

        if label_dict:
            for n_id, label in label_dict.items():
                if n_id not in g.nodes:
                    logging.error('{}: Invalid node id {} detected'.format(nodes_name, n_id))
                    return None

                g.nodes[n_id]['label'] = label

    return g


def handle_edges(path, name, simplify=False):
    """
    Handle a file that has edge list format (extension '.edges') using the custom
    parser.

    :param path: Path to the unzipped graph data
    :param name: Filename of the .edges-file
    :param simplify: Simplify the graph
    :return: A graph or None on error
    :rtype: :class:`networkx.Graph` | None
    """

    if simplify:
        return _handle_edges_simple(path, name)
    else:
        return _handle_edges_strict(path, name)


def transform(g):
    """
    Some data formats are zero-indexed, some one-indexed. Some others use explicit labels for the nodes.
    At least for the RG datastructure and algorithms we need the graph to be zero-indexed.
    Therefore we relabel every graph to have node labels/ids from 0 to n-1.

    :param g: A graph
    :return: A graph with possibly relabeled node ids
    """
    min_nidx = min(g.nodes)
    max_nidx = max(g.nodes)

    if min_nidx == 0 and max_nidx == g.number_of_nodes() - 1:  # Everything already labeled as wanted
        return g

    nodes = sorted(g.nodes)  # Get the sorted nodes
    # Relabel the nodes by their index in the list
    relabel_dict = {nidx: idx for idx, nidx in enumerate(nodes)}

    # Also shift node labels (important for saucy)
    if 'label' in g.nodes[g.nodes.keys()[0]]:
        for n_id in g.nodes:
            g.nodes[n_id]['label'] = relabel_dict[g.nodes[n_id]['label']]

    g = nx.relabel_nodes(g, relabel_dict)
    assert min(g.nodes) == 0 and max(g.nodes) == g.number_of_nodes() - 1

    return g


def _compute_modularity(g):
    graph = pycggcrg.make_graph(g.number_of_nodes(), list(g.edges), check_connectedness=False)
    q = pycggcrg.run_rg(graph, runs=10)

    return q


def _get_color_partition(graph):
    if 'label' not in graph.nodes[0]:
        return None

    colors = [None] * graph.number_of_nodes()

    for n_id in graph.nodes:
        colors[n_id] = graph.nodes[n_id]['label']

    assert None not in colors

    return colors


def analyze(g):
    """
    This function analyzes a graph and returns a data-record which is a dictionary of results for the
    different performance indices.

    :param g: A graph
    :return: Result record
    """
    record = {}
    logging.debug('Analyzing {} ({}, {})'.format(g.name, g.number_of_nodes(), g.number_of_edges()))

    if not nx.is_connected(g):
        raise DisconnectedError('{} is not connected: {}'.format(g, nx.number_connected_components(g)))

    record['name'] = g.name
    record['n'] = g.number_of_nodes()
    record['m'] = g.number_of_edges()
    record['density'] = nx.density(g)
    # record['degrees'] = networkx.degree(g)
    # record['diameter'] = networkx.diameter(g)

    # Compute modularity
    logging.debug('Compute modularity')
    record['modularity'] = _compute_modularity(g)

    empty_result = {'num_orbits': 'NA', 'num_generators': 'NA', 'aut_group_size': 'NA'}
    # Compute the automorphism group with saucy
    logging.debug('Convert saucy')
    g_pysaucy = networkx_to_pysaucy(g)
    colors = _get_color_partition(g)
    del g

    try:
        logging.debug('Run saucy')
        result = pysaucy.run_saucy(g_pysaucy, colors)
    except Exception as e:
        logging.error('{}: {}'.format(record['name'], e))
        record.update(empty_result)
    else:
        grpsize1, grpsize2, levels, nodes, bads, num_generators, support, orbit_ids = result

        orbit_sizes = collections.defaultdict(int)
        for orbit_id in orbit_ids:
            orbit_sizes[orbit_id] += 1
        record['num_orbits'] = len(orbit_sizes)
        record['num_generators'] = num_generators
        record['aut_group_size'] = grpsize1
        record['aut_group_size_exp'] = grpsize2

    return record


if __name__ == '__main__':
    main()
