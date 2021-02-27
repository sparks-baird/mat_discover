'''
The Element Movers Distance is an application of the Wasserstein metric between
two compositional vectors
Copyright (C) 2020 Cameron Hargreaves
This file is part of The Element Movers Distance
<https://github.com/lrcfmd/ElMD>
The Element Movers Distance is free software: you can redistribute it and/or 
modify it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
The Element Movers Distance is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
You should have received a copy of the GNU General Public License
along with dogtag.  If not, see <http://www.gnu.org/licenses/>.
__author__ = "Cameron Hargreaves"
__copyright__ = "2019, Cameron Hargreaves"
__credits__ = ["https://github.com/Zapaan", "Loïc Séguin-C. <loicseguin@gmail.com>", "https://github.com/Bowserinator/"]
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Cameron Hargreaves"
'''

import re
from collections import Counter
from copy import deepcopy

import numpy as np
from scipy.spatial.distance import squareform
from numba import njit

def main():
    x = ElMD("NaCl")
    print(x.elmd("LiCl"))
    x = ElMD("Li7La3Hf2O12", metric="atomic")
    y = ElMD("CsPbI3", metric="atomic")
    z = ElMD("Zr3AlN", metric="atomic")

    print(x.elmd(y))
    print(y.elmd(z))
    print(x)

    elmd = ElMD().elmd

    print(elmd("Zr3AlN", "CaTiO3"))

def EMD(comp1, comp2):
    '''
    A numba compiled EMD function to compare two sets of pettifor labels and
    ratios as a ratio vector in the form (103, )
    '''
    
    if type(comp1) is str:
        comp1_elmd = ElMD(comp1).vector_form
    else:
        comp1_elmd = comp1 

    if type(comp2) is ElMD:
        comp2_elmd = comp2.vector_form
    elif type(comp2) is str:
        comp2_elmd = ElMD(comp2).vector_form
    else:
        comp2_elmd = comp2

    source_labels = np.where(comp1_elmd > 0)[0]
    source_demands = comp1_elmd[source_labels]
    sink_labels = np.where(comp2_elmd > 0)[0]
    sink_demands = comp2_elmd[sink_labels]

    return network_simplex(source_labels, source_demands, sink_labels, sink_demands)

class ElMD():
    ATOM_REGEX = '([A-Z][a-z]*)(\d*\.*\d*)'
    OPENERS = '({['
    CLOSERS = ')}]'

    # As the current optimization solver only takes in ints we must multiply
    # all floats to capture the decimal places
    FP_MULTIPLIER = 100000000

    def __init__(self, formula="", metric="mod_petti"):
        self.metric = metric
        self.formula = ''.join(formula.split()) # Remove all whitespace
        self.periodic_tab = self._get_periodic_tab()
        self.composition = self._parse_formula(self.formula)
        self.normed_composition = self._normalise_composition(self.composition)
        self.vector_form = self._gen_vector(self.normed_composition)
        self.pretty_formula = self._gen_pretty(self.vector_form)

    def elmd(self, comp2 = None, comp1 = None, verbose=False):
        '''
        Calculate the minimal cost flow between two weighted vectors using the
        network simplex method. This is overloaded to accept a range of input
        types.
        TODO Remove some of the extra function parameters?
        '''
        if comp1 == None:
            comp1 = self.vector_form

        if isinstance(comp1, str):
            comp1 = self._parse_formula(comp1)
            comp1 = self._normalise_composition(comp1)
            comp1 = self._gen_vector(comp1)

        if isinstance(comp1, ElMD):
            comp1 = comp1.normed_composition

        if isinstance(comp2, str):
            comp2 = self._parse_formula(comp2)
            comp2 = self._normalise_composition(comp2)
            comp2 = self._gen_vector(comp2)

        if isinstance(comp2, ElMD):
            comp2 = comp2.vector_form

        return EMD(comp1, comp2)

    def _gen_vector(self, comp):
        '''
        Create a numpy array from a composition dictionary
        '''
        if isinstance(comp, str):
            comp = self._parse_formula(comp)
            comp = self._normalise_composition(comp)

        comp_labels = []
        comp_ratios = []

        for k in sorted(comp.keys()):
            comp_labels.append(self._get_position(k))
            comp_ratios.append(comp[k])

        indices = np.array(comp_labels, dtype=np.int64)
        ratios = np.array(comp_ratios, dtype=np.float64)

        numeric = np.zeros(shape=103, dtype=np.float64)
        numeric[indices] = ratios

        return numeric

    def _gen_pretty(self, vector):
        '''
        Return a normalized formula string from the vector format, TODO clearer
        as a list comprehension or too long?
        '''
        inds = np.where(vector != 0.0)[0]
        pretty_form = ""
        lookup = self.periodic_tab[self.metric]

        for i, ind in enumerate(inds):
            if vector[ind] == 1:
                pretty_form = pretty_form + f"{lookup[ind]}"
            else:
                pretty_form = pretty_form + f"{lookup[ind]}{vector[ind]:.3f}".strip('0') + ' '

        return pretty_form.strip()

    def _get_periodic_tab(self):
        """
        Attempt to load periodic data from the same folder, else download
        it from the web
        """
        return ElementDict

    def _is_balanced(self, formula):
        """Check if all sort of brackets come in pairs."""
        # Very naive check, just here because you always need some input checking
        c = Counter(formula)
        return c['['] == c[']'] and c['{'] == c['}'] and c['('] == c[')']

    def _dictify(self, tuples):
        """Transform tuples of tuples to a dict of atoms."""
        res = dict()
        for atom, n in tuples:
            try:
                res[atom] += float(n or 1)
            except KeyError:
                res[atom] = float(n or 1)
        return res

    def _fuse(self, mol1, mol2, w=1):
        """ Fuse 2 dicts representing molecules. Return a new dict. """
        return {atom: (mol1.get(atom, 0) + mol2.get(atom, 0)) * w for atom in set(mol1) | set(mol2)}

    def _parse(self, formula):
        """
        Return the molecule dict and length of parsed part.
        Recurse on opening brackets to parse the subpart and
        return on closing ones because it is the end of said subpart.
        """
        q = []
        mol = {}
        i = 0

        while i < len(formula):
            # Using a classic loop allow for manipulating the cursor
            token = formula[i]

            if token in self.CLOSERS:
                # Check for an index for this part
                m = re.match('\d+\.*\d*|\.\d*', formula[i+1:])
                if m:
                    weight = float(m.group(0))
                    i += len(m.group(0))
                else:
                    weight = 1

                submol = self._dictify(re.findall(self.ATOM_REGEX, ''.join(q)))
                return self._fuse(mol, submol, weight), i

            elif token in self.OPENERS:
                submol, l = self._parse(formula[i+1:])
                mol = self._fuse(mol, submol)
                # skip the already read submol
                i += l + 1
            else:
                q.append(token)

            i += 1

        # Fuse in all that's left at base level
        return self._fuse(mol, self._dictify(re.findall(self.ATOM_REGEX, ''.join(q)))), i

    def _parse_formula(self, formula):
        """Parse the formula and return a dict with occurences of each atom."""
        if not self._is_balanced(formula):
            raise ValueError("Your brackets not matching in pairs ![{]$[&?)]}!]")

        return self._parse(formula)[0]

    def _normalise_composition(self, input_comp):
        """ Sum up the numbers in our counter to get total atom count """
        composition = deepcopy(input_comp)
        # check it has been processed
        if isinstance(composition, str):
            composition = self._parse_formula(composition)

        atom_count =  sum(composition.values(), 0.0)

        for atom in composition:
            composition[atom] /= atom_count

        return composition

    def _get_atomic_num(self, element_string):
        """ Return atomic number from element string """
        for i, element in enumerate(self.periodic_tab[self.metric]):
            if element['symbol'] == element_string:
                return i

    def _get_position(self, element, metric=None):
        """
        Return either the x, y coordinate of an elements position, or the
        x-coordinate on the Pettifor numbering system as a 2-dimensional
        """
        try:
            atomic_num = self.periodic_tab[self.metric][element]

            return atomic_num
        # If this fails for any reason return -1
        except:

            return -1

    def _return_positions(self, composition):
        """ Return a dictionary of associated positions for each element """
        element_pos = {}

        for element in composition:
            element_pos[element] = self._get_position(element, metric="manhattan")

        return element_pos

    def __repr__(self):
        return f"ElMD({self.pretty_formula})"

    def __len__(self):
        return len(self.normed_composition)

    def __eq__(self, other):
        return self.pretty_formula == other.pretty_formula

'''
This is an implementation of the network simplex algorithm for computing the
minimal flow atomic similarity distance between two compounds
Copyright (C) 2019  Cameron Hargreaves
Copyright (C) 2010 Loïc Séguin-C. <loicseguin@gmail.com>
All rights reserved.
BSD license.
'''

@njit()
def reduced_cost(i, costs, potentials, tails, heads, flows):
    """Return the reduced cost of an edge i.
    """
    c = costs[i] - potentials[tails[i]] + potentials[heads[i]]

    if flows[i] == 0:
        return c
    else:
        return -c

@njit()
def find_entering_edges(e, f, tails, heads, costs, potentials, flows):
    """Yield entering edges until none can be found.
    """
    # Entering edges are found by combining Dantzig's rule and Bland's
    # rule. The edges are cyclically grouped into blocks of size B. Within
    # each block, Dantzig's rule is applied to find an entering edge. The
    # blocks to search is determined following Bland's rule.

    B = np.int64(np.ceil(np.sqrt(e))) # block size

    M = (e + B - 1) // B    # number of blocks needed to cover all edges
    m = 0

    while m < M:
        # Determine the next block of edges.
        l = f + B
        if l <= e:
            edge_inds = np.arange(f, l)
        else:
            l -= e
            edge_inds = np.concatenate((np.arange(f, e), np.arange(l)))

        f = l

        # Find the first edge with the lowest reduced cost.
        r_costs = np.empty(edge_inds.shape[0])

        for y, z in np.ndenumerate(edge_inds):
            r_costs[y] = reduced_cost(z, costs, potentials, tails, heads, flows)

        # This takes the first occurrence which should stop cycling
        h = np.argmin(r_costs)

        i = edge_inds[h]
        c = reduced_cost(i, costs, potentials, tails, heads, flows)

        p = q = -1

        if c >= 0:
            m += 1

        # Entering edge found.
        else:
            if flows[i] == 0:
                p = tails[i]
                q = heads[i]
            else:
                p = heads[i]
                q = tails[i]

            return i, p, q, f

    # All edges have nonnegative reduced costs. The flow is optimal.
    return -1, -1, -1, -1

@njit()
def find_apex(p, q, size, parent):
    """Find the lowest common ancestor of nodes p and q in the spanning
    tree.
    """
    size_p = size[p]
    size_q = size[q]

    while True:
        while size_p < size_q:
            p = parent[p]
            size_p = size[p]
        while size_p > size_q:
            q = parent[q]
            size_q = size[q]
        if size_p == size_q:
            if p != q:
                p = parent[p]
                size_p = size[p]
                q = parent[q]
                size_q = size[q]
            else:
                return p

@njit()
def trace_path(p, w, edge, parent):
    """Return the nodes and edges on the path from node p to its ancestor
    w.
    """
    cycle_nodes = [p]
    cycle_edges = []

    while p != w:
        cycle_edges.append(edge[p])
        p = parent[p]
        cycle_nodes.append(p)

    return cycle_nodes, cycle_edges

@njit()
def find_cycle(i, p, q, size, edge, parent):
    """Return the nodes and edges on the cycle containing edge i == (p, q)
    when the latter is added to the spanning tree.
    The cycle is oriented in the direction from p to q.
    """
    w = find_apex(p, q, size, parent)
    cycle_nodes, cycle_edges = trace_path(p, w, edge, parent)
    cycle_nodes = np.array(cycle_nodes[::-1])
    cycle_edges = np.array(cycle_edges[::-1])

    if cycle_edges.shape[0] < 1:
        cycle_edges = np.concatenate((cycle_edges, np.array([i])))

    elif cycle_edges[0] != i:
        cycle_edges = np.concatenate((cycle_edges, np.array([i])))

    cycle_nodes_rev, cycle_edges_rev = trace_path(q, w, edge, parent)

    cycle_nodes = np.concatenate((cycle_nodes, np.int64(cycle_nodes_rev[:-1])))
    cycle_edges = np.concatenate((cycle_edges, np.int64(cycle_edges_rev)))

    return cycle_nodes, cycle_edges

@njit()
def residual_capacity(i, p, capac, flows, tails):
    """Return the residual capacity of an edge i in the direction away
    from its endpoint p.
    """
    if tails[np.int64(i)] == np.int64(p):
        return capac[np.int64(i)] - flows[np.int64(i)]

    else:
        return flows[np.int64(i)]

@njit()
def find_leaving_edge(cycle_nodes, cycle_edges, capac, flows, tails, heads):
    """Return the leaving edge in a cycle represented by cycle_nodes and
    cycle_edges.
    """
    cyc_edg_rev = np.flip(cycle_edges)
    cyc_nod_rev = np.flip(cycle_nodes)

    res_caps = []
    i = 0
    for edg in cyc_edg_rev:
        res_caps.append(residual_capacity(edg, cyc_nod_rev[i], capac, flows, tails))
        i += 1

    res_caps = np.array(res_caps)

    j = cyc_edg_rev[np.argmin(res_caps)]
    s = cyc_nod_rev[np.argmin(res_caps)]

    t = heads[np.int64(j)] if tails[np.int64(j)] == s else tails[np.int64(j)]
    return j, s, t

@njit()
def augment_flow(cycle_nodes, cycle_edges, f, tails, flows):
    """Augment f units of flow along a cycle represented by Wn and cycle_edges.
    """
    for i, p in zip(cycle_edges, cycle_nodes):
        if tails[int(i)] == np.int64(p):
            flows[int(i)] += f
        else:
            flows[int(i)] -= f

@njit()
def trace_subtree(p, last, next):
    """Yield the nodes in the subtree rooted at a node p.
    """
    tree = []
    tree.append(p)

    l = last[p]
    while p != l:
        p = next[p]
        tree.append(p)

    return np.array(tree, dtype=np.int64)

@njit()
def remove_edge(s, t, size, prev, last, next, parent, edge):
    """Remove an edge (s, t) where parent[t] == s from the spanning tree.
    """
    size_t = size[t]
    prev_t = prev[t]
    last_t = last[t]
    next_last_t = next[last_t]
    # Remove (s, t).
    parent[t] = -2
    edge[t] = -2
    # Remove the subtree rooted at t from the depth-first thread.
    next[prev_t] = next_last_t
    prev[next_last_t] = prev_t
    next[last_t] = t
    prev[t] = last_t

    # Update the subtree sizes and last descendants of the (old) ancestors
    # of t.
    while s != np.int64(-2):
        size[s] -= size_t
        if last[s] == last_t:
            last[s] = prev_t
        s = parent[s]

@njit()
def make_root(q, parent, size, last, prev, next, edge):
    """
    Make a node q the root of its containing subtree.
    """
    ancestors = []
    # -2 means node is checked
    while q != np.int64(-2):
        ancestors.append(q)
        q = parent[q]
    ancestors.reverse()

    ancestors_min_last = ancestors[:-1]
    next_ancs = ancestors[1:]

    for p, q in zip(ancestors_min_last, next_ancs):
        size_p = size[p]
        last_p = last[p]
        prev_q = prev[q]
        last_q = last[q]
        next_last_q = next[last_q]

        # Make p a child of q.
        parent[p] = q
        parent[q] = -2
        edge[p] = edge[q]
        edge[q] = -2
        size[p] = size_p - size[q]
        size[q] = size_p

        # Remove the subtree rooted at q from the depth-first thread.
        next[prev_q] = next_last_q
        prev[next_last_q] = prev_q
        next[last_q] = q
        prev[q] = last_q

        if last_p == last_q:
            last[p] = prev_q
            last_p = prev_q

        # Add the remaining parts of the subtree rooted at p as a subtree
        # of q in the depth-first thread.
        prev[p] = last_q
        next[last_q] = p
        next[last_p] = q
        prev[q] = last_p
        last[q] = last_p

@njit()
def add_edge(i, p, q, next, prev, last, size, parent, edge):
    """Add an edge (p, q) to the spanning tree where q is the root of a
    subtree.
    """
    last_p = last[p]
    next_last_p = next[last_p]
    size_q = size[q]
    last_q = last[q]
    # Make q a child of p.
    parent[q] = p
    edge[q] = i
    # Insert the subtree rooted at q into the depth-first thread.
    next[last_p] = q
    prev[q] = last_p
    prev[next_last_p] = last_q
    next[last_q] = next_last_p

    # Update the subtree sizes and last descendants of the (new) ancestors
    # of q.
    while p != np.int64(-2):
        size[p] += size_q
        if last[p] == last_p:
            last[p] = last_q
        p = parent[p]

@njit()
def update_potentials(i, p, q, heads, potentials, costs, last, next):
    """Update the potentials of the nodes in the subtree rooted at a node
    q connected to its parent p by an edge i.
    """
    if q == heads[i]:
        d = potentials[p] - costs[i] - potentials[q]
    else:
        d = potentials[p] + costs[i] - potentials[q]

    tree = trace_subtree(q, last, next)
    for q in tree:
        potentials[q] += d

@njit()
def network_simplex(source_labels, source_demands, sink_labels, sink_demands):
    '''
    This is a port of the network simplex algorithm implented by Loïc Séguin-C
    for the networkx package to allow acceleration via the numba package
    Copyright (C) 2010 Loïc Séguin-C. <loicseguin@gmail.com>
    All rights reserved.
    BSD license.
    References
    ----------
    .. [1] Z. Kiraly, P. Kovacs.
           Efficient implementation of minimum-cost flow algorithms.
           Acta Universitatis Sapientiae, Informatica 4(1):67--118. 2012.
    .. [2] R. Barr, F. Glover, D. Klingman.
           Enhancement of spanning tree labeling procedures for network
           optimization.
           INFOR 17(1):16--34. 1979.
    '''
    # Constant used throughout for conversions from floating point to integer
    fp_multiplier = np.array([1000000], dtype=np.int64)

    # Using numerical ordering is nice for indexing
    sources = np.arange(source_labels.shape[0]).astype(np.int64)
    sinks = np.arange(sink_labels.shape[0]).astype(np.int64) + source_labels.shape[0]

    # Add one additional node for a dummy source and sink
    nodes = np.arange(source_labels.shape[0] + sink_labels.shape[0]).astype(np.int64)

    # Multiply by a large number and cast to int to remove floating points
    source_d_fp = source_demands * fp_multiplier.astype(np.int64)
    source_d_int = source_d_fp.astype(np.int64)
    sink_d_fp = sink_demands * fp_multiplier.astype(np.int64)
    sink_d_int = sink_d_fp.astype(np.int64)

    # FP conversion error correction
    source_sum = np.sum(source_d_int)
    sink_sum = np.sum(sink_d_int)
    if  source_sum < sink_sum:
        source_ind = np.argmax(source_d_int)
        source_d_int[source_ind] += sink_sum - source_sum

    elif sink_sum < source_sum:
        sink_ind = np.argmax(sink_d_int)
        sink_d_int[sink_ind] += source_sum - sink_sum

    # Create demands array
    demands = np.concatenate((-source_d_int, sink_d_int)).astype(np.int64)

    # Create fully connected arcs between all sources and sinks
    conn_tails = np.array([i for i, x in enumerate(sources) for j, y in enumerate(sinks)], dtype=np.int64)
    conn_heads = np.array([j + sources.shape[0] for i, x in enumerate(sources) for j, y in enumerate(sinks)], dtype=np.int64)

    # Add arcs to and from the dummy node
    dummy_tails = []
    dummy_heads = []

    for node, demand in np.ndenumerate(demands):
        if demand > 0:
            dummy_tails.append(node[0])
            dummy_heads.append(-1)
        else:
            dummy_tails.append(-1)
            dummy_heads.append(node[0])

    # Concatenate these all together
    tails = np.concatenate((conn_tails, np.array(dummy_heads).T)).astype(np.int64)
    heads = np.concatenate((conn_heads, np.array(dummy_heads).T)).astype(np.int64)  # edge targets

    # Create costs and capacities for the arcs between nodes
    network_costs = np.array([abs(x - y) for x in source_labels for y in sink_labels], dtype=np.int64)
    network_capac = np.array([np.array([source_demands[i], sink_demands[j]]).min() for i, x in np.ndenumerate(sources) for j, y in np.ndenumerate(sinks)], dtype=np.float64) * fp_multiplier

    # TODO finish
    # If there is only one node on either side we can return capacity and costs
    # if sources.shape[0] == 1 or sinks.shape[0] == 1:
    #     tot_costs = np.array([cost * network_capac[i_ret] for i_ret, cost in np.ndenumerate(network_costs)], dtype=np.float64)
    #     return np.float64(np.sum(tot_costs))

    # inf_arr = (np.sum(network_capac.astype(np.int64)), np.sum(np.absolute(network_costs)), np.max(np.absolute(demands)))

    # Set a suitably high integer for infinity
    faux_inf = 3 * np.max(np.array((np.sum(network_capac.astype(np.int64)), np.sum(np.absolute(network_costs)), np.max(np.absolute(demands))), dtype=np.int64))

    # Add the costs and capacities to the dummy nodes
    costs = np.concatenate((network_costs, np.ones(nodes.shape[0]) * faux_inf)).astype(np.int64)
    capac = np.concatenate((network_capac, np.ones(nodes.shape[0]) * fp_multiplier)).astype(np.int64)

    # Construct the initial spanning tree.
    e = conn_tails.shape[0]
    n = nodes.shape[0]

    # Initialise zero flow in the connected arcs, and full flow to the dummy
    flows = np.concatenate((np.zeros(e), np.array([abs(d) for d in demands]))).astype(np.int64)

    # General arrays for the spanning tree
    potentials = np.array([faux_inf if d <= 0 else -faux_inf for d in demands]).T
    parent = np.concatenate((np.ones(n) * -1, np.array([-2]))).astype(np.int64)
    edge = np.arange(e, e+n).astype(np.int64)
    size = np.concatenate((np.ones(n), np.array([n + 1]))).astype(np.int64)
    next = np.concatenate((np.arange(1, n), np.array([-1, 0]))).astype(np.int64)
    prev = np.arange(-1, n)          # previous nodes in depth-first thread
    last = np.concatenate((np.arange(n), np.array([n - 1]))).astype(np.int64)     # last descendants in depth-first thread

    ###########################################################################
    # Main Pivot loop
    ###########################################################################

    f = 0

    while True:
        i, p, q, f = find_entering_edges(e, f, tails, heads, costs, potentials, flows)
        if p == -1: # If no entering edges then the optimal score is found
            break

        cycle_nodes, cycle_edges = find_cycle(i, p, q, size, edge, parent)
        j, s, t = find_leaving_edge(cycle_nodes, cycle_edges, capac, flows, tails, heads)
        augment_flow(cycle_nodes, cycle_edges, residual_capacity(j, s, capac, flows, tails), tails, flows)

        if i != j:  # Do nothing more if the entering edge is the same as the
                    # the leaving edge.
            if parent[t] != s:
                # Ensure that s is the parent of t.
                s, t = t, s

            if np.where(cycle_edges == i)[0][0] > np.where(cycle_edges == j)[0][0]:
                # Ensure that q is in the subtree rooted at t.
                p, q = q, p

            remove_edge(s, t, size, prev, last, next, parent, edge)
            make_root(q, parent, size, last, prev, next, edge)
            add_edge(i, p, q, next, prev, last, size, parent, edge)
            update_potentials(i, p, q, heads, potentials, costs, last, next)

    flow_cost = 0
    final_flows = flows[:e].astype(np.float64)
    edge_costs = costs[:e].astype(np.float64)

    # dot product is returning wrong values for some reason...
    for arc_ind, flow in np.ndenumerate(final_flows):
        flow_cost += flow * edge_costs[arc_ind]

    final = flow_cost / fp_multiplier
    # final_cost = np.sum(flow_cost) / fp_multiplier
    return final[0]

ElementDict = {'mendeleev': {'H': 91, 'D': 91, 'T': 91, 'He': 97, 'Li': 0, 
                            'Be': 6, 'B': 71, 'C': 76, 'N': 81, 'O': 86, 
                            'F': 92, 'Ne': 98, 'Na': 1, 'Mg': 7, 'Al': 72, 
                            'Si': 77, 'P': 82, 'S': 87, 'Cl': 93, 'Ar': 99, 
                            'K': 2, 'Ca': 8, 'Sc': 12, 'Ti': 44, 'V': 47, 
                            'Cr': 50, 'Mn': 53, 'Fe': 56, 'Co': 59, 'Ni': 62, 
                            'Cu': 65, 'Zn': 68, 'Ga': 73, 'Ge': 78, 'As': 83, 
                            'Se': 88, 'Br': 94, 'Kr': 100, 'Rb': 3, 'Sr': 9, 
                            'Y': 13, 'Zr': 45, 'Nb': 48, 'Mo': 51, 'Tc': 54, 
                            'Ru': 57, 'Rh': 60, 'Pd': 63, 'Ag': 66, 'Cd': 69, 
                            'In': 74, 'Sn': 79, 'Sb': 84, 'Te': 89, 'I': 95, 
                            'Xe': 101, 'Cs': 4, 'Ba': 10, 'La': 14, 'Ce': 16, 
                            'Pr': 17, 'Nd': 20, 'Pm': 22, 'Sm': 24, 'Eu': 26, 
                            'Gd': 28, 'Tb': 30, 'Dy': 32, 'Ho': 34, 'Er': 36, 
                            'Tm': 38, 'Yb': 40, 'Lu': 42, 'Hf': 46, 'Ta': 49, 
                            'W': 52, 'Re': 55, 'Os': 58, 'Ir': 61, 'Pt': 64, 
                            'Au': 67, 'Hg': 70, 'Tl': 75, 'Pb': 80, 'Bi': 85, 
                            'Po': 90, 'At': 96, 'Rn': 102, 'Fr': 5, 'Ra': 11, 
                            'Ac': 15, 'Th': 17, 'Pa': 19, 'U': 21, 'Np': 23, 
                            'Pu': 25, 'Am': 27, 'Cm': 29, 'Bk': 31, 'Cf': 33, 
                            'Es': 35, 'Fm': 37, 'Md': 39, 'No': 41, 'Lr': 43, 
                            'Rf': 0, 'Db': 0, 'Sg': 0, 'Bh': 0, 'Hs': 0, 
                            'Mt': 0, 'Ds': 0, 'Rg': 0, 'Cn': 0, 'Nh': 0, 
                            'Fl': 0, 'Mc': 0, 'Lv': 0, 'Ts': 0, 'Og': 0, 
                            'Uue': 0,
                            91: 'H', 91: 'D', 91: 'T', 97: 'He', 0: 'Li', 
                            6: 'Be', 71: 'B', 76: 'C', 81: 'N', 86: 'O', 
                            92: 'F', 98: 'Ne', 1: 'Na', 7: 'Mg', 72: 'Al', 
                            77: 'Si', 82: 'P', 87: 'S', 93: 'Cl', 99: 'Ar', 
                            2: 'K', 8: 'Ca', 12: 'Sc', 44: 'Ti', 47: 'V', 
                            50: 'Cr', 53: 'Mn', 56: 'Fe', 59: 'Co', 62: 'Ni', 
                            65: 'Cu', 68: 'Zn', 73: 'Ga', 78: 'Ge', 83: 'As', 
                            88: 'Se', 94: 'Br', 100: 'Kr', 3: 'Rb', 9: 'Sr', 
                            13: 'Y', 45: 'Zr', 48: 'Nb', 51: 'Mo', 54: 'Tc', 
                            57: 'Ru', 60: 'Rh', 63: 'Pd', 66: 'Ag', 69: 'Cd', 
                            74: 'In', 79: 'Sn', 84: 'Sb', 89: 'Te', 95: 'I', 
                            101: 'Xe', 4: 'Cs', 10: 'Ba', 14: 'La', 16: 'Ce', 
                            17: 'Pr', 20: 'Nd', 22: 'Pm', 24: 'Sm', 26: 'Eu', 
                            28: 'Gd', 30: 'Tb', 32: 'Dy', 34: 'Ho', 36: 'Er', 
                            38: 'Tm', 40: 'Yb', 42: 'Lu', 46: 'Hf', 49: 'Ta', 
                            52: 'W', 55: 'Re', 58: 'Os', 61: 'Ir', 64: 'Pt', 
                            67: 'Au', 70: 'Hg', 75: 'Tl', 80: 'Pb', 85: 'Bi', 
                            90: 'Po', 96: 'At', 102: 'Rn', 5: 'Fr', 11: 'Ra', 
                            15: 'Ac', 17: 'Th', 19: 'Pa', 21: 'U', 23: 'Np', 
                            25: 'Pu', 27: 'Am', 29: 'Cm', 31: 'Bk', 33: 'Cf', 
                            35: 'Es', 37: 'Fm', 39: 'Md', 41: 'No', 43: 'Lr', 
                            0: 'Rf', 0: 'Db', 0: 'Sg', 0: 'Bh', 0: 'Hs', 
                            0: 'Mt', 0: 'Ds', 0: 'Rg', 0: 'Cn', 0: 'Nh', 
                            0: 'Fl', 0: 'Mc', 0: 'Lv', 0: 'Ts', 0: 'Og', 
                            0: 'Uue'},
               'petti': {'H': 102, 'D': 102, 'T': 102, 'He': 0, 'Li': 11, 
                        'Be': 76, 'B': 85, 'C': 94, 'N': 99, 'O': 100, 
                        'F': 101, 'Ne': 1, 'Na': 10, 'Mg': 72, 'Al': 79, 
                        'Si': 84, 'P': 89, 'S': 93, 'Cl': 98, 'Ar': 2, 
                        'K': 9, 'Ca': 15, 'Sc': 19, 'Ti': 50, 'V': 53, 
                        'Cr': 56, 'Mn': 59, 'Fe': 60, 'Co': 63, 'Ni': 66, 
                        'Cu': 71, 'Zn': 75, 'Ga': 80, 'Ge': 83, 'As': 88, 
                        'Se': 92, 'Br': 97, 'Kr': 3, 'Rb': 8, 'Sr': 14, 
                        'Y': 18, 'Zr': 48, 'Nb': 51, 'Mo': 54, 'Tc': 57, 
                        'Ru': 62, 'Rh': 65, 'Pd': 68, 'Ag': 70, 'Cd': 74, 
                        'In': 78, 'Sn': 82, 'Sb': 87, 'Te': 91, 'I': 96, 
                        'Xe': 4, 'Cs': 7, 'Ba': 13, 'La': 32, 'Ce': 31, 
                        'Pr': 30, 'Nd': 29, 'Pm': 28, 'Sm': 27, 'Eu': 17, 
                        'Gd': 26, 'Tb': 25, 'Dy': 24, 'Ho': 23, 'Er': 22, 
                        'Tm': 21, 'Yb': 16, 'Lu': 20, 'Hf': 49, 'Ta': 52, 
                        'W': 55, 'Re': 58, 'Os': 61, 'Ir': 64, 'Pt': 67, 
                        'Au': 69, 'Hg': 73, 'Tl': 77, 'Pb': 81, 'Bi': 86, 
                        'Po': 90, 'At': 95, 'Rn': 5, 'Fr': 6, 'Ra': 12, 
                        'Ac': 47, 'Th': 46, 'Pa': 45, 'U': 44, 'Np': 43, 
                        'Pu': 42, 'Am': 41, 'Cm': 40, 'Bk': 39, 'Cf': 38, 
                        'Es': 37, 'Fm': 36, 'Md': 35, 'No': 34, 'Lr': 33, 
                        'Rf': 0, 'Db': 0, 'Sg': 0, 'Bh': 0, 'Hs': 0, 
                        'Mt': 0, 'Ds': 0, 'Rg': 0, 'Cn': 0, 'Nh': 0, 
                        'Fl': 0, 'Mc': 0, 'Lv': 0, 'Ts': 0, 'Og': 0, 
                        'Uue': 0,
                        102: 'H', 102: 'D', 102: 'T', 0: 'He', 11: 'Li', 
                        76: 'Be', 85: 'B', 94: 'C', 99: 'N', 100: 'O', 
                        101: 'F', 1: 'Ne', 10: 'Na', 72: 'Mg', 79: 'Al', 
                        84: 'Si', 89: 'P', 93: 'S', 98: 'Cl', 2: 'Ar', 
                        9: 'K', 15: 'Ca', 19: 'Sc', 50: 'Ti', 53: 'V', 
                        56: 'Cr', 59: 'Mn', 60: 'Fe', 63: 'Co', 66: 'Ni', 
                        71: 'Cu', 75: 'Zn', 80: 'Ga', 83: 'Ge', 88: 'As', 
                        92: 'Se', 97: 'Br', 3: 'Kr', 8: 'Rb', 14: 'Sr', 
                        18: 'Y', 48: 'Zr', 51: 'Nb', 54: 'Mo', 57: 'Tc', 
                        62: 'Ru', 65: 'Rh', 68: 'Pd', 70: 'Ag', 74: 'Cd', 
                        78: 'In', 82: 'Sn', 87: 'Sb', 91: 'Te', 96: 'I',
                        4: 'Xe', 7: 'Cs', 13: 'Ba', 32: 'La', 31: 'Ce', 
                        30: 'Pr', 29: 'Nd', 28: 'Pm', 27: 'Sm', 17: 'Eu', 
                        26: 'Gd', 25: 'Tb', 24: 'Dy', 23: 'Ho', 22: 'Er', 
                        21: 'Tm', 16: 'Yb', 20: 'Lu', 49: 'Hf', 52: 'Ta', 
                        55: 'W', 58: 'Re', 61: 'Os', 64: 'Ir', 67: 'Pt', 
                        69: 'Au', 73: 'Hg', 77: 'Tl', 81: 'Pb', 86: 'Bi', 
                        90: 'Po', 95: 'At', 5: 'Rn', 6: 'Fr', 12: 'Ra', 
                        47: 'Ac', 46: 'Th', 45: 'Pa', 44: 'U', 43: 'Np', 
                        42: 'Pu', 41: 'Am', 40: 'Cm', 39: 'Bk', 38: 'Cf', 
                        37: 'Es', 36: 'Fm', 35: 'Md', 34: 'No', 33: 'Lr', 
                        0: 'Rf', 0: 'Db', 0: 'Sg', 0: 'Bh', 0: 'Hs', 
                        0: 'Mt', 0: 'Ds', 0: 'Rg', 0: 'Cn', 0: 'Nh', 
                        0: 'Fl', 0: 'Mc', 0: 'Lv', 0: 'Ts', 0: 'Og', 
                        0: 'Uue'},

                'atomic': {'H': 0, 'D': 0, 'T': 0, 'He': 1, 'Li': 2, 'Be': 3, 
                          'B': 4, 'C': 5, 'N': 6, 'O': 7, 'F': 8, 'Ne': 9, 
                          'Na': 10, 'Mg': 11, 'Al': 12, 'Si': 13, 'P': 14, 
                          'S': 15, 'Cl': 16, 'Ar': 17, 'K': 18, 'Ca': 19, 
                          'Sc': 20, 'Ti': 21, 'V': 22, 'Cr': 23, 'Mn': 24, 
                          'Fe': 25, 'Co': 26, 'Ni': 27, 'Cu': 28, 'Zn': 29, 
                          'Ga': 30, 'Ge': 31, 'As': 32, 'Se': 33, 'Br': 34, 
                          'Kr': 35, 'Rb': 36, 'Sr': 37, 'Y': 38, 'Zr': 39, 
                          'Nb': 40, 'Mo': 41, 'Tc': 42, 'Ru': 43, 'Rh': 44, 
                          'Pd': 45, 'Ag': 46, 'Cd': 47, 'In': 48, 'Sn': 49, 
                          'Sb': 50, 'Te': 51, 'I': 52, 'Xe': 53, 'Cs': 54, 
                          'Ba': 55, 'La': 56, 'Ce': 57, 'Pr': 58, 'Nd': 59, 
                          'Pm': 60, 'Sm': 61, 'Eu': 62, 'Gd': 63, 'Tb': 64, 
                          'Dy': 65, 'Ho': 66, 'Er': 67, 'Tm': 68, 'Yb': 69, 
                          'Lu': 70, 'Hf': 71, 'Ta': 72, 'W': 73, 'Re': 74, 
                          'Os': 75, 'Ir': 76, 'Pt': 77, 'Au': 78, 'Hg': 79, 
                          'Tl': 80, 'Pb': 81, 'Bi': 82, 'Po': 83, 'At': 84,
                          'Rn': 85, 'Fr': 86, 'Ra': 87, 'Ac': 88, 'Th': 89,
                          'Pa': 90, 'U': 91, 'Np': 92, 'Pu': 93, 'Am': 94, 
                          'Cm': 95, 'Bk': 96, 'Cf': 97, 'Es': 98, 'Fm': 99, 
                          'Md': 100, 'No': 101, 'Lr': 102, 'Rf': 103, 'Db': 104, 
                          'Sg': 105, 'Bh': 106, 'Hs': 107, 'Mt': 108, 'Ds': 109, 
                          'Rg': 110, 'Cn': 111, 'Nh': 112, 'Fl': 113, 'Mc': 114, 
                          'Lv': 115, 'Ts': 116, 'Og': 117, 'Uue': 118,
                          0: 'H', 0: 'D', 0: 'T', 1: 'He', 2: 'Li', 3: 'Be', 
                          4: 'B', 5: 'C', 6: 'N', 7: 'O', 8: 'F', 9: 'Ne', 
                          10: 'Na', 11: 'Mg', 12: 'Al', 13: 'Si', 14: 'P', 
                          15: 'S', 16: 'Cl', 17: 'Ar', 18: 'K', 19: 'Ca', 
                          20: 'Sc', 21: 'Ti', 22: 'V', 23: 'Cr', 24: 'Mn', 
                          25: 'Fe', 26: 'Co', 27: 'Ni', 28: 'Cu', 29: 'Zn', 
                          30: 'Ga', 31: 'Ge', 32: 'As', 33: 'Se', 34: 'Br', 
                          35: 'Kr', 36: 'Rb', 37: 'Sr', 38: 'Y', 39: 'Zr', 
                          40: 'Nb', 41: 'Mo', 42: 'Tc', 43: 'Ru', 44: 'Rh', 
                          45: 'Pd', 46: 'Ag', 47: 'Cd', 48: 'In', 49: 'Sn', 
                          50: 'Sb', 51: 'Te', 52: 'I', 53: 'Xe', 54: 'Cs', 
                          55: 'Ba', 56: 'La', 57: 'Ce', 58: 'Pr', 59: 'Nd', 
                          60: 'Pm', 61: 'Sm', 62: 'Eu', 63: 'Gd', 64: 'Tb', 
                          65: 'Dy', 66: 'Ho', 67: 'Er', 68: 'Tm', 69: 'Yb', 
                          70: 'Lu', 71: 'Hf', 72: 'Ta', 73: 'W', 74: 'Re', 
                          75: 'Os', 76: 'Ir', 77: 'Pt', 78: 'Au', 79: 'Hg', 
                          80: 'Tl', 81: 'Pb', 82: 'Bi', 83: 'Po', 84: 'At', 
                          85: 'Rn', 86: 'Fr', 87: 'Ra', 88: 'Ac', 89: 'Th', 
                          90: 'Pa', 91: 'U', 92: 'Np', 93: 'Pu', 94: 'Am', 
                          95: 'Cm', 96: 'Bk', 97: 'Cf', 98: 'Es', 99: 'Fm', 
                          100: 'Md', 101: 'No', 102: 'Lr', 103: 'Rf', 104: 'Db', 
                          105: 'Sg', 106: 'Bh', 107: 'Hs', 108: 'Mt', 109: 'Ds', 
                          110: 'Rg', 111: 'Cn', 112: 'Nh', 113: 'Fl', 114: 'Mc', 
                          115: 'Lv', 116: 'Ts', 117: 'Og', 118: 'Uue'},
                
                'mod_petti': {'D': 102, 'T': 102, 'H': 102, 102: 'H', 
                                 0: 'He', 'He': 0, 11: 'Li', 'Li': 11, 76: 'Be', 
                                 'Be': 76, 85: 'B', 'B': 85, 86: 'C', 'C': 86, 
                                 87: 'N', 'N': 87, 96: 'O', 'O': 96, 101: 'F', 
                                 'F': 101, 1: 'Ne', 'Ne': 1, 10: 'Na', 'Na': 10, 
                                 72: 'Mg', 'Mg': 72, 77: 'Al', 'Al': 77, 84: 'Si', 
                                 'Si': 84, 88: 'P', 'P': 88, 95: 'S', 'S': 95, 
                                 100: 'Cl', 'Cl': 100, 2: 'Ar', 'Ar': 2, 9: 'K', 
                                 'K': 9, 15: 'Ca', 'Ca': 15, 47: 'Sc', 'Sc': 47, 
                                 50: 'Ti', 'Ti': 50, 53: 'V', 'V': 53, 54: 'Cr', 
                                 'Cr': 54, 71: 'Mn', 'Mn': 71, 70: 'Fe', 'Fe': 70, 
                                 69: 'Co', 'Co': 69, 68: 'Ni', 'Ni': 68, 67: 'Cu', 
                                 'Cu': 67, 73: 'Zn', 'Zn': 73, 78: 'Ga', 'Ga': 78, 
                                 83: 'Ge', 'Ge': 83, 89: 'As', 'As': 89, 94: 'Se', 
                                 'Se': 94, 99: 'Br', 'Br': 99, 3: 'Kr', 'Kr': 3, 
                                 8: 'Rb', 'Rb': 8, 14: 'Sr', 'Sr': 14, 20: 'Y', 
                                 'Y': 20, 48: 'Zr', 'Zr': 48, 52: 'Nb', 'Nb': 52, 
                                 55: 'Mo', 'Mo': 55, 58: 'Tc', 'Tc': 58, 60: 'Ru', 
                                 'Ru': 60, 62: 'Rh', 'Rh': 62, 64: 'Pd', 'Pd': 64, 
                                 66: 'Ag', 'Ag': 66, 74: 'Cd', 'Cd': 74, 79: 'In', 
                                 'In': 79, 82: 'Sn', 'Sn': 82, 90: 'Sb', 'Sb': 90, 
                                 93: 'Te', 'Te': 93, 98: 'I', 'I': 98, 4: 'Xe', 
                                 'Xe': 4, 7: 'Cs', 'Cs': 7, 13: 'Ba', 'Ba': 13, 
                                 31: 'La', 'La': 31, 30: 'Ce', 'Ce': 30, 29: 'Pr', 
                                 'Pr': 29, 28: 'Nd', 'Nd': 28, 27: 'Pm', 'Pm': 27, 
                                 26: 'Sm', 'Sm': 26, 16: 'Eu', 'Eu': 16, 25: 'Gd', 
                                 'Gd': 25, 24: 'Tb', 'Tb': 24, 23: 'Dy', 'Dy': 23, 
                                 22: 'Ho', 'Ho': 22, 21: 'Er', 'Er': 21, 19: 'Tm', 
                                 'Tm': 19, 17: 'Yb', 'Yb': 17, 18: 'Lu', 'Lu': 18, 
                                 49: 'Hf', 'Hf': 49, 51: 'Ta', 'Ta': 51, 56: 'W', 
                                 'W': 56, 57: 'Re', 'Re': 57, 59: 'Os', 'Os': 59, 
                                 61: 'Ir', 'Ir': 61, 63: 'Pt', 'Pt': 63, 65: 'Au', 
                                 'Au': 65, 75: 'Hg', 'Hg': 75, 80: 'Tl', 'Tl': 80, 
                                 81: 'Pb', 'Pb': 81, 91: 'Bi', 'Bi': 91, 92: 'Po', 
                                 'Po': 92, 97: 'At', 'At': 97, 5: 'Rn', 'Rn': 5, 
                                 6: 'Fr', 'Fr': 6, 12: 'Ra', 'Ra': 12, 32: 'Ac', 
                                 'Ac': 32, 33: 'Th', 'Th': 33, 34: 'Pa', 'Pa': 34, 
                                 35: 'U', 'U': 35, 36: 'Np', 'Np': 36, 37: 'Pu', 
                                 'Pu': 37, 38: 'Am', 'Am': 38, 39: 'Cm', 'Cm': 39, 
                                 40: 'Bk', 'Bk': 40, 41: 'Cf', 'Cf': 41, 42: 'Es', 
                                 'Es': 42, 43: 'Fm', 'Fm': 43, 44: 'Md', 'Md': 44, 
                                 45: 'No', 'No': 45, 46: 'Lr', 'Lr': 46, 'Rf': 0, 
                                 'Db': 0, 'Sg': 0, 'Bh': 0, 'Hs': 0, 'Mt': 0, 
                                 'Ds': 0, 'Rg': 0, 'Cn': 0, 'Nh': 0, 'Fl': 0, 
                                 'Mc': 0, 'Lv': 0, 'Ts': 0, 'Og': 0, 'Uue': 0}}

if __name__ == "__main__":
    main()