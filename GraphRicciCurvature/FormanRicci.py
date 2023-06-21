#!/usr/bin/env python3

# -*- coding: utf-8 -*-

"""

Created on Sun Jun 18 19:47:33 2023



@author: Yu Tian



Forman Ricci Curvature class

"""

#%%

###### FRC class ######

"""

A class to compute the Forman-Ricci curvature of a given NetworkX graph.

"""



# Author:

#       Yu Tian (https://ytian.netlify.app/)

#     Note that the code is largely based on the one by Chien-Chun Ni (http://www3.cs.stonybrook.edu/~chni/)

#     https://github.com/saibalmars/GraphRicciCurvature



# References for Chien-Chun Ni's code:

#     Forman. 2003. “Bochner’s Method for Cell Complexes and Combinatorial Ricci Curvature.”

#         Discrete & Computational Geometry 29 (3). Springer-Verlag: 323–74.

#     Sreejith, R. P., Karthikeyan Mohanraj, Jürgen Jost, Emil Saucan, and Areejit Samal. 2016.

#         “Forman Curvature for Complex Networks.” Journal of Statistical Mechanics: Theory and Experiment 2016 (6).

#         IOP Publishing: 063206.

#     Samal, A., Sreejith, R.P., Gu, J. et al.

#         "Comparative analysis of two discretizations of Ricci curvature for complex networks."

#         Scientific Report 8, 8650 (2018).



import math

import time



import networkx as nx

from functools import lru_cache



import multiprocessing as mp



from .util import logger, set_verbose



EPSILON = 1e-7  # to prevent divided by zero



# ---Shared global variables for multiprocessing used.---

_Gk = nx.Graph()

_weight = "weight"

_method = "augmented-quad"

_aug_method='area'

_proc = mp.cpu_count()

_cache_maxsize = 1000000

_nbr_topk = 3000



@lru_cache(_cache_maxsize)

def _get_ricci_curavture_1d(source, target):

    """Get the forman ricci Curvature in 1d

    """

    v1_nbr = set(_Gk.neighbors(source)) - {target}

    v2_nbr = set(_Gk.neighbors(target)) - {source}

    

    w_e = _Gk[source][target][_weight]

    w_v1 = _Gk.nodes[source][_weight]

    w_v2 = _Gk.nodes[target][_weight]

    ev1_sum = sum([w_v1 / math.sqrt(w_e * _Gk[source][node][_weight]) for node in v1_nbr])

    ev2_sum = sum([w_v2 / math.sqrt(w_e * _Gk[target][node][_weight]) for node in v2_nbr])

    

    result = w_e * (w_v1 / w_e + w_v2 / w_e - (ev1_sum + ev2_sum))



    return result



def _get_ricci_curvature_augmented(source, target):

    """Get the forman ricci Curvature augmented by triangular faces

    """

    v1_nbr = set(_Gk.neighbors(source)) - {target}

    v2_nbr = set(_Gk.neighbors(target)) - {source}



    face = v1_nbr & v2_nbr

    # prl_nbr = (v1_nbr | v2_nbr) - face



    w_e = _Gk[source][target][_weight]

    # Heron's formula

    sum_ef = 0

    if _aug_method == 'area':

        for node in face:

            w_e1 = _Gk[source][node][_weight]

            w_e2 = _Gk[target][node][_weight]

            s123 = (w_e + w_e1 + w_e2)/2

            # print('({}, {}, {})'.format(source, target, node))

            s2 = (s123-w_e)*(s123-w_e1)*(s123-w_e2)

            # only consider proper triangles

            if s2 > EPSILON:

                w_f = math.sqrt(s123*s2)

                sum_ef += w_e / w_f

            # else:

            #     w_f = math.sqrt(-s123*s2)

            #     sum_ef -= w_e / w_f

    elif _aug_method == 'sum':

        for node in face:

            w_f = (w_e + _Gk[source][node][_weight] + _Gk[target][node][_weight])/3

            sum_ef += w_e / w_f

    

    w_v1 = _Gk.nodes[source][_weight]

    w_v2 = _Gk.nodes[target][_weight]

    sum_ve = sum([w_v1 / w_e + w_v2 / w_e])



    sum_ehef = 0  # Always 0 for cycle = 3 case.

    sum_veeh = sum([w_v1 / math.sqrt(w_e * _Gk[source][node][_weight]) for node in (v1_nbr - face)] +

                   [w_v2 / math.sqrt(w_e * _Gk[target][node][_weight]) for node in (v2_nbr - face)])



    result = w_e * (sum_ef + sum_ve - math.fabs(sum_ehef - sum_veeh))



    return result



def _get_ricci_curvature_augmented_quad(source, target):

    """Get the forman ricci Curvature augmented by both triangular and quadrangular faces

    """

    v1_nbr = set(_Gk.neighbors(source)) - {target}

    v2_nbr = set(_Gk.neighbors(target)) - {source}



    face = v1_nbr & v2_nbr

    sum_ef = 0

    w_e = _Gk[source][target][_weight]

    

    # Heron's formula

    if _aug_method == 'area':

        for node in face:

            w_e1 = _Gk[source][node][_weight]

            w_e2 = _Gk[target][node][_weight]

            s123 = (w_e + w_e1 + w_e2)/2

            s2 = (s123-w_e)*(s123-w_e1)*(s123-w_e2)

            if s2 > EPSILON:

                w_f = math.sqrt(s123*s2)

                sum_ef += w_e / w_f

            # else:

            #     w_f = math.sqrt(-s123*s2)

            #     sum_ef -= w_e / w_f

    elif _aug_method == 'sum':

        for node in face:

            w_f = (w_e + _Gk[source][node][_weight] + _Gk[target][node][_weight])/3

            sum_ef += w_e / w_f



    w_v1 = _Gk.nodes[source][_weight]

    w_v2 = _Gk.nodes[target][_weight]

    sum_ve = sum([w_v1 / w_e + w_v2 / w_e])



    sum_ehef = 0  # initialisation (0 when cycle = 3, but now we have quadrangles)

    prl_nbr1 = v1_nbr - face # no diagonal edges

    prl_nbr2 = v2_nbr - face

    for v1n in prl_nbr1:

        v1n_nbr = set(_Gk.neighbors(v1n))

        # v1n_nbr.remove(source)

        quad = v1n_nbr & prl_nbr2 # (source, v1n, quad, target) quadrangles

        # rec_count += len(quad)

        if _aug_method == 'area':

            for v2n in quad:

                # compute the weight

                w_e1 = _Gk[source][v1n][_weight]

                w_e2 = _Gk[target][v2n][_weight]

                w_ep = _Gk[v1n][v2n][_weight]

                # construct quadrangle by sorting the edges

                w_es = sorted([w_e, w_e1, w_e2, w_ep], reverse=True) # descending order

                # compute the area

                if (w_es[0] == w_es[1]) and (w_es[2] == w_es[3]): # rectangle

                    w_fq = w_es[0]*w_es[2]

                    sum_ef += w_e/w_fq

                    sum_ehef += math.sqrt(w_e*w_ep)/w_fq

                    # rec_weis.append(w_fq)

                else:

                    w_t1 = w_es[0]-w_es[1]

                    w_t2 = w_es[2]

                    w_t3 = w_es[3]

                    s123 = (w_t1 + w_t2 + w_t3)/2

                    s2 = (s123-w_t1)*(s123-w_t2)*(s123-w_t3)

                    # only consider proper triangles

                    if s2 > EPSILON:

                        s_tri = math.sqrt(s123*s2)

                        w_fq = s_tri*(1 + 2*w_es[1]/w_t1)

                        sum_ef += w_e/w_fq

                        sum_ehef += math.sqrt(w_e*w_ep)/w_fq

                        # rec_weis.append(w_fq)

        elif _aug_method == 'sum':

            for v2n in quad:

                # compute the weight

                w_e1 = _Gk[source][v1n][_weight]

                w_e2 = _Gk[target][v2n][_weight]

                w_ep = _Gk[v1n][v2n][_weight]

                w_fq = (w_e + w_e1 + w_e2 + w_ep)/4

                sum_ef += w_e/w_fq

                sum_ehef += math.sqrt(w_e*w_ep)/w_fq

                # rec_weis.append(w_fq)



    sum_veeh = sum([w_v1 / math.sqrt(w_e * _Gk[source][v][_weight]) for v in (v1_nbr - face)] +

                    [w_v2 / math.sqrt(w_e * _Gk[target][v][_weight]) for v in (v2_nbr - face)])



    result = w_e * (sum_ef + sum_ve - math.fabs(sum_ehef - sum_veeh))

    return result





def _compute_ricci_curvature_single_edge(source, target):

    """Ricci curvature computation for a given single edge.

    Parameters

    ----------

    source : int

        Source node index in Networkit graph `_Gk`.

    target : int

        Target node index in Networkit graph `_Gk`.

    Returns

    -------

    result : dict[(int,int), float]

        The Ricci curvature of given edge in dict format. E.g.: {(node1, node2): ricciCurvature}

    """

    # logger.debug("EDGE:%s,%s"%(source,target))

    assert source != target, "Self loop is not allowed."  # to prevent self loop



    # If the weight of edge is too small, return 0 instead.

    if _Gk[source][target][_weight] < EPSILON:

        logger.trace("Zero weight edge detected for edge (%s,%s), return Ricci Curvature as 0 instead." %

                       (source, target))

        return {(source, target): 0}



    # compute FRC

    result = 1  # assign an initial cost

    assert _method in ["1d", "augmented", "augmented-quad"], \

        'Method %s not found, support method:["1d", "augmented", "augmented-quad"]' % _method

    if _method == "1d":

        result = _get_ricci_curavture_1d(source, target)

    elif _method == "augmented":

        result = _get_ricci_curvature_augmented(source, target)

    elif _method == "augmented-quad":

        result = _get_ricci_curvature_augmented_quad(source, target)



    logger.debug("Ricci curvature (%s,%s) = %f" % (source, target, result))



    return {(source, target): result}





def _wrap_compute_single_edge(stuff):

    """Wrapper for args in multiprocessing."""

    return _compute_ricci_curvature_single_edge(*stuff)





def _compute_ricci_curvature_edges(G: nx.Graph, weight="weight", edge_list=[],

                                   method="augmented-quad", aug_method='area',

                                   proc=mp.cpu_count(), chunksize=None, 

                                   cache_maxsize=1000000, nbr_topk=3000):

    """Compute Ricci curvature for edges in  given edge lists.

    """



    logger.trace("Number of nodes: %d" % G.number_of_nodes())

    logger.trace("Number of edges: %d" % G.number_of_edges())



    if not nx.get_edge_attributes(G, weight):

        logger.info('Edge weight not detected in graph, use "weight" as default edge weight.')

        for (v1, v2) in G.edges():

            G[v1][v2][weight] = 1.0

    if not nx.get_node_attributes(G, weight):

            logger.info('Node weight not detected in graph, use "weight" as default node weight.')

            for v in G.nodes():

                G.nodes[v][weight] = 1.0

    if G.is_directed():

        logger.info("Forman-Ricci curvature is not supported for directed graph yet, "

                    "covert input graph to undirected.")

        G = G.to_undirected()



    # ---set to global variable for multiprocessing used.---

    global _Gk

    global _weight

    global _method

    global _aug_method

    global _proc

    global _cache_maxsize

    global _nbr_topk

    # -------------------------------------------------------



    # Construct nx to nk dictionary

    nx2nk_ndict, nk2nx_ndict = {}, {}

    for idx, n in enumerate(G.nodes()):

        nx2nk_ndict[n] = idx

        nk2nx_ndict[idx] = n



    # _Gk = nk.nxadapter.nx2nk(G, weightAttr=weight)

    _Gk = nx.relabel_nodes(G, mapping=nx2nk_ndict)

    _weight = weight

    _method = method

    _aug_method = aug_method

    _proc = proc

    _cache_maxsize = cache_maxsize

    _nbr_topk = nbr_topk



    if edge_list:

        args = [(nx2nk_ndict[source], nx2nk_ndict[target]) for source, target in edge_list]

    else:

        args = [(nx2nk_ndict[source], nx2nk_ndict[target]) for source, target in G.edges()]



    # Start compute edge Ricci curvature

    t0 = time.time()



    with mp.get_context('fork').Pool(processes=_proc) as pool:

        # WARNING: Now only fork works, spawn will hang.



        # Decide chunksize following method in map_async

        if chunksize is None:

            chunksize, extra = divmod(len(args), proc * 4)

            if extra:

                chunksize += 1



        # Compute Ricci curvature for edges

        result = pool.imap_unordered(_wrap_compute_single_edge, args, chunksize=chunksize)

        pool.close()

        pool.join()



    # Convert edge index from nk back to nx for final output

    output = {}

    for rc in result:

        for k in list(rc.keys()):

            output[(nk2nx_ndict[k[0]], nk2nx_ndict[k[1]])] = rc[k]



    logger.info("%8f secs for Forman Ricci curvature computation." % (time.time() - t0))



    return output



def _compute_ricci_curvature(G: nx.Graph, weight="weight", **kwargs):

    """Compute Forman Ricci curvature of edges and nodes.

    The node Ricci curvature is defined as the average of node's adjacency edges.

    Parameters

    ----------

    G : NetworkX graph

        A given directional or undirectional NetworkX graph.

    weight : str

        The edge weight used to compute Ricci curvature. (Default value = "weight")

    **kwargs

        Additional keyword arguments passed to `_compute_ricci_curvature_edges`.

    Returns

    -------

    G: NetworkX graph

        A NetworkX graph with "forman" on nodes and edges.

    """



    # compute Ricci curvature for all edges

    edge_ricci = _compute_ricci_curvature_edges(G, weight=weight, **kwargs)



    # Assign edge Ricci curvature from result to graph G

    nx.set_edge_attributes(G, edge_ricci, "forman")



    # Compute node Ricci curvature

    for n in G.nodes():

        rc_sum = 0  # sum of the neighbor Ricci curvature

        if G.degree(n) != 0:

            for nbr in G.neighbors(n):

                if 'forman' in G[n][nbr]:

                    rc_sum += G[n][nbr]['forman']



            # Assign the node Ricci curvature to be the average of node's adjacency edges

            G.nodes[n]['forman'] = rc_sum / G.degree(n)

            logger.debug("node %s, Forman Ricci Curvature = %f" % (n, G.nodes[n]['forman']))



    return G



def _compute_ricci_flow(G: nx.Graph, weight="weight",

                        iterations=10, fac=1.1, delta=1e-4, surgery=(lambda G, *args, **kwargs: G, 100),

                        **kwargs

                        ):

    """Compute the given Ricci flow metric of each edge of a given connected NetworkX graph.

    Parameters

    ----------

    G : NetworkX graph

        A given directional or undirectional NetworkX graph.

    weight : str

        The edge weight used to compute Ricci curvature. (Default value = "weight")

    iterations : int

        Iterations to require Ricci flow metric. (Default value = 20)

    fac : float

        factor in the denomenator of the step for gradient decent process. (Default value = 1)

    delta : float

        process stop when difference of Ricci curvature is within delta. (Default value = 1e-4)

    surgery : (function, int)

        A tuple of user define surgery function that will execute every certain iterations.

        (Default value = (lambda G, *args, **kwargs: G, 100))

    Returns

    -------

    G: NetworkX graph

        A NetworkX graph with ``weight`` as Ricci flow metric.

    """



    if not nx.is_connected(G):

        logger.info("Not connected graph detected, compute on the largest connected component instead.")

        G = nx.Graph(G.subgraph(max(nx.connected_components(G), key=len)))



    # Set normalized weight to be the number of edges.

    normalized_weight = float(G.number_of_edges())



    

    # Start compute edge Ricci flow

    # t0 = time.time()



    if nx.get_edge_attributes(G, "original_RC"):

        logger.info("original_RC detected, continue to refine the ricci flow.")

        # print("original_RC detected, continue to refine the ricci flow.")

        

    else:

        logger.info("No forman detected, compute original_RC...")

        # print("No forman detected, compute original_RC...")

        G = _compute_ricci_curvature(G, weight=weight, **kwargs)



        for (v1, v2) in G.edges():

            G[v1][v2]["original_RC"] = G[v1][v2]["forman"]



        

    # obtain the step size

    rc = nx.get_edge_attributes(G, "original_RC")

    rc_max = max(rc.values())

    rc_min = min(rc.values())

    step = 1/(max([rc_max, abs(rc_min)])*fac)

    logger.info(" === Initial step size %f === " % step)

    



    # Start the Ricci flow process

    for i in range(iterations):

        for (v1, v2) in G.edges():

            G[v1][v2][weight] -= step * (G[v1][v2]["forman"]) * G[v1][v2][weight]



        # Do normalization on all weight to prevent weight expand to infinity

        w = nx.get_edge_attributes(G, weight)

        sumw = sum(w.values())

        for k, v in w.items():

            w[k] = w[k] * (normalized_weight / sumw)

        nx.set_edge_attributes(G, values=w, name=weight)

        logger.info(" === Ricci flow iteration %d === " % i)

        logger.info("Current step size %f" % step)

        # print(" === Ricci flow iteration %d === " % i)

        # print("Current step size %f" % step)



        G = _compute_ricci_curvature(G, weight=weight, **kwargs)



        rc = nx.get_edge_attributes(G, "forman")

        rc_max = max(rc.values())

        rc_min = min(rc.values())

        step = 1/(max([rc_max, abs(rc_min)])*fac)

        diff = max(rc.values()) - min(rc.values())



        logger.trace("Ricci curvature difference: %f" % diff)

        logger.trace("max:%f, min:%f | maxw:%f, minw:%f" % (

            max(rc.values()), min(rc.values()), max(w.values()), min(w.values())))

        # print("Ricci curvature difference: %f" % diff)

        # print("max:%f, min:%f | maxw:%f, minw:%f" % (

            # max(rc.values()), min(rc.values()), max(w.values()), min(w.values())))



        if diff < delta:

            logger.trace("Ricci curvature converged, process terminated.")

            # print("Ricci curvature converged, process terminated.")

            break



        # do surgery or any specific evaluation

        surgery_func, do_surgery = surgery

        if i != 0 and i % do_surgery == 0:

            G = surgery_func(G, weight)

            normalized_weight = float(G.number_of_edges())



        for n1, n2 in G.edges():

            logger.debug("%s %s %s" % (n1, n2, G[n1][n2]))



    # logger.info("%8f secs for Ricci flow computation." % (time.time() - t0))



    return G





class FormanRicci:

    def __init__(self, G: nx.Graph, weight="weight", method="augmented-quad", aug_method='area', 

                 proc=mp.cpu_count(), chunksize=None, cache_maxsize=1000000,

                 nbr_topk=3000, verbose="ERROR"):

        """A class to compute Forman-Ricci curvature for all nodes and edges in G.

        Parameters

        ----------

        G : NetworkX graph

            A given NetworkX graph, unweighted graph only for now, edge weight will be ignored.

        weight : str

            The edge weight used to compute Ricci curvature. (Default value = "weight")

        method : {"1d", "augmented"}

            The method used to compute Forman-Ricci curvature. (Default value = "augmented")

            - "1d": Computed with 1-dimensional simplicial complex (vertex, edge).

            - "augmented": Computed with 2-dimensional simplicial complex, length <=3 (vertex, edge, face).

        verbose: {"INFO","DEBUG","ERROR"}

            Verbose level. (Default value = "ERROR")

                - "INFO": show only iteration process log.

                - "DEBUG": show all output logs.

                - "ERROR": only show log if error happened.

        """



        self.G = G.copy()

        self.weight = weight

        self.method = method

        self.aug_method = aug_method

        self.proc = proc

        self.chunksize = chunksize

        self.cache_maxsize = cache_maxsize

        self.nbr_topk = nbr_topk



        if not nx.get_edge_attributes(self.G, self.weight):

            logger.info('Edge weight not detected in graph, use "weight" as default edge weight.')

            for (v1, v2) in self.G.edges():

                self.G[v1][v2][self.weight] = 1.0

        if not nx.get_node_attributes(self.G, self.weight):

            logger.info('Node weight not detected in graph, use "weight" as default node weight.')

            for v in self.G.nodes():

                self.G.nodes[v][self.weight] = 1.0

        if self.G.is_directed():

            logger.info("Forman-Ricci curvature is not supported for directed graph yet, "

                        "covert input graph to undirected.")

            self.G = self.G.to_undirected()



        # self_loop_edges = list(nx.selfloop_edges(self.G))

        # if self_loop_edges:

        #     logger.info('Self-loop edge detected. Removing %d self-loop edges.' % len(self_loop_edges))

        #     self.G.remove_edges_from(self_loop_edges)



        # set_verbose(verbose)

    

    def set_verbose(self, verbose):

        """Set the verbose level for this process.

        Parameters

        ----------

        verbose : {"INFO", "TRACE","DEBUG","ERROR"}

            Verbose level. (Default value = "ERROR")

                - "INFO": show only iteration process log.

                - "TRACE": show detailed iteration process log.

                - "DEBUG": show all output logs.

                - "ERROR": only show log if error happened.

        """

        set_verbose(verbose)



    def compute_ricci_curvature_edges(self, edge_list=None):

        """Compute Forman Ricci curvature for edges in given edge lists.

        Parameters

        ----------

        edge_list : list of edges

            The list of edges to compute Ricci curvature, set to [] to run for all edges in G. (Default value = [])

        Returns

        -------

        output : dict[(int,int), float]

            A dictionary of edge Ricci curvature. E.g.: {(node1, node2): ricciCurvature}.

        """

        return _compute_ricci_curvature_edges(G=self.G, weight=self.weight, edge_list=edge_list,

                                              method=self.method, aug_method=self.aug_method,

                                              proc=self.proc, chunksize=self.chunksize,

                                              cache_maxsize=self.cache_maxsize, nbr_topk=self.nbr_topk)



    def compute_ricci_curvature(self):

        """Compute Ricci curvature of edges and nodes.

        The node Ricci curvature is defined as the average of node's adjacency edges.

        Returns

        -------

        G: NetworkX graph

            A NetworkX graph with "ricciCurvature" on nodes and edges.

        Examples

        --------

        To compute the Ollivier-Ricci curvature for karate club graph::

            >>> G = nx.karate_club_graph()

            >>> orc = OllivierRicci(G, alpha=0.5, verbose="INFO")

            >>> orc.compute_ricci_curvature()

            >>> orc.G[0][1]

            {'weight': 1.0, 'ricciCurvature': 0.11111111071683011}

        """



        self.G = _compute_ricci_curvature(G=self.G, weight=self.weight, 

                                          method=self.method, aug_method=self.aug_method,

                                          proc=self.proc, chunksize=self.chunksize,

                                          cache_maxsize=self.cache_maxsize, nbr_topk=self.nbr_topk)

        return self.G



    

    def compute_ricci_flow(self, iterations=10, fac=1.1, delta=1e-4, surgery=(lambda G, *args, **kwargs: G, 100)):

        """Compute the given Forman Ricci flow metric of each edge of a given connected NetworkX graph.

        Parameters

        ----------

        iterations : int

            Iterations to require Ricci flow metric. (Default value = 10)

        step : float

            Step size for gradient decent process. (Default value = 1)

        delta : float

            Process stop when difference of Ricci curvature is within delta. (Default value = 1e-4)

        surgery : (function, int)

            A tuple of user define surgery function that will execute every certain iterations.

            (Default value = (lambda G, *args, **kwargs: G, 100))

        Returns

        -------

        G: NetworkX graph

            A graph with ``weight`` as Ricci flow metric.

        Examples

        --------

        To compute the Forman-Ricci flow for karate club graph::

            >>> G = nx.karate_club_graph()

            >>> frc = FormanRicci(G, verbose="INFO")

            >>> frc.compute_ricci_flow(iterations=10)

            >>> frc.G[0][1]

            {'weight': 0.06399135316908759,

             'ricciCurvature': 0.18608249978652802,

             'original_RC': 0.11111111071683011}

        """

        self.G = _compute_ricci_flow(G=self.G, weight=self.weight,

                                     iterations=iterations, fac=fac, delta=delta, surgery=surgery,

                                     method=self.method, aug_method=self.aug_method,

                                     proc=self.proc, chunksize=self.chunksize,

                                     cache_maxsize=self.cache_maxsize, nbr_topk=self.nbr_topk)

        return self.G 
