"""author: Connor Stone

summary:
  Nodes which encapsulate or act on other nodes or full flowcharts.

description:
  A "Chart" Node is a container for a whole flowchart (in this way
  flowcharts can be nested) and is the primary interface for building
  flowcharts. A "Pipe" Node is a wrapper for a flowchart (or node)
  which can iterate on the state and apply the flowchart (or node) in
  parallel, itteratively, or simply pass the state along. Similarly,
  other nodes in this module ought to have behaviour which acts on
  nodes or flowcharts.
"""

import pygraphviz as pgv
from pickle import dumps, loads
from multiprocessing import Pool
from copy import deepcopy
from .core import Node
from .first_order_nodes import Start, End, Process, Decision
from datetime import datetime
from time import time
import traceback
import logging


class Chart(Node):
    """Main container for a flowchart.

    Stores all the nodes and links between them composing a
    flowchart. The run method for this object will iteratively apply
    each node in the flowchart and progress through the path from
    start to end. This is the main object that users should interact
    with when constructing a flowchart. Includes methods to add/link
    nodes, draw the flowchart, save/load the flowchart, and visualize
    it. This class inherits from the Node object and so can itself
    be a node in another larger flowchart. In this case it will be
    represented visually as a hexagon.

    Arguments
    -----------------
    name: string
      name of the node, should be unique in the flowchart. This is how
      other nodes (i.e. decision nodes) will identify the node.

    filename: string
      path to a file containing a saved flowchart. This will be loaded
      and used to initialize the current flowchart. Note that the user
      can still set the name of the flowchart to whatever they like.

      :default:
        None
    """

    def __init__(self, name=None, filename=None, logfile=None, safe_mode=False):
        if isinstance(logfile, str):
            logging.basicConfig(
                filename=logfile, filemode="w", level=logging.INFO
            )
        if isinstance(filename, str):
            res = self.load(filename)
            super().__init__(res["name"])
            self.__dict__.update(res)
            return

        super().__init__(name)
        self.nodes = {}
        self.safe_mode = safe_mode
        self.structure_dict = {}
        self._linear_mode = False
        self._linear_mode_link = "start"
        self.current_node = "start"
        self.add_node(Start())
        self.add_node(End())
        self.path = []
        self.benchmarks = []
        self.istidy = False
        self.visual_kwargs['shape'] = "hexagon"

    def linear_mode(self, mode):
        """Activate a mode where new nodes are automatically added to the end
        of the flowchart. This way a simple chart without a complex
        decision structure can be constructed with minimal redundant
        linking.

        Arguments
        -----------------
        mode: bool
          If True, linear mode will be turned on. If False, it will be
          turned off

        """
        if mode and not self._linear_mode:
            self._linear_mode = True
            while not self.nodes[self._linear_mode_link].forward is None:
                prev = self._linear_mode_link
                self._linear_mode_link = self.nodes[self._linear_mode_link].forward.name
            if self._linear_mode_link == "end":
                self._linear_mode_link = prev
            else:
                self.link_nodes(self._linear_mode_link, "end")
        elif not mode and self._linear_mode:
            self._linear_mode = False

    def add_node(self, node):
        """Add a new Node to the flowchart. This merely makes the flowchart
        aware of the Node, it will need to be linked in order to
        take part in the calculation (unless linear mode is on).

        Arguments
        -----------------
        node: Node
          A Node object to be added to the flowchart.
        """
        self.nodes[node.name] = node
        self.structure_dict[node.name] = []
        self.istidy = False
        if self._linear_mode:
            self.unlink_nodes(self._linear_mode_link, "end")
            self.link_nodes(self._linear_mode_link, node.name)
            self.link_nodes(node.name, "end")
            self._linear_mode_link = node.name

    def add_process_node(self, name, func=None):
        """Utility wrapper to first create a process object then add it to
        the flowchart with the "add_node" method.

        Arguments
        -----------------
        name: string
          name of the node, should be unique in the flowchart. This is how
          other nodes (i.e. decision nodes) will identify the node.

        func: function
          function object of the form: func(state) returns state. This can
          be given on initialization to set the behavior of the node in
          the flowchart. This function should operate on the state and
          return the new updated state object.

        :default:
          None
        """
        newprocess = Process(name, func)
        self.add_node(newprocess)

    def add_decision_node(self, name, func=None):
        """Utility wrapper to first create a decision object then add it to
        the flowchart with the "add_node" method.

        Arguments
        -----------------
        name: string
          name of the node, should be unique in the flowchart. This is how
          other nodes (i.e. decision nodes) will identify the node.

        func: function
          function object of the form: func(state) returns state. This can
          be given on initialization to set the behavior of the node in
          the flowchart. This function should operate on the state and
          return the new updated state object.

        :default:
          None
        """
        newdecision = Decision(name, func)
        self.add_node(newdecision)

    def link_nodes(self, node1, node2):
        """Link two nodes in the flowchart. node1 will be linked forward to node2.

        Arguments
        -----------------
        node1: string
          A Node name in the flowchart which will be linked
          forward to node2.

        node2: string
          A Node name in the flowchart which will have node1 linked to it.
        """
        self.nodes[node1].link_forward(self.nodes[node2])
        self.structure_dict[node1].append(node2)
        self.istidy = False

    def unlink_nodes(self, node1, node2):
        """Undo the operations of "link_nodes" and return to previous state.

        Arguments
        -----------------
        node1: string
          A Node name in the flowchart which was linked forward to node2.

        node2: string
          A Node name in the flowchart which did have node1 linked to it.
        """
        self.nodes[node1].unlink_forward(self.nodes[node2])
        self.structure_dict[node1].pop(self.structure_dict[node1].index(node2))
        self.istidy = False

    def insert_node(self, node1, node2):
        """Insert node1 in the place of node2, and link to node2

        Arguments
        -----------------
        node1: string
          A Node name in the flowchart which will take the place of node2.

        node2: string
          A Node name in the flowchart which will now come after node1.
        """

        for reverse_node in list(self.nodes[node2].reverse):
            self.unlink_nodes(reverse_node.name, node2)
            self.link_nodes(reverse_node.name, node1)
        self.link_nodes(node1, node2)

    def build_chart(self, nodes=[], structure={}):
        """Compact way to build a chart.

        Through this function a user may supply all necessary
        information to construct a flowchart. A list of nodes can be
        added to the chart instead of adding them one at a time. Also
        a structure dictionary can be provided which gives all of the
        linkages between nodes. Essentially this function just
        condenses a number of "add_node" and "link_nodes" calls into a
        single operation. This function may be called multiple times,
        each call will add to the previous, not replace it.

        Arguments
        -----------------
        nodes: list
          A list of Node objects to add to the flowchart. These will
          be added one at a time in the order provided, thus if
          "linear mode" is on then each one will be appended to the
          end of the flowchart.

        structure: dict
          A dictonary that gives the structure of the flowchart. The
          keys in the dictionary are the name strings of nodes, the
          values can be either name strings or lists of name
          strings. The key will be linked forward to the value(s).
        """
        for node in nodes:
            self.add_node(node)

        if isinstance(structure, list):
            for node1, node2 in zip(structure[:-1], structure[1:]):
                self.link_nodes(node1, node2)
            if not structure[0] == "start":
                self.link_nodes("start", structure[0])
            if not structure[-1] == "end":
                self.link_nodes(structure[-1], "end")
        else:
            for node1 in structure.keys():
                if isinstance(structure[node1], str):
                    self.link_nodes(node1, structure[node1])
                else:
                    for node2 in structure[node1]:
                        self.link_nodes(node1, node2)

    def draw(self, filename):
        """Visual representation of the flowchart.

        Creates a visual flowchart using pygraphviz. Every node will
        be drawn, including those that don't have links to other
        nodes, make sure to fully input the desired structure before
        running this method.

        Arguments
        -----------------
        filename: string
          path to save final graphical representation. Should end in
          .png, .jpg, etc.
        """
        visual = self._construct_chart_visual()
        visual.layout()
        visual.draw(filename)

    def save(self, filename):
        """Save the flowchart to file.

        Applies pickling to the core information in the flowchart and
        saves to a given file location. Some python objects cannot be
        pickled and so cannot be saved this way. The user may need to
        write a specialized save function for such structures.

        Arguments
        -----------------
        filename: string
          path to save current flowchart to.
        """
        with open(filename, "wb") as flowchart_file:
            flowchart_file.write(dumps(self.__dict__))

    def load(self, filename):
        """Loads the flowchart representation.

        Reads a pickle file as created by "save" to reconstruct a
        saved flowchart. This function should generally not be
        accessed by the user, instead provide a filename when
        initializing the flowchart and the loading will be handled
        properly. In case you wish to use load directly, it returns a
        dictionary of all the class structures/methods/variables.

        Arguments
        -----------------
        filename: string
          path to load flowchart from.
        """
        with open(filename, "rb") as flowchart_file:
            res = loads(flowchart_file.read())
        return res

    def _tidy_ends(self):
        for name, node in self.nodes.items():
            if name == "end":
                continue
            if node.forward is None:
                RuntimeWarning(f"{name} is undirected, linking to 'end' node")
                self.link_nodes(name, "end")
        self.istidy = True

    def _run(self, state):
        assert (
            not self.nodes["start"].forward is None
        ), "chart has no structure! start must be linked to a node"
        if not self.istidy:
            self._tidy_ends()
        for node in self:
            logging.info(f"{self.name}: {node.name} ({datetime.now()})")
            self.path.append(node.name)
            if self.safe_mode:
                try:
                    state = node(state)
                    self.benchmarks.append(node.benchmark)
                except Exception as e:
                    logging.error(f"on step '{self.current_node}' got error: {str(e)}")
                    logging.error("with full trace: %s" % traceback.format_exc())
                    self.benchmarks.append(np.nan)
            else:
                state = node(state)
                self.benchmarks.append(node.benchmark)
        return state

    def _construct_chart_visual(self):
        if not self.istidy:
            self._tidy_ends()
        visual = pgv.AGraph(strict=True, directed=True, splines="line", overlap=False)
        for node in self.nodes.values():
            visual.add_node(
                node.name, **node.visual_kwargs
            )
        for node1, links in self.structure_dict.items():
            for node2 in links:
                visual.add_edge(node1, node2)
        return visual

    def __str__(self):
        visual = self._construct_chart_visual()
        return visual.string()

    def __iter__(self):
        self.current_node = "start"
        return self

    def __next__(self):
        self.current_node = self.nodes[self.current_node]["next"].name
        if self.current_node == "end":
            raise StopIteration
        return self.nodes[self.current_node]


class Pipe(Node):
    """Basic object for running flowcharts on states.

    A pipe is initialized with a Chart object (or Node) and can then
    be called on a state, the pipe will make a copy of the flowchart
    to run on the state and will apply that copy. This way each state
    is processed with a fresh version of the class (otherwise some
    class variables could be altered). This is most important when
    running processes in parallel. There are three processing modes
    for a Pipe: parallelize, iterate, and pass.  The parallelize mode
    will apply the flowchart on each element of the state in parallel
    up to the specified number of cores. The iterate mode will do the
    same but in serial instead of parallel. The pass mode will simply
    pass on the state to the flowchart without any iteration. The
    reason for the three modes is to allow a single Pipe object to
    play many roles in an analysis task. One may wish to nest analysis
    tasks, in which case only the top level Pipe object should run in
    parallel, but later may wish to run an inner task
    independently. Finally the user may wish to streamline the final
    result and ignore the parallelization all together. Instead of
    creating new Pipes for each case, a single pipe will suffice with
    a changing process_mode value.

    Arguments
    -----------------
    name: string
      name of the node, should be unique in the flowchart. This is how
      other nodes (i.e. decision nodes) will identify the node.

    flowchart: Chart
      Instance of a Chart object which is to be called on a number of
      states.

    safe_mode: bool
      indicate how to handle errors. In safe mode, any error raised by
      an individual run will simply return None. However, the path and
      benchmarks for the chart will still be saved thus allowing one
      to diagnose where the error occured. When safe mode is off,
      errors will be raised out of the Pipe object.

    process_mode: string
      There are three processing modes for a Pipe: parallelize,
      iterate, and pass.  The parallelize mode will apply the
      flowchart on each element of the state in parallel up to the
      specified number of cores. The iterate mode will do the same but
      in serial instead of parallel. The pass mode will simply pass on
      the state to the flowchart without any iteration.

    cores: int
      number of processes to generate in parallelize mode.

    """

    def __init__(
        self, name, flowchart, safe_mode=True, process_mode="parallelize", cores=4
    ):

        super().__init__(name)
        self.update_flowchart(flowchart)
        self.safe_mode = safe_mode
        self.process_mode = process_mode
        self.cores = cores
        self.visual_kwargs['shape'] = "parallelogram"

    def update_flowchart(self, flowchart):

        self.flowchart = flowchart
        self.benchmarks = []
        self.paths = []

    def apply_chart(self, state):

        chart = deepcopy(self.flowchart)
        logging.info(f"PIPE:{self.name}({self.process_mode}): {chart.name} ({datetime.now()})")
        if self.safe_mode:
            try:
                state = chart(state)
            except Exception as e:
                logging.error(f"on step '{chart.current_node}' got error: {str(e)}")
                logging.error("with full trace: %s" % traceback.format_exc())
                state = None
        else:
            state = chart(state)
            
        if isinstance(chart, Chart):
            timing = chart.benchmarks
            path = chart.path
        else:
            timing = [chart.benchmark]
            path = [chart.name]
        return state, timing, path

    def _run(self, state):
        if self.process_mode == "parallelize":
            starttime = time()
            with Pool(self.cores) as pool:
                result = pool.map(self.apply_chart, state)
                for r in result:
                    self.benchmarks.append(r[1])
                    self.paths.append(r[2])
                logging.info(f"PIPE:Finished parallelize run in {time() - starttime} sec")
                return list(r[0] for r in result)
        elif self.process_mode == "iterate":
            result = map(self.apply_chart, state)
            ret = []
            for r in result:
                self.benchmarks.append(r[1])
                self.paths.append(r[2])
                ret.append(r[0])
            return ret
        elif self.process_mode == "pass":
            result = self.apply_chart(state)
            self.benchmarks.append(result[1])
            self.paths.append(result[2])
            return result[0]
        raise ValueError(
            "Unrecognized process_mode: '{self.process_mode}',"
            " should be one of: parallelize, iterate, pass."
        )
