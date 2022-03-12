"""author: Connor Stone

summary:
  core objects used to construct flowchart analysis pipelines.

description:
  This module contains classes allowing the user to encapsulate code
  in a framework which fundamentally represents code as a
  flowchart. This makes for more modular code which can be tested and
  updated easily. Conceptually every element of the flowchart is a
  "Node" and these nodes are linked together. 
"""

from time import time
from inspect import signature


class Node:
    """Base object for all nodes in the flowchart

    Core functionality for nodes in the flowchart. Includes methods
    for linking flowchart elements, a run method which operates on the
    state as it passes through the flowchart, and the specifications
    for the nodes visual appearance. A given node can be linked to any
    other node, when doing so it will store a pointer to that
    object. Nodes also retain a list of nodes which point to them,
    though this has no defaul functional purpose and is there only to
    support inheriting node functionality.

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

    def __init__(self, name, func=None):
        self.name = name
        self.forward = None
        self.reverse = set()
        self.status = "initialized"
        self.benchmark = -1
        self.visual_kwargs = {'color': "black",
                              'shape': 'plain',
                              'style': 'solid'}
        if not func is None:
            self.update_run(func)

    def update_run(self, func):
        """Method to update the function applied by the current Node. Do
        not directly alter the internal _run variable.

        Arguments
        -----------------
        func: function
          Function which takes the state as an argument and returns
          the updated state. Can also take the "self" object for the
          current Node to allow access to state variables of the
          Node.

        """
        self._run = func

    def link_forward(self, node):
        """Link to the next Node in the flow chart. Stores a reference to
        the next Node object, also calls the "link_reverse" function
        of the forward Node so that it knows what has linked to
        it. This is the primary function for building the
        flowchart. Though it should mostly be used internally by the
        chart object, users may wish to make use of it.

        Arguments
        -----------------
        node: Node
          A Node object to be linked as the next step in the
          flowchart.

        """
        self.forward = node
        node.link_reverse(self)

    def unlink_forward(self, node):
        """Undo the operations of the "link_forward" method. Returns the
        Node to its initial state where it is no longer linked
        forward to any object.

        Arguments
        -----------------
        node: Node
          A Node object to be unlinked from this step in the
          flowchart.

        """
        self.forward = None
        node.unlink_reverse(self)

    def link_reverse(self, node):
        """Store a reference to a Node which has linked to this
        Node. This function should only be used internally as it is
        expected that the "link_forward" method will call this
        method. Anyone constructing their own flowchart step type
        which inherits from Node should keep this behaviour in mind.

        Arguments
        -----------------
        node: Node
          A Node object which was linked to this step in the
          flowchart.

        """
        self.reverse.update([node])

    def unlink_reverse(self, node):
        """Undo the operations of the "link_reverse" method. Returns the
        Node to its initial state where it is no longer linked to
        any object.

        Arguments
        -----------------
        node: Node
          A Node object to be unlinked from this step in the
          flowchart.

        """
        self.reverse.remove(node)

    def _run(self, state):
        """Placeholder function for the behaviour of a Node. Simply
        passes the state on unaltered.

        Arguments
        -----------------
        state: object
          container for all information related to a flowchart
          analysis task.

        """
        return state

    def __call__(self, state):
        self.status = "running"
        start = time()
        sig = signature(self._run)
        if "self" in sig.parameters:
            res = self._run(self, state)
        else:
            res = self._run(state)
        self.benchmark = time() - start
        self.status = "complete"
        return res

    def __getitem__(self, key):
        if key == "next" and not self.forward is None:
            return self.forward
        else:
            raise StopIteration()

    def __str__(self):
        return self.name


class State:
    """Dummy object to store state information

    This can be used as a state object to pass information through a
    flowchart.
    """
