# This makes the simulator a package for easy importing
# and allows us to separate the quantum simulator from the circuit logic.
# The simulator package contains the QuantumSimulator class,
# the utils module, and the circuit module.
# The QuantumSimulator class is responsible for simulating the quantum state
# and applying quantum gates to the state.
# The utils module contains helper functions for the simulator,
# such as normalizing the quantum state.
# The circuit module defines the QuantumCircuit class,
# which allows users to build quantum circuits by adding gates
# and executing them on the quantum simulator.
# This structure makes it easy to extend the simulator
# with new gates and features in the future.
# The simulator package can be used as a standalone library
# or integrated into a larger quantum computing framework.
