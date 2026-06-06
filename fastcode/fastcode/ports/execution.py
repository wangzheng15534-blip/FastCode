"""External execution capability ports — moved to infrastructure/execution/ports.py.

The execution tool traits (ScipIndexerRuntime, SemanticHelperRuntime, etc.) were
generic tool traits that referenced effect_tool types (SCIPIndex). They have been
relocated to their proper FCIS role (effect_tool) at
fastcode.infrastructure.execution.ports.
"""
