"""Runtime determinism capability ports — removed.

Clock and IdGenerator were pure generic helpers with no need for Protocol
traits. Consumers now use concrete SystemClock / PrefixedIdGenerator from
fastcode.utils.clock and fastcode.utils.ids directly.
"""
