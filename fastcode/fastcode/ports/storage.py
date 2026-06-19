"""Store-facing semantic capability ports.

This module keeps the semantic store-facing capability ports consumed only by
use_flow code. Low-level effect runtime traits (``StoreDatabaseRuntime``) are
owned by their use_flow consumer at ``fastcode.app.store.runtime_contracts``
per the FCIS consumer-owns-the-port rule; concrete effect_facility adapters
satisfy them structurally without importing them.
"""
