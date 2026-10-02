"""Tool-calling conformance for every text AI provider (RFC §6.4, §6.7, RMK-378).

The same scenarios run on every provider, through a driver per wire family that
puts a neutral :class:`~tests.text_conformance.script.Script` behind it as that
vendor's SDK objects, streamed and through ``generate()``, on no network, and
reads back the request the provider sent. A scenario a wire cannot express is
skipped with the driver's reason, never silently, and a provider class with
neither a driver nor a stated exemption fails the suite (``test_coverage``).
"""
