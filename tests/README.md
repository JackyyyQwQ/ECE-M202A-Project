# Focused checks — planned

There is no automated experiment-reproduction or firmware test suite yet.
When implementing the device path, check exact sample retrieval across ring
wraparound, overwritten/future requests, timestamp and label alignment, and
request/response parsing. Use known synthetic sample values before real audio.
For model changes, verify causality and compare against context-only on fixed
recording splits. Preserve missed/false events and retrieval costs in evaluation.
