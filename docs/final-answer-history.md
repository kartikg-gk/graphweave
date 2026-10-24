# Final-answer history persistence

A completed browser answer is attached to its existing PostgreSQL TraceLog JSON
payload. Repository graph storage, evidence retrieval and the database schema
remain unchanged. The authenticated save route verifies session ownership;
older traces without an answer remain readable.

The browser requests newline-delimited answer events and saves only after the
explicit completion event. Partial/error responses are neither saved nor cached.
Legacy clients retain the existing plain-text streaming contract. Save failures
are visible and do not remove the answer currently on screen. History selection
restores the recorded answer without a new model call. Navigation within the app
does not cancel saving a completed answer to its original trace.

Closing the tab before the completion/save request finishes can still leave a
trace without an answer. This change does not introduce background server jobs.
History graph expansion continues using the existing subgraph endpoint; this is
not an immutable snapshot of the entire repository graph.

Validation: focused API/history and frontend regression tests; both frontend
builds. Independent Sol review was requested but encountered an account usage
limit; do not describe that review as completed unless a later review succeeds.
