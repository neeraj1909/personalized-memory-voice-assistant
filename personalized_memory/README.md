# Memory service

This package contains the personalized chat and SQLite memory service. It is based on the MyNaksh assignment and retains its database schema. Start it with `uvicorn personalized_memory.main:app` and configure the model and database as described in the [project README](../README.md).

The optional fast mode combines context selection and generation in one model call. It loads candidate memories before selection, so irrelevant context may influence a response. Its bounded cache requires a single writer per database; process restarts do not delete saved memory. Concurrent turns for the same user are serialized.

The `/chat` endpoint accepts caller-supplied user IDs and has no independent authentication. Keep it on loopback behind the web application.
