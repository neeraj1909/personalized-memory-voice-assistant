# Voice web client

This directory contains the browser client for Personalized Memory Voice Assistant. The FastAPI application serves it and handles account sessions and Realtime signaling. Setup and test commands are in the [project README](../README.md).

The client uses WebRTC for audio and displays conversation transcripts. Its tests use simulated media and do not validate microphone quality or provider latency.
