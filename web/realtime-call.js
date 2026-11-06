const CALL_PATH = '/api/realtime/call';
const CLEANUP_TIMEOUT_MS = 10_000;

// A tool response also emits response.created, before any new audio exists.
// Wait for the provider's audio-buffer event before restoring playback.
export class PlaybackGate {
  constructor(audio) {
    this.audio = audio;
    this.activeResponseId = null;
    this.interrupted = new Set();
    this.userSpeaking = false;
  }

  receive(event) {
    if (event.type === 'response.created') {
      this.activeResponseId = event.response?.id ?? null;
    } else if (event.type === 'input_audio_buffer.speech_started') {
      this.userSpeaking = true;
      if (this.activeResponseId) {
        this.interrupted.add(this.activeResponseId);
        if (this.interrupted.size > 128) this.interrupted.delete(this.interrupted.values().next().value);
      }
      this.audio.muted = true;
    } else if (event.type === 'input_audio_buffer.speech_stopped') {
      this.userSpeaking = false;
    } else if (event.type === 'output_audio_buffer.started' || event.type === 'response.output_audio_buffer.started') {
      const responseId = event.response_id ?? this.activeResponseId;
      if (!this.userSpeaking && !this.interrupted.has(responseId)) this.audio.muted = false;
    }
  }
}

export class RealtimeCall {
  constructor({
    mediaDevices = globalThis.navigator?.mediaDevices,
    RTCPeerConnection: Peer = globalThis.RTCPeerConnection,
    MediaStream: Media = globalThis.MediaStream,
    fetchImpl = globalThis.fetch?.bind(globalThis),
    audioElement = null,
    cleanupTimeoutMs = CLEANUP_TIMEOUT_MS,
    onEvent = () => {},
    onStatus = () => {},
    onError = () => {},
  } = {}) {
    this.mediaDevices = mediaDevices;
    this.Peer = Peer;
    this.Media = Media;
    this.fetchImpl = fetchImpl;
    this.audioElement = audioElement;
    this.cleanupTimeoutMs = cleanupTimeoutMs;
    this.onEvent = onEvent;
    this.onStatus = onStatus;
    this.onError = onError;
    this.generation = 0;
    this.callId = null;
    this.peer = null;
    this.channel = null;
    this.stream = null;
    this.starting = false;
    this.pendingStart = null;
    this.readyPromise = null;
    this.deletePromise = null;
    this.ready = false;
  }

  start() {
    if (this.pendingStart) return Promise.reject(new Error('A voice call is already active.'));
    const pending = this.#startCall();
    this.pendingStart = pending;
    return pending.finally(() => {
      if (this.pendingStart === pending) this.pendingStart = null;
    });
  }

  async #startCall() {
    if (this.starting || this.peer || this.callId) throw new Error('A voice call is already active.');
    if (!this.mediaDevices?.getUserMedia || !this.Peer || !this.fetchImpl) {
      throw this.#fail(new Error('Live voice is not supported in this browser.'));
    }

    const generation = ++this.generation;
    this.starting = true;
    this.onStatus('connecting');
    try {
      try {
        this.stream = await this.mediaDevices.getUserMedia({ audio: true, video: false });
      } catch (error) {
        if (error?.name === 'NotAllowedError' || error?.name === 'PermissionDeniedError') {
          throw new Error('Microphone permission was denied. Allow microphone access in your browser settings and try again.');
        }
        if (error?.name === 'NotFoundError' || error?.name === 'DevicesNotFoundError') {
          throw new Error('No microphone was found. Connect a microphone and try again.');
        }
        throw new Error('The microphone could not be opened. Check browser permissions and try again.');
      }
      if (!this.#isCurrent(generation)) {
        this.#stopTracks();
        return false;
      }

      const peer = new this.Peer();
      this.peer = peer;
      this.channel = peer.createDataChannel('oai-events');
      const channel = this.channel;
      channel.onopen = () => {
        if (this.#isCurrent(generation) && this.channel === channel && this.ready) this.onStatus('listening');
      };
      channel.onmessage = (event) => {
        if (!this.#isCurrent(generation) || this.channel !== channel) return;
        try {
          const data = JSON.parse(event.data);
          if (data && typeof data.type === 'string') this.onEvent(data);
        } catch {
          // Ignore malformed provider events; connection state remains visible.
        }
      };
      channel.onerror = () => {
        if (this.#isCurrent(generation)) this.onError(new Error('The voice event channel failed.'));
      };

      peer.ontrack = async (event) => {
        if (!this.#isCurrent(generation) || !this.audioElement) return;
        const stream = event.streams?.[0] ?? (this.Media ? new this.Media([event.track]) : null);
        if (!stream) return;
        this.audioElement.srcObject = stream;
        try {
          await this.audioElement.play?.();
        } catch {
          this.onError(new Error('Audio playback was blocked. Check your browser audio controls.'));
        }
      };
      peer.onconnectionstatechange = () => {
        if (!this.#isCurrent(generation)) return;
        if (peer.connectionState === 'connected') {
          if (this.ready) this.onStatus('listening');
          else void this.#markReady(generation).catch((error) => {
            if (this.#isCurrent(generation)) {
              this.#fail(error);
              void this.end().catch(() => {}).finally(() => this.onStatus('error'));
            }
          });
        }
        if (peer.connectionState === 'disconnected') {
          this.onStatus('connecting');
          this.onError(new Error('The voice connection was interrupted. Waiting for the network to recover.'));
        }
        if (peer.connectionState === 'failed') {
          this.onError(new Error('The voice connection failed. End the conversation and try again.'));
          void this.end().catch(() => {}).finally(() => this.onStatus('error'));
        }
      };
      for (const track of this.stream.getAudioTracks()) peer.addTrack(track, this.stream);

      const offer = await peer.createOffer();
      await peer.setLocalDescription(offer);
      await this.#waitForIce(peer, generation);
      if (!this.#isCurrent(generation)) return false;

      // Do not abort this request on End: its response contains the call ID needed
      // to release any server-side session created while negotiation was pending.
      const response = await this.fetchImpl(CALL_PATH, {
        method: 'POST',
        credentials: 'same-origin',
        headers: { 'Content-Type': 'application/sdp' },
        body: peer.localDescription?.sdp ?? offer.sdp,
      });
      const callId = response.headers.get('X-Call-ID');
      if (!this.#isCurrent(generation)) {
        if (callId) await this.#deleteCall(callId);
        return false;
      }
      if (callId) this.callId = callId;
      if (!response.ok) throw this.#httpError(response.status);
      if (!callId) throw new Error('The voice service did not identify the new call.');
      const answer = await response.text();
      if (!answer.trim()) throw new Error('The voice service returned an empty connection answer.');
      if (!this.#isCurrent(generation)) {
        await this.#deleteCall(callId);
        return false;
      }
      await peer.setRemoteDescription({ type: 'answer', sdp: answer });
      if (!this.#isCurrent(generation)) {
        await this.#deleteCall(callId);
        return false;
      }
      if (peer.connectionState === 'connected') await this.#markReady(generation);
      return true;
    } catch (error) {
      const safeError = this.#fail(error);
      if (this.#isCurrent(generation)) {
        await this.#dispose({ deleteServerCall: true });
        this.onStatus('error');
      }
      throw safeError;
    } finally {
      if (this.#isCurrent(generation)) this.starting = false;
    }
  }

  async end() {
    this.generation += 1;
    this.starting = false;
    this.ready = false;
    // Stop local capture immediately, but retain the cookie and wait for any
    // outstanding offer/ready request to reveal the server call ID first.
    await this.#dispose({ deleteServerCall: false });
    const pendingStart = this.pendingStart;
    if (pendingStart) {
      try { await this.#withCleanupTimeout(pendingStart); }
      catch (error) {
        if (error?.message?.includes('cleanup is still pending')) throw error;
      }
    }
    if (this.readyPromise) {
      try { await this.#withCleanupTimeout(this.readyPromise); }
      catch (error) {
        if (error?.message?.includes('cleanup is still pending')) throw error;
      }
    }
    await this.#dispose({ deleteServerCall: true });
    this.onStatus('ready');
  }

  async #markReady(generation) {
    if (!this.#isCurrent(generation) || this.ready || this.readyPromise) return this.readyPromise;
    if (this.peer?.connectionState !== 'connected') return;
    const callId = this.callId;
    if (!callId) return;
    const promise = (async () => {
      const response = await this.#fetchWithTimeout(`${CALL_PATH}/${encodeURIComponent(callId)}/ready`, {
        method: 'POST', credentials: 'same-origin',
      });
      if (!response.ok) throw this.#httpError(response.status);
      if (!this.#isCurrent(generation)) return;
      this.ready = true;
      if (this.peer?.connectionState === 'connected') this.onStatus('listening');
    })();
    this.readyPromise = promise;
    try { await promise; }
    finally { if (this.readyPromise === promise) this.readyPromise = null; }
  }

  #withCleanupTimeout(promise) {
    let timer;
    return Promise.race([
      promise,
      new Promise((_, reject) => { timer = setTimeout(() => reject(new Error('Voice cleanup is still pending. Keep this page open and try End again.')), this.cleanupTimeoutMs); }),
    ]).finally(() => clearTimeout(timer));
  }

  #fetchWithTimeout(url, options) {
    return this.fetchImpl(url, { ...options, signal: AbortSignal.timeout(this.cleanupTimeoutMs) });
  }

  async #waitForIce(peer, generation) {
    if (peer.iceGatheringState === 'complete') return;
    await new Promise((resolve) => {
      const done = () => {
        clearTimeout(timeout);
        peer.removeEventListener?.('icegatheringstatechange', onState);
        if (!peer.removeEventListener && peer.onicegatheringstatechange === onState) {
          peer.onicegatheringstatechange = null;
        }
        resolve();
      };
      const onState = () => {
        if (peer.iceGatheringState === 'complete') done();
      };
      const timeout = setTimeout(done, 4000);
      peer.addEventListener?.('icegatheringstatechange', onState);
      if (!peer.addEventListener) peer.onicegatheringstatechange = onState;
      onState();
      if (!this.#isCurrent(generation)) done();
    });
  }

  async #dispose({ deleteServerCall }) {
    const callId = this.callId;
    const channel = this.channel;
    const peer = this.peer;
    this.channel = null;
    this.peer = null;
    if (channel) {
      channel.onopen = null;
      channel.onmessage = null;
      channel.onerror = null;
      try { channel.close(); } catch { /* already closed */ }
    }
    if (peer) {
      peer.ontrack = null;
      peer.onconnectionstatechange = null;
      try { peer.close(); } catch { /* already closed */ }
    }
    this.#stopTracks();
    if (this.audioElement) {
      try { this.audioElement.pause?.(); } catch { /* no-op */ }
      this.audioElement.srcObject = null;
    }
    if (deleteServerCall && callId) await this.#deleteCall(callId);
  }

  #stopTracks() {
    const stream = this.stream;
    this.stream = null;
    stream?.getTracks().forEach((track) => {
      try { track.stop(); } catch { /* already stopped */ }
    });
  }

  async #deleteCall(callId) {
    if (this.deletePromise) return this.deletePromise;
    this.callId ??= callId;
    const pending = (async () => {
      try {
        const response = await this.#fetchWithTimeout(`${CALL_PATH}/${encodeURIComponent(callId)}`, {
          method: 'DELETE',
          credentials: 'same-origin',
          keepalive: true,
        });
        if (!response.ok) throw new Error('Unconfirmed cleanup');
        if (this.callId === callId) this.callId = null;
      } catch (error) {
        const safe = new Error('The server could not confirm that the voice session ended. Keep this page open and try End again.');
        this.onError(safe);
        throw safe;
      }
    })();
    this.deletePromise = pending;
    try { await pending; }
    finally { if (this.deletePromise === pending) this.deletePromise = null; }
  }

  #httpError(status) {
    if (status === 401 || status === 403) return new Error('Your Personalized Memory sign-in is no longer active. Sign in again to start a voice conversation.');
    if (status === 429 || status === 503) return new Error('Voice service is unavailable right now. Try again shortly.');
    return new Error('Could not start the voice conversation. Check your connection and try again.');
  }

  #fail(error) {
    const safeError = error instanceof Error ? error : new Error('The voice conversation could not start.');
    this.onError(safeError);
    return safeError;
  }

  #isCurrent(generation) {
    return generation === this.generation;
  }
}
