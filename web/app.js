import { getAuthSession, login, logout, register } from './auth-api.js';
import { PlaybackGate, RealtimeCall } from './realtime-call.js';
import { TranscriptModel } from './transcript-model.js';
import { endCallSafely } from './end-call.js';

const app = document.querySelector('#app');
const statusLabels = {
  ready: 'Ready',
  connecting: 'Connecting…',
  listening: 'Listening to you',
  thinking: 'Thinking…',
  speaking: 'Speaking',
  error: 'Voice connection issue',
  ending: 'Ending conversation…',
};

function setPage(markup) {
  app.setAttribute('aria-busy', 'false');
  app.innerHTML = markup;
  return app;
}

function renderAuth(mode = 'login') {
  const isRegister = mode === 'register';
  setPage(`
    <header class="site-header"><a class="brand" href="/" aria-label="Personalized Memory home"><span class="brand-mark" aria-hidden="true">✦</span><span>PERSONALIZED MEMORY</span></a></header>
    <main class="auth-wrap">
      <p class="eyebrow">Your conversational voice assistant</p>
      <h1>${isRegister ? 'Create your Personalized Memory account' : 'Welcome back'}</h1>
      <p class="auth-intro">${isRegister ? 'Create a local demo account to continue your conversations.' : 'Sign in to continue your conversations.'}</p>
      <form class="auth-form" id="auth-form">
        <label class="field" for="username">Username<input id="username" name="username" autocomplete="username" required maxlength="80" /></label>
        <label class="field" for="password">Password<input id="password" name="password" type="password" autocomplete="${isRegister ? 'new-password' : 'current-password'}" required minlength="${isRegister ? 12 : 1}" /></label>
        <button class="start-button auth-submit" type="submit">${isRegister ? 'Create account' : 'Sign in'}</button>
      </form>
      <p class="auth-switch">${isRegister ? 'Already have an account?' : 'New to Personalized Memory?'} <button id="switch-auth" type="button">${isRegister ? 'Sign in' : 'Create account'}</button></p>
      <p class="error-box auth-alert" id="auth-error" role="alert" hidden></p>
      <p class="auth-footer">Your account identity is handled by Personalized Memory. This page does not store account IDs or provider credentials.</p>
    </main>
  `);
  const form = app.querySelector('#auth-form');
  const error = app.querySelector('#auth-error');
  const submit = form.querySelector('button[type="submit"]');
  app.querySelector('#switch-auth').addEventListener('click', () => renderAuth(isRegister ? 'login' : 'register'));
  form.addEventListener('submit', async (event) => {
    event.preventDefault();
    error.hidden = true;
    submit.disabled = true;
    submit.textContent = isRegister ? 'Creating account…' : 'Signing in…';
    const values = new FormData(form);
    try {
      const action = isRegister ? register : login;
      await action(String(values.get('username') ?? '').trim(), String(values.get('password') ?? ''));
      const session = await getAuthSession();
      if (!session.authenticated) throw new Error('Personalized Memory could not confirm your sign-in. Please try again.');
      renderVoice(session);
    } catch (err) {
      error.textContent = err instanceof Error ? err.message : 'Sign-in could not be completed. Please try again.';
      error.hidden = false;
      submit.disabled = false;
      submit.textContent = isRegister ? 'Create account' : 'Sign in';
    }
  });
  app.querySelector('#username').focus();
}

function renderVoice(session) {
  const root = setPage(`
    <header class="site-header">
      <a class="brand" href="/" aria-label="Personalized Memory home"><span class="brand-mark" aria-hidden="true">✦</span><span>PERSONALIZED MEMORY</span></a>
      <div class="account-tools"><span id="account-name"></span><button class="quiet-button" id="sign-out" type="button">Sign out</button></div>
    </header>
    <main class="page">
      <section class="page-intro" aria-labelledby="page-title">
        <p class="eyebrow">Your conversational voice assistant</p>
        <h1 id="page-title">Let's talk.</h1>
        <p class="intro-copy">Ask what’s on your mind. Speak naturally and pause when you’re ready for a reply. You can interrupt at any time.</p>
      </section>
      <section class="voice-panel" id="voice-panel" data-state="ready" aria-label="Voice conversation">
        <div class="voice-orb-wrap"><div class="voice-orb" aria-hidden="true"></div></div>
        <div class="state-line" role="status" aria-live="polite" aria-atomic="true"><span class="state-indicator" aria-hidden="true"></span><span id="voice-status">Ready</span></div>
        <p class="voice-hint" id="voice-hint">Start a conversation when you’re ready.</p>
        <button class="start-button" id="toggle-call" type="button">Start talking</button>
        <p class="privacy-note">Audio connects to the voice service only after you start. The service uses your signed-in Personalized Memory session.</p>
        <button class="audio-enable" id="enable-audio" type="button" hidden>Enable spoken replies</button>
      </section>
      <div class="error-box" id="voice-error" role="alert" hidden></div>
      <section class="transcript-card" aria-labelledby="transcript-title">
        <div class="transcript-heading" id="transcript-title">Conversation</div>
        <div class="transcript-list" id="transcript" role="log" aria-live="polite" aria-relevant="additions" aria-label="Live conversation transcript">
          <p class="empty-transcript" id="empty-transcript">Your conversation will appear here as you speak.</p>
        </div>
      </section>
      <p class="transcript-footnote">Voice timing and recognition depend on your connection and surroundings.</p>
      <audio id="remote-audio" autoplay playsinline class="sr-only" aria-label="Personalized Memory spoken reply"></audio>
    </main>
  `);

  root.querySelector('#account-name').textContent = session.username || 'Signed in';
  const callButton = root.querySelector('#toggle-call');
  const panel = root.querySelector('#voice-panel');
  const statusText = root.querySelector('#voice-status');
  const hint = root.querySelector('#voice-hint');
  const error = root.querySelector('#voice-error');
  const transcript = root.querySelector('#transcript');
  const audio = root.querySelector('#remote-audio');
  const enableAudio = root.querySelector('#enable-audio');
  let activeCall = null;
  let ending = false;

  const model = new TranscriptModel({
    onChange: (messages) => {
      transcript.replaceChildren();
      if (!messages.length) {
        const empty = document.createElement('p');
        empty.className = 'empty-transcript';
        empty.textContent = 'Your conversation will appear here as you speak.';
        transcript.append(empty);
        return;
      }
      for (const message of messages) {
        const row = document.createElement('article');
        row.className = 'turn';
        row.dataset.role = message.role;
        row.dataset.complete = String(Boolean(message.complete));
        row.dataset.messageId = message.id;
        const role = document.createElement('span');
        role.className = 'turn-role';
        role.textContent = message.role === 'user' ? 'YOU' : message.textOnly ? 'SERVICE NOTICE' : 'ASSISTANT';
        const text = document.createElement('p');
        text.className = 'turn-text';
        text.textContent = message.text;
        row.append(role, text);
        if (message.interrupted) {
          const note = document.createElement('span');
          note.className = 'turn-meta';
          note.textContent = 'Interrupted';
          row.append(note);
        }
        transcript.append(row);
      }
      transcript.scrollTop = transcript.scrollHeight;
    },
    onStatus: setStatus,
    onError: showError,
  });

  function setStatus(state) {
    const label = statusLabels[state] ?? statusLabels.error;
    statusText.textContent = label;
    panel.dataset.state = state;
    if (state === 'listening') error.hidden = true;
    if (state === 'thinking') hint.textContent = 'Waiting for your reply…';
    else if (state === 'speaking') hint.textContent = 'Speak whenever you want to interrupt.';
    else if (state === 'listening') hint.textContent = 'I’m listening. Pause when you’re ready for a reply.';
    else if (state === 'connecting') hint.textContent = 'Connecting your voice session…';
    else if (state === 'ending') hint.textContent = 'Closing the voice connection…';
    else if (state === 'error') hint.textContent = 'Check the message below, then try again.';
    else hint.textContent = 'Start a conversation when you’re ready.';
  }

  function showError(value) {
    const text = value instanceof Error ? value.message : 'The voice connection encountered a problem.';
    error.textContent = text;
    error.hidden = false;
    if (/audio playback was blocked/i.test(text)) enableAudio.hidden = false;
  }

  function updateButton({ connected = false } = {}) {
    const isActive = Boolean(activeCall) || ending;
    callButton.textContent = ending ? 'Ending…' : isActive ? 'End conversation' : 'Start talking';
    callButton.classList.toggle('is-ending', isActive);
    callButton.disabled = ending;
    if (!connected && !ending && !activeCall && panel.dataset.state !== 'error') setStatus('ready');
  }

  async function startCall() {
    error.hidden = true;
    enableAudio.hidden = true;
    model.clear();
    const playback = new PlaybackGate(audio);
    audio.muted = false;
    let call;
    call = new RealtimeCall({
      audioElement: audio,
      onEvent: (event) => {
        playback.receive(event);
        model.receive(event);
      },
      onStatus: (state) => {
        setStatus(state);
        if (state === 'ready' && activeCall === call) {
          activeCall = null;
          updateButton();
        }
      },
      onError: showError,
    });
    activeCall = call;
    updateButton({ connected: true });
    setStatus('connecting');
    try {
      await call.start();
      if (activeCall === call) updateButton({ connected: true });
    } catch {
      if (activeCall === call) {
        // A failed server cleanup leaves an owner-bound call ID for retry.
        if (!call.callId) activeCall = null;
        updateButton();
      }
    }
  }

  async function endCall() {
    if (!activeCall || ending) return false;
    const call = activeCall;
    return endCallSafely(call, {
      setEnding: (value) => { ending = value; updateButton(); },
      setStatus,
      onEnded: () => { if (activeCall === call) activeCall = null; updateButton(); },
      onError: (err) => { showError(err); updateButton(); },
    });
  }

  callButton.addEventListener('click', () => {
    if (activeCall) void endCall();
    else if (!ending) void startCall();
  });
  enableAudio.addEventListener('click', async () => {
    try {
      await audio.play();
      enableAudio.hidden = true;
      error.hidden = true;
    } catch {
      showError(new Error('Audio is still blocked. Check your browser audio settings.'));
    }
  });
  root.querySelector('#sign-out').addEventListener('click', async (event) => {
    const button = event.currentTarget;
    button.disabled = true;
    if (activeCall && !(await endCall())) {
      button.disabled = false;
      return;
    }
    try {
      await logout();
      renderAuth('login');
    } catch (err) {
      showError(err);
      button.disabled = false;
    }
  });
  window.addEventListener('pagehide', () => {
    if (activeCall) void activeCall.end();
  }, { once: true });
}

async function boot() {
  try {
    const session = await getAuthSession();
    if (session.authenticated) renderVoice(session);
    else renderAuth('login');
  } catch (error) {
    setPage('<header class="site-header"><a class="brand" href="/">PERSONALIZED MEMORY</a></header><main class="auth-wrap"><h1>Personalized Memory is unavailable</h1><p class="auth-intro">Could not check your sign-in session.</p><div class="error-box" role="alert"></div><button class="start-button auth-submit" id="retry-session" type="button">Try again</button></main>');
    app.querySelector('.error-box').textContent = error instanceof Error ? error.message : 'Check your connection and try again.';
    app.querySelector('#retry-session').addEventListener('click', boot);
  }
}

boot();
