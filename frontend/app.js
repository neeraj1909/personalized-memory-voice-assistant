import { createChatClient, getSessionIdentity } from "./api.js";
import { createBrowserSpeech } from "./speech.js";
import { applyTranscript, createSpeechTurnGuard } from "./speech-session.js";
import { createLocalTranscriber } from "./local-transcription.js";
import { createLocalTranscriptionFlow } from "./local-transcription-flow.js";
import { createLocalWavRecorder } from "./local-wav.js";

const form = document.querySelector("#chat-form");
const input = document.querySelector("#message");
const conversation = document.querySelector("#conversation");
const status = document.querySelector("#status");
const sendButton = document.querySelector("#send-button");
const micButton = document.querySelector("#mic-button");
const micLabel = document.querySelector("#mic-label");
const localConsent = document.querySelector("#local-consent");
const localRecordButton = document.querySelector("#local-record");
const localStopButton = document.querySelector("#local-stop");
const localTranscribeButton = document.querySelector("#local-transcribe");
const localDiscardButton = document.querySelector("#local-discard");
const localStatus = document.querySelector("#local-status");
const identity = getSessionIdentity();
const sendMessage = createChatClient();
const speech = createBrowserSpeech();
let recognition = null;
let listening = false;
let pendingTranscript = "";
let transcriptTruncated = false;
let busy = false;
let speaking = false;
const speechTurnGuard = createSpeechTurnGuard();
const localRecorder = createLocalWavRecorder();
const localTranscriptionSupported = localRecorder.supported && window.isSecureContext &&
  Boolean(window.OfflineAudioContext || window.webkitOfflineAudioContext);
let localFlow;

function announce(message) { status.textContent = message; }

function updateLocalControls(stage = localFlow?.stage || "idle") {
  const active = ["starting", "recording", "ready", "transcribing"].includes(stage);
  localConsent.disabled = active;
  localRecordButton.disabled = !localTranscriptionSupported || !localConsent.checked || listening || ["starting", "recording", "ready", "transcribing"].includes(stage);
  localStopButton.hidden = stage !== "recording";
  localStopButton.disabled = stage !== "recording";
  localTranscribeButton.hidden = stage !== "ready";
  localTranscribeButton.disabled = stage !== "ready";
  localDiscardButton.hidden = !["starting", "recording", "ready", "transcribing", "error"].includes(stage);
  localDiscardButton.disabled = stage === "starting";
  micButton.disabled = !speech.recognitionAvailable || ["starting", "recording", "transcribing"].includes(stage);
  input.disabled = busy || stage === "transcribing";
  sendButton.disabled = busy || stage === "transcribing";
}

localFlow = createLocalTranscriptionFlow({
  recorder: localRecorder,
  transcribe: createLocalTranscriber(),
  onState: ({ stage, detail }) => {
    updateLocalControls(stage);
    const messages = {
      idle: detail || "Recording discarded. No audio was sent.",
      starting: "Requesting microphone access. No audio has been sent.",
      recording: "Recording in this page’s memory. Nothing is sent until you choose Transcribe recording.",
      ready: detail ? `${detail} Review the clip, then choose Transcribe recording or discard it.` : "Recording ready. Nothing has been sent. Choose Transcribe recording or discard it.",
      transcribing: "Sending the WAV to this app’s transcription endpoint. Discarding cancels the browser request where possible, but may not stop server processing.",
      complete: "Transcription ready in the message box. Review or edit it before sending.",
      error: detail || "Audio recording failed. You can use text or browser speech input instead.",
    };
    localStatus.textContent = messages[stage] || detail || "";
  },
  onTranscript: (text) => {
    const applied = applyTranscript(input, text);
    input.focus();
    localStatus.textContent = applied.truncated
      ? "Transcript was longer than the message limit and has been shortened. Review it before sending."
      : "Transcription ready in the message box. Review or edit it before sending.";
  },
});

if (!localTranscriptionSupported) {
  localRecordButton.disabled = true;
  localStatus.textContent = "Local recording needs microphone access, a secure browser context, and safe audio resampling support. Use text or browser speech input instead.";
}
localConsent.addEventListener("change", () => updateLocalControls());
localRecordButton.addEventListener("click", async () => {
  try { await localFlow.start(localConsent.checked); }
  catch (error) { localStatus.textContent = error.message; updateLocalControls(); }
});
localStopButton.addEventListener("click", async () => {
  try { await localFlow.stop(); }
  catch (error) { localStatus.textContent = error.message; }
});
localTranscribeButton.addEventListener("click", async () => {
  try { await localFlow.transcribe(); }
  catch (error) { localStatus.textContent = error.message; }
});
localDiscardButton.addEventListener("click", () => { void localFlow.discard(); });
window.addEventListener("pagehide", () => { void localFlow.dispose(); });
document.addEventListener("visibilitychange", () => {
  if (document.hidden && ["starting", "recording"].includes(localFlow.stage)) void localFlow.discard();
});
updateLocalControls();

function appendMessage(role, text) {
  document.querySelector("#welcome")?.remove();
  const article = document.createElement("article");
  article.className = `message message-${role}`;
  const label = document.createElement("div");
  label.className = "message-label";
  label.textContent = role === "user" ? "You" : "Assistant";
  const body = document.createElement("p");
  body.className = "message-body";
  body.textContent = text;
  article.append(label, body);
  if (role === "assistant") {
    const controls = document.createElement("div");
    controls.className = "speech-controls";
    const speak = document.createElement("button");
    speak.type = "button";
    speak.className = "text-button";
    speak.textContent = speech.synthesisAvailable ? "Read aloud" : "Speech playback unavailable";
    speak.disabled = !speech.synthesisAvailable;
    const stop = document.createElement("button");
    stop.type = "button";
    stop.className = "text-button";
    stop.textContent = "Stop speaking";
    stop.hidden = true;
    speak.addEventListener("click", () => {
      speaking = speech.speak(text, () => { speaking = false; stop.hidden = true; announce("Finished reading the response."); }, (error) => {
        speaking = false; stop.hidden = true; announce(`Speech playback error: ${error}.`);
      });
      stop.hidden = !speaking;
      announce(speaking ? "Reading the response aloud." : "Speech playback is not available in this browser.");
    });
    stop.addEventListener("click", () => {
      speech.stopSpeaking(); speaking = false; stop.hidden = true; announce("Speech playback stopped.");
    });
    controls.append(speak, stop);
    article.append(controls);
  }
  conversation.append(article);
  conversation.scrollTop = conversation.scrollHeight;
}

function setBusy(value) {
  busy = value;
  sendButton.disabled = value;
  input.disabled = value;
  sendButton.querySelector("span").textContent = value ? "Sending…" : "Send";
}

if (!speech.recognitionAvailable) {
  micButton.disabled = true;
  micButton.title = "Speech input is not supported in this browser.";
  micLabel.textContent = "Speech input unavailable";
  document.querySelector("#voice-help").textContent = "Your browser does not support speech input. You can still type and send messages.";
}

micButton.addEventListener("click", () => {
  if (listening) {
    stopRecognition();
    announce("Speech input stopped. Review your transcript before sending.");
    return;
  }
  pendingTranscript = "";
  transcriptTruncated = false;
  const turn = speechTurnGuard.begin();
  recognition = speech.createRecognition((transcript) => {
    if (!speechTurnGuard.isCurrent(turn)) return;
    const applied = applyTranscript(input, transcript);
    pendingTranscript = applied.value;
    transcriptTruncated = transcriptTruncated || applied.truncated;
    input.focus();
    if (applied.truncated) announce("Transcript reached the message limit and was shortened. Review it before sending.");
  }, (state, message) => {
    if (!speechTurnGuard.isCurrent(turn)) return;
    if (state === "listening") {
      listening = true; micButton.setAttribute("aria-pressed", "true");
      micLabel.textContent = "Stop listening"; announce("Listening. Speak your message, then review the transcript.");
      updateLocalControls();
    } else if (state === "error") {
      listening = false; micButton.setAttribute("aria-pressed", "false");
      micLabel.textContent = "Speak a message"; announce(message);
      updateLocalControls();
      recognition = null;
      speechTurnGuard.invalidate();
    } else {
      listening = false; micButton.setAttribute("aria-pressed", "false");
      micLabel.textContent = "Speak a message";
      updateLocalControls();
      if (pendingTranscript && transcriptTruncated) announce("Transcript reached the message limit and was shortened. Review it before sending.");
      else if (pendingTranscript) announce("Transcript ready. Review or edit it, then choose Send.");
      recognition = null;
      speechTurnGuard.invalidate();
    }
  });
  if (!recognition) { announce("Speech input is unavailable. You can type your message instead."); return; }
  try { recognition.start(); }
  catch {
    speechTurnGuard.invalidate(); recognition = null;
    announce("Could not start speech input. Check microphone permission or type your message instead.");
  }
});

function stopRecognition() {
  speechTurnGuard.invalidate();
  const active = recognition;
  recognition = null;
  listening = false;
  micButton.setAttribute("aria-pressed", "false");
  micLabel.textContent = "Speak a message";
  updateLocalControls();
  try {
    if (typeof active?.abort === "function") active.abort();
    else active?.stop();
  } catch { /* The turn is already invalidated; the draft remains available. */ }
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const message = input.value.trim();
  if (!message || busy) {
    if (!message) { input.focus(); announce("Enter a message before sending."); }
    return;
  }
  if (!["idle", "complete"].includes(localFlow.stage)) await localFlow.discard();
  stopRecognition();
  speech.stopSpeaking(); speaking = false;
  appendMessage("user", message);
  input.value = "";
  setBusy(true);
  announce("Sending your message.");
  try {
    const data = await sendMessage({ ...identity, message });
    appendMessage("assistant", data.response);
    announce("Response received. You can read it aloud using the button below the response.");
  } catch (error) {
    appendMessage("assistant", `I couldn’t get a response. ${error.message}`);
    announce("The message could not be sent. You can try again.");
  } finally {
    setBusy(false);
  }
});
