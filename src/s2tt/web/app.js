'use strict';
const el = (id) => document.getElementById(id);
let socket, context, stream, node, sequence = 0, offset = 0, lastEvent = 0, corrections = 0;
let flushResolve, readyResolve, readyReject;
function status(text) { el('status').textContent = text; }
function controls(active) { el('start').disabled = active; el('stop').disabled = !active; el('cancel').disabled = !active; }
function sendAudio(samples) {
  if (!socket || socket.readyState !== WebSocket.OPEN) return;
  if (socket.bufferedAmount > 1024 * 1024) { cancel('发送积压过大，已取消本次会话。'); return; }
  const buffer = new ArrayBuffer(12 + samples.length * 2);
  const view = new DataView(buffer);
  view.setUint32(0, sequence++, true); view.setBigUint64(4, BigInt(offset), true);
  for (let i = 0; i < samples.length; i++) {
    const value = Math.max(-1, Math.min(1, Number.isFinite(samples[i]) ? samples[i] : 0));
    view.setInt16(12 + i * 2, Math.max(-32768, Math.min(32767, Math.round(value * 32768))), true);
  }
  offset += samples.length; socket.send(buffer);
}
function onEvent(event) {
  if (event.event_seq && event.event_seq <= lastEvent) return;
  if (event.event_seq) lastEvent = event.event_seq;
  if (event.type === 'Ready') {
    el('badge').textContent = event.model_kind === 'mock' ? '模拟模式 · 不进行翻译' : 'Echo · 英译中待验证';
    readyResolve?.(); return;
  }
  if (Number.isFinite(event.received_end_sample)) {
    el('received').textContent = (event.received_end_sample / 16000).toFixed(2) + ' s';
    el('used').textContent = (event.used_audio_end_sample / 16000).toFixed(2) + ' s';
    el('backlog').textContent = ((event.received_end_sample - event.used_audio_end_sample) / 16000).toFixed(2) + ' s';
  }
  if (['CommitAppend', 'Correction', 'DraftSnapshot'].includes(event.type)) {
    el('committed').textContent = event.committed_text || '等待稳定提交…';
    el('draft').textContent = event.draft_text || '等待下一段语音…';
    el('notice').textContent = event.conflict ? '新假设与已提交文字存在冲突，等待确认。' : '草稿可能随新增语音调整。';
  }
  if (event.type === 'Correction') { el('corrections').textContent = ++corrections; status('已收到明确更正'); }
  if (event.type === 'Error') {
    readyReject?.(new Error(event.detail || event.code));
    status('会话未完成：' + (event.detail || event.code)); cleanup(); controls(false);
  }
  if (event.type === 'StreamEnd') {
    status(event.complete ? '本次语音处理完成' : '会话已取消，尾段未终态化');
    cleanup(); controls(false);
  }
}
async function cleanup() {
  const oldStream = stream, oldNode = node, oldContext = context;
  stream = node = context = undefined;
  oldStream?.getTracks().forEach((track) => track.stop());
  oldNode?.disconnect();
  if (oldContext && oldContext.state !== 'closed') await oldContext.close();
}
async function start() {
  try {
    controls(true); status('请求麦克风权限…');
    sequence = offset = lastEvent = corrections = 0;
    el('corrections').textContent = '0'; el('committed').textContent = '等待稳定提交…'; el('draft').textContent = '等待语音输入…';
    stream = await navigator.mediaDevices.getUserMedia({audio: {channelCount: 1, echoCancellation: true}});
    context = new AudioContext({sampleRate: 48000});
    await context.audioWorklet.addModule('capture-worklet.js');
    await context.suspend();
    const ready = new Promise((resolve, reject) => { readyResolve = resolve; readyReject = reject; });
    const currentSocket = new WebSocket(el('address').value.trim());
    socket = currentSocket;
    socket.onmessage = ({data}) => { if (socket !== currentSocket) return; try { onEvent(JSON.parse(data)); } catch (error) { status(error.message); } };
    socket.onerror = () => { if (socket === currentSocket) readyReject?.(new Error('无法连接 WebSocket 服务')); };
    socket.onclose = () => {
      if (socket !== currentSocket) return;
      readyReject?.(new Error('连接已关闭'));
      if (stream) { status('连接中断，当前会话不可恢复'); cleanup(); controls(false); }
    };
    socket.onopen = () => socket.send(JSON.stringify({type: 'Start', protocol_version: 1,
      source_language: 'en', target_language: 'zh', sample_rate: context.sampleRate, channels: 1,
      token: el('token').value, glossary: el('glossary').value.split('\n').map((s) => s.trim()).filter(Boolean)}));
    let readyTimeout;
    try { await Promise.race([ready, new Promise((_, reject) => { readyTimeout = setTimeout(() => reject(new Error('服务启动响应超时')), 15000); })]); }
    finally { clearTimeout(readyTimeout); readyResolve = readyReject = undefined; }
    node = new AudioWorkletNode(context, 'echo-capture');
    const currentNode = node;
    node.port.onmessage = ({data}) => {
      if (node !== currentNode) return;
      if (data.type === 'audio') sendAudio(data.samples);
      if (data.type === 'flushed') { flushResolve?.(); flushResolve = undefined; }
    };
    context.createMediaStreamSource(stream).connect(node);
    const muted = context.createGain(); muted.gain.value = 0; node.connect(muted).connect(context.destination);
    await context.resume(); status('正在收音');
  } catch (error) { status(error.message); socket?.close(); await cleanup(); controls(false); }
}
async function stop() {
  el('stop').disabled = true; status('停止收音，等待尾段处理…');
  try {
    if (context) await context.suspend();
    if (node) {
      await new Promise((resolve, reject) => {
        const timer = setTimeout(() => reject(new Error('音频尾包排空超时')), 2000);
        flushResolve = () => { clearTimeout(timer); resolve(); };
        node.port.postMessage({type: 'flush'});
      });
    }
    if (socket?.readyState === WebSocket.OPEN) socket.send(JSON.stringify({type: 'End', last_frame_seq: sequence - 1}));
    await cleanup();
  } catch (error) { cancel(error.message); }
}
function cancel(message = '已取消本次会话') {
  if (socket?.readyState === WebSocket.OPEN) socket.send(JSON.stringify({type: 'Cancel'}));
  cleanup(); controls(false); status(message);
}
el('start').onclick = start; el('stop').onclick = stop; el('cancel').onclick = () => cancel();
window.addEventListener('beforeunload', () => { stream?.getTracks().forEach((track) => track.stop()); socket?.close(); });
