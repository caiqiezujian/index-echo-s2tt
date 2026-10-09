class EchoCapture extends AudioWorkletProcessor {
  constructor() {
    super();
    this.pending = [];
    this.count = 0;
    this.limit = Math.round(sampleRate * 0.04);
    this.port.onmessage = ({data}) => {
      if (data.type === 'flush') {
        this.emit();
        this.port.postMessage({type: 'flushed'});
      }
    };
  }
  emit() {
    if (!this.count) return;
    const samples = new Float32Array(this.count);
    let offset = 0;
    for (const block of this.pending) { samples.set(block, offset); offset += block.length; }
    this.port.postMessage({type: 'audio', samples}, [samples.buffer]);
    this.pending = []; this.count = 0;
  }
  process(inputs) {
    const channel = inputs[0]?.[0];
    if (channel) {
      this.pending.push(channel.slice()); this.count += channel.length;
      if (this.count >= this.limit) this.emit();
    }
    return true;
  }
}
registerProcessor('echo-capture', EchoCapture);
