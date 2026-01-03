/**
 * Voice Service - Hybrid Implementation
 * 
 * TTS: First 2 sentences via Native Windows SpeechKit (Fast start)
 *      Following sentences via Google Gemini TTS (High quality)
 * STT: MediaRecorder + Groq Whisper
 */

class VoiceService {
    constructor() {
        this.isVoiceEnabled = true;
        this.isPaused = false;
        this.isRecording = false;
        this.playbackId = 0;
        
        // Native Speech Synthesis
        this.synth = window.speechSynthesis;
        this.nativeVoice = null;
        
        // High Quality Audio Context
        this.audioContext = null;
        this.audioQueue = [];
        this.currentSource = null;
        this.audioChunks = []; // Collect all chunks before decoding
        this.isCollectingChunks = false;
        this.completeAudioData = null; // Store complete audio blob
        
        // Sentence processing for Hybrid TTS
        this.sentenceIndex = 0;
        this.isProcessingQueue = false;
        
        // STT
        this.mediaRecorder = null;
        this.recordedChunks = [];
        this.sttAvailable = true;
        
        // Context
        this.lastAnalysisText = '';
        this.onVoiceInput = null;
        
        this.elements = {
            toggleBtn: null,
            pauseBtn: null,
            recordBtn: null,
            statusEl: null
        };
        
        this.initNativeVoices();
    }
    
    initNativeVoices() {
        const load = () => {
            const voices = this.synth.getVoices();
            this.nativeVoice = voices.find(v => v.lang.includes('ru') && v.name.includes('Google')) || 
                               voices.find(v => v.lang.includes('ru')) || 
                               voices[0];
        };
        if (this.synth.onvoiceschanged !== undefined) this.synth.onvoiceschanged = load;
        load();
    }

    initAudioContext() {
        if (!this.audioContext) {
            this.audioContext = new (window.AudioContext || window.webkitAudioContext)({
                sampleRate: 24000
            });
        }
    }

    init(elements) {
        this.elements = elements;
        this.setupEventListeners();
    }
    
    setupEventListeners() {
        if (this.elements.toggleBtn) this.elements.toggleBtn.addEventListener('click', () => this.toggleVoice());
        if (this.elements.pauseBtn) this.elements.pauseBtn.addEventListener('click', () => this.togglePause());
        if (this.elements.recordBtn) {
            this.elements.recordBtn.addEventListener('mousedown', () => this.startRecording());
            this.elements.recordBtn.addEventListener('mouseup', () => this.stopRecording());
            this.elements.recordBtn.addEventListener('mouseleave', () => { if (this.isRecording) this.stopRecording(); });
        }
    }
    
    toggleVoice() {
        this.isVoiceEnabled = !this.isVoiceEnabled;
        if (this.elements.toggleBtn) {
            // Toggle class for SVG icon visibility (no text change needed)
            this.elements.toggleBtn.classList.toggle('voice-on', this.isVoiceEnabled);
        }
        if (!this.isVoiceEnabled) this.stopPlayback();
    }
    
    togglePause() {
        if (this.synth.speaking) {
            if (this.synth.paused) this.synth.resume(); else this.synth.pause();
        }
        if (this.audioContext) {
            if (this.audioContext.state === 'running') this.audioContext.suspend();
            else if (this.audioContext.state === 'suspended') this.audioContext.resume();
        }
        this.isPaused = !this.isPaused;
        if (this.elements.pauseBtn) this.elements.pauseBtn.textContent = this.isPaused ? '▶️' : '⏸️';
    }
    
    stopPlayback() {
        this.playbackId++;
        this.synth.cancel();
        if (this.currentSource) {
            try { this.currentSource.stop(); } catch(e) {}
            this.currentSource = null;
        }
        this.audioQueue = [];
        this.byteBuffer = new Uint8Array(0);
        this.completeAudioData = new Uint8Array(0);
        this.isCollectingChunks = false;
        this.sentenceIndex = 0;
        this.isPaused = false;
        
        if (this.elements.pauseBtn) {
            this.elements.pauseBtn.disabled = true;
            this.elements.pauseBtn.textContent = '⏸️';
        }
        this.updateStatus('');
    }

    /**
     * Required by app.js for streaming
     */
    queueSpeak(sentence) {
        // #region agent log
        fetch('http://127.0.0.1:7243/ingest/c331841f-7e4f-4c50-9c5e-a68c9827234e',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'voice.js:125',message:'queueSpeak',data:{sentence,sentenceIndex:this.sentenceIndex,playbackId:this.playbackId},timestamp:Date.now(),sessionId:'debug-session',hypothesisId:'1'})}).catch(()=>{});
        // #endregion
        if (!this.isVoiceEnabled || !sentence) return;
        
        // First 4 sentences are fast (Native) - increased from 3 as requested
        if (this.sentenceIndex < 4) {
            this.speakNative(sentence, this.playbackId);
        } else {
            // Rest are high quality (Google)
            this.speakGoogle(sentence, this.playbackId);
        }
        this.sentenceIndex++;
    }

    /**
     * Fallback speak for whole text
     */
    async speak(text) {
        // #region agent log
        fetch('http://127.0.0.1:7243/ingest/c331841f-7e4f-4c50-9c5e-a68c9827234e',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'voice.js:145',message:'speak called',data:{textLen:text?.length,playbackId:this.playbackId},timestamp:Date.now(),sessionId:'debug-session',hypothesisId:'1'})}).catch(()=>{});
        // #endregion
        if (!this.isVoiceEnabled || !text) return;
        this.stopPlayback();
        const pid = this.playbackId;
        this.lastAnalysisText = text;
        
        const sentences = text.match(/[^.!?]+[.!?]*/g) || [text];
        sentences.forEach(s => this.queueSpeak(s));
    }

    speakNative(text, pid) {
        const clean = this.cleanText(text);
        if (!clean) return;

        const utterance = new SpeechSynthesisUtterance(clean);
        if (this.nativeVoice) utterance.voice = this.nativeVoice;
        utterance.lang = 'ru-RU';
        utterance.rate = 1.1;
        
        utterance.onstart = () => {
            if (this.playbackId !== pid) { this.synth.cancel(); return; }
            this.updateStatus('Озвучка (Fast)...');
            if (this.elements.pauseBtn) this.elements.pauseBtn.disabled = false;
        };
        
        this.synth.speak(utterance);
    }

    async speakGoogle(text, pid) {
        const clean = this.cleanText(text);
        if (!clean) return;

        // #region agent log
        fetch('http://127.0.0.1:7243/ingest/c331841f-7e4f-4c50-9c5e-a68c9827234e',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'voice.js:175',message:'speakGoogle entry',data:{clean,pid,sentenceIndex:this.sentenceIndex},timestamp:Date.now(),sessionId:'debug-session',hypothesisId:'2'})}).catch(()=>{});
        // #endregion

        this.initAudioContext();
        this.completeAudioData = new Uint8Array(0); // Reset for new audio
        this.isCollectingChunks = true;
        this.byteBuffer = new Uint8Array(0); // For base64 decoding

        try {
            const response = await fetch('/api/voice/tts/stream', {
                method: 'POST',
                headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
                body: `text=${encodeURIComponent(clean)}`
            });

            // #region agent log
            fetch('http://127.0.0.1:7243/ingest/c331841f-7e4f-4c50-9c5e-a68c9827234e',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'voice.js:185',message:'speakGoogle response',data:{status:response.status,ok:response.ok,pid},timestamp:Date.now(),sessionId:'debug-session',hypothesisId:'2'})}).catch(()=>{});
            // #endregion

            if (!response.ok) return;
            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let buffer = '';

            while (true) {
                if (this.playbackId !== pid) {
                    // #region agent log
                    fetch('http://127.0.0.1:7243/ingest/c331841f-7e4f-4c50-9c5e-a68c9827234e',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'voice.js:195',message:'speakGoogle cancelled due to playbackId mismatch',data:{current:this.playbackId,pid},timestamp:Date.now(),sessionId:'debug-session',hypothesisId:'2'})}).catch(()=>{});
                    // #endregion
                    await reader.cancel();
                    return;
                }
                const { done, value } = await reader.read();
                if (done) break;

                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split('\n');
                buffer = lines.pop() || '';

                for (const line of lines) {
                    if (!line.trim() || this.playbackId !== pid) continue;
                    try {
                        const binary = atob(line);
                        const bytes = new Uint8Array(binary.length);
                        for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);

                        // Collect all decoded bytes into completeAudioData
                        const combined = new Uint8Array(this.completeAudioData.length + bytes.length);
                        combined.set(this.completeAudioData);
                        combined.set(bytes, this.completeAudioData.length);
                        this.completeAudioData = combined;

                        // #region agent log
                        fetch('http://127.0.0.1:7243/ingest/c331841f-7e4f-4c50-9c5e-a68c9827234e',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'voice.js:205',message:'collected decoded chunk',data:{bytesLen:bytes.length,totalLen:this.completeAudioData.length},timestamp:Date.now(),sessionId:'debug-session',hypothesisId:'2'})}).catch(()=>{});
                        // #endregion

                    } catch (e) {
                        // #region agent log
                        fetch('http://127.0.0.1:7243/ingest/c331841f-7e4f-4c50-9c5e-a68c9827234e',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'voice.js:230',message:'speakGoogle atob error',data:{err:e.message},timestamp:Date.now(),sessionId:'debug-session',hypothesisId:'2'})}).catch(()=>{});
                        // #endregion
                    }
                }
            }

            // All chunks collected and decoded, now decode audio and play
            if (this.playbackId === pid && this.completeAudioData.length > 0) {
                await this.decodeAndPlayCompleteAudio(pid);
            }
            
            const check = () => {
                if (this.playbackId !== pid) return;
                if (!this.synth.speaking && this.audioQueue.length > 0 && !this.currentSource) {
                    this.playNextChunk(pid);
                } else if (this.synth.speaking || this.currentSource || this.audioQueue.length > 0) {
                    setTimeout(check, 100);
                }
            };
            check();
        } catch (e) {
            // #region agent log
            fetch('http://127.0.0.1:7243/ingest/c331841f-7e4f-4c50-9c5e-a68c9827234e',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'voice.js:245',message:'speakGoogle fatal error',data:{err:e.message},timestamp:Date.now(),sessionId:'debug-session',hypothesisId:'2'})}).catch(()=>{});
            // #endregion
        }
    }

    async decodeAndPlayCompleteAudio(pid) {
        if (this.playbackId !== pid || this.completeAudioData.length === 0) {
            return;
        }

        try {
            if (this.audioContext.state === 'suspended') await this.audioContext.resume();

            // Decode the complete audio data using Web Audio API
            const audioBuffer = await this.audioContext.decodeAudioData(this.completeAudioData.buffer.slice(this.completeAudioData.byteOffset, this.completeAudioData.byteOffset + this.completeAudioData.byteLength));

            const source = this.audioContext.createBufferSource();
            source.buffer = audioBuffer;
            source.connect(this.audioContext.destination);
            this.currentSource = source;

            source.onended = () => {
                // #region agent log
                fetch('http://127.0.0.1:7243/ingest/c331841f-7e4f-4c50-9c5e-a68c9827234e',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'voice.js:280',message:'complete audio ended',data:{pid},timestamp:Date.now(),sessionId:'debug-session',hypothesisId:'3'})}).catch(()=>{});
                // #endregion
                this.currentSource = null;
                this.completeAudioData = new Uint8Array(0);
            };

            source.start();
            this.updateStatus('Озвучка (Gemini)...');

            // #region agent log
            fetch('http://127.0.0.1:7243/ingest/c331841f-7e4f-4c50-9c5e-a68c9827234e',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'voice.js:290',message:'started complete audio playback',data:{audioLen:this.completeAudioData.length,duration:audioBuffer.duration},timestamp:Date.now(),sessionId:'debug-session',hypothesisId:'3'})}).catch(()=>{});
            // #endregion

        } catch (e) {
            console.error('[TTS] Audio decoding error:', e);
            // #region agent log
            fetch('http://127.0.0.1:7243/ingest/c331841f-7e4f-4c50-9c5e-a68c9827234e',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'voice.js:285',message:'complete audio decode error',data:{err:e.message,audioLen:this.completeAudioData.length},timestamp:Date.now(),sessionId:'debug-session',hypothesisId:'3'})}).catch(()=>{});
            // #endregion
            this.completeAudioData = new Uint8Array(0);
        }
    }

    async playNextChunk(pid) {
        if (this.playbackId !== pid || this.audioQueue.length === 0) {
            // #region agent log
            fetch('http://127.0.0.1:7243/ingest/c331841f-7e4f-4c50-9c5e-a68c9827234e',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'voice.js:260',message:'playNextChunk exit',data:{pid,match:this.playbackId===pid,queueLen:this.audioQueue.length},timestamp:Date.now(),sessionId:'debug-session',hypothesisId:'3'})}).catch(()=>{});
            // #endregion
            return;
        }
        
        const chunk = this.audioQueue.shift();
        // #region agent log
        fetch('http://127.0.0.1:7243/ingest/c331841f-7e4f-4c50-9c5e-a68c9827234e',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'voice.js:265',message:'playNextChunk starting',data:{chunkLen:chunk.length,queueLen:this.audioQueue.length},timestamp:Date.now(),sessionId:'debug-session',hypothesisId:'3'})}).catch(()=>{});
        // #endregion
        try {
            if (this.audioContext.state === 'suspended') await this.audioContext.resume();

            // Use Web Audio API decodeAudioData for proper audio file decoding
            // Gemini returns complete audio files (MP3/WAV), not raw PCM
            const audioBuffer = await this.audioContext.decodeAudioData(chunk.buffer.slice(chunk.byteOffset, chunk.byteOffset + chunk.byteLength));

            const source = this.audioContext.createBufferSource();
            source.buffer = audioBuffer;
            source.connect(this.audioContext.destination);
            this.currentSource = source;

            source.onended = () => {
                // #region agent log
                fetch('http://127.0.0.1:7243/ingest/c331841f-7e4f-4c50-9c5e-a68c9827234e',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'voice.js:280',message:'audio source ended',data:{pid},timestamp:Date.now(),sessionId:'debug-session',hypothesisId:'3'})}).catch(()=>{});
                // #endregion
                this.currentSource = null;
                this.playNextChunk(pid);
            };

            source.start();
            this.updateStatus('Озвучка (Quality)...');

        } catch (e) {
            console.error('[TTS] Audio decoding error:', e);
            // #region agent log
            fetch('http://127.0.0.1:7243/ingest/c331841f-7e4f-4c50-9c5e-a68c9827234e',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'voice.js:285',message:'audio decode error',data:{err:e.message,chunkLen:chunk.length},timestamp:Date.now(),sessionId:'debug-session',hypothesisId:'3'})}).catch(()=>{});
            // #endregion
            this.playNextChunk(pid);
        }
    }

    cleanText(text) {
        if (!text) return '';
        return text
            .replace(/[\u{1F300}-\u{1F9FF}]/gu, '') // Emojis
            .replace(/[📊🔍🎯⚠️💡📜🤖═─│┌┐└┘├┤┬┴┼]/g, '') // Special chars
            .replace(/[*#_~`]/g, '') // Markdown symbols
            .replace(/\n+/g, ' ') // Multiple newlines to single space
            .trim();
    }

    async startRecording() {
        if (this.isRecording) return;
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            this.recordedChunks = [];
            this.mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm;codecs=opus' });
            this.mediaRecorder.ondataavailable = (e) => { if (e.data.size > 0) this.recordedChunks.push(e.data); };
            this.mediaRecorder.onstop = async () => {
                stream.getTracks().forEach(t => t.stop());
                const blob = new Blob(this.recordedChunks, { type: 'audio/webm' });
                const fd = new FormData();
                fd.append('file', blob, 'rec.webm');
                fd.append('language', 'ru');
                const resp = await fetch('/api/voice/stt', { method: 'POST', body: fd });
                const data = await resp.json();
                if (data.text) this.onVoiceInput?.(data.text, this.lastAnalysisText);
            };
            this.mediaRecorder.start();
            this.isRecording = true;
            if (this.elements.recordBtn) this.elements.recordBtn.classList.add('recording');
            this.updateStatus('Слушаю...');
        } catch (e) {}
    }
    
    stopRecording() {
        if (!this.isRecording) return;
        this.mediaRecorder.stop();
        this.isRecording = false;
        if (this.elements.recordBtn) this.elements.recordBtn.classList.remove('recording');
    }
    
    setVoiceInputCallback(cb) { this.onVoiceInput = cb; }
    updateStatus(t) {
        if (!this.elements.statusEl) return;
        
        if (t.includes('Quality') || t.includes('Fast')) {
            // Show mesmerizing loader during speech generation/playback
            this.elements.statusEl.innerHTML = `
                <div class="loader-geometry" style="width: 20px; height: 20px; display: inline-block; vertical-align: middle; margin-right: 8px;">
                    <span></span><span></span><span></span>
                </div>
                <span>${t}</span>
            `;
            this.elements.statusEl.classList.add('speaking');
        } else {
            this.elements.statusEl.textContent = t;
            this.elements.statusEl.classList.remove('speaking');
        }
    }
}

const voiceService = new VoiceService();
