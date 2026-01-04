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
        this.voiceMode = 'fast'; // 'fast' (Native) or 'cool' (Gemini TTS)
        this.sentenceQueue = []; // Queue for sequential playback
        this.isPlayingSentence = false; // Track if currently playing a sentence
        
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
        // Show voice mode selection menu
        if (!this.isVoiceEnabled) {
            // If voice is disabled, just enable it with current mode
            this.isVoiceEnabled = true;
            if (this.elements.toggleBtn) {
                this.elements.toggleBtn.classList.add('voice-on');
            }
            return;
        }

        // If voice is enabled, show mode selection
        this.showVoiceModeMenu();
    }

    showVoiceModeMenu() {
        // Remove existing menu if any
        const existingMenu = document.getElementById('voiceModeMenu');
        if (existingMenu) {
            existingMenu.remove();
        }

        // Create new menu
        const menu = document.createElement('div');
        menu.id = 'voiceModeMenu';
        menu.className = 'voice-mode-menu';
            menu.innerHTML = `
                <div class="voice-mode-option ${this.voiceMode === 'fast' ? 'active' : ''}" data-mode="fast">
                    <span class="mode-icon">⚡</span>
                    <span class="mode-name">Fast (Native)</span>
                </div>
                <div class="voice-mode-option ${this.voiceMode === 'cool' ? 'active' : ''}" data-mode="cool">
                    <span class="mode-icon">🎵</span>
                    <span class="mode-name">Cool (Gemini TTS)</span>
                </div>
            `;
            document.body.appendChild(menu);

            // Position menu near button
            if (this.elements.toggleBtn) {
                const rect = this.elements.toggleBtn.getBoundingClientRect();
                menu.style.position = 'fixed';
                menu.style.top = `${rect.bottom + window.scrollY + 5}px`;
                menu.style.left = `${rect.left + window.scrollX}px`;
            }

            // Handle clicks
            menu.querySelectorAll('.voice-mode-option').forEach(option => {
                option.addEventListener('click', (e) => {
                    const mode = e.currentTarget.dataset.mode;
                    this.voiceMode = mode;
                    this.sentenceIndex = 0; // Reset for new mode
                    menu.remove();
                    this.updateStatus(`Режим: ${mode === 'fast' ? 'Fast' : 'Cool'}`);
                });
            });

            // Close menu on outside click
            setTimeout(() => {
                const closeMenu = (e) => {
                    if (!menu.contains(e.target) && e.target !== this.elements.toggleBtn) {
                        menu.remove();
                        document.removeEventListener('click', closeMenu);
                    }
                };
                document.addEventListener('click', closeMenu);
            }, 100);
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
        this.currentAudioData = new Uint8Array(0);
        this.audioChunks = [];
        this.sentenceIndex = 0;
        this.sentenceQueue = []; // Clear sentence queue
        this.isPlayingSentence = false; // Reset playing state
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
        fetch('http://127.0.0.1:7243/ingest/c331841f-7e4f-4c50-9c5e-a68c9827234e',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'voice.js:125',message:'queueSpeak',data:{sentence,sentenceIndex:this.sentenceIndex,playbackId:this.playbackId,voiceMode:this.voiceMode},timestamp:Date.now(),sessionId:'debug-session',hypothesisId:'1'})}).catch(()=>{});
        // #endregion
        if (!this.isVoiceEnabled || !sentence) return;
        
        // Add sentence to queue
        this.sentenceQueue.push(sentence);
        this.sentenceIndex++;
        
        // Process queue if not already playing
        if (!this.isPlayingSentence) {
            this.processSentenceQueue();
        }
    }

    async processSentenceQueue() {
        if (this.sentenceQueue.length === 0) {
            this.isPlayingSentence = false;
            return;
        }

        this.isPlayingSentence = true;
        const sentence = this.sentenceQueue.shift();
        const pid = this.playbackId;

        // Use selected voice mode
        if (this.voiceMode === 'fast') {
            // Wait for Native to finish
            await this.speakNativeAsync(sentence, pid);
        } else {
            // Cool mode: Use Gemini TTS with chunking
            await this.speakGoogleAsync(sentence, pid);
        }

        // Process next sentence
        if (this.playbackId === pid) {
            this.processSentenceQueue();
        } else {
            this.isPlayingSentence = false;
        }
    }

    async speakNativeAsync(text, pid) {
        return new Promise((resolve) => {
            const clean = this.cleanText(text);
            if (!clean) {
                resolve();
                return;
            }

            const utterance = new SpeechSynthesisUtterance(clean);
            if (this.nativeVoice) utterance.voice = this.nativeVoice;
            utterance.lang = 'ru-RU';
            utterance.rate = 1.1;

            utterance.onstart = () => {
                if (this.playbackId !== pid) {
                    this.synth.cancel();
                    resolve();
                    return;
                }
                this.updateStatus('Озвучка (Fast)...');
                if (this.elements.pauseBtn) this.elements.pauseBtn.disabled = false;
            };

            utterance.onend = () => {
                resolve();
            };

            utterance.onerror = () => {
                resolve();
            };

            this.synth.speak(utterance);
        });
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

    async speakGoogleAsync(text, pid) {
        return new Promise((resolve) => {
            this.speakGoogle(text, pid, resolve);
        });
    }

    async speakGoogle(text, pid, onComplete = null) {
        const clean = this.cleanText(text);
        if (!clean) {
            if (onComplete) onComplete();
            return;
        }

        // #region agent log
        fetch('http://127.0.0.1:7243/ingest/c331841f-7e4f-4c50-9c5e-a68c9827234e',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'voice.js:175',message:'speakGoogle entry',data:{clean,pid,sentenceIndex:this.sentenceIndex},timestamp:Date.now(),sessionId:'debug-session',hypothesisId:'2'})}).catch(()=>{});
        // #endregion

        this.initAudioContext();
        this.currentAudioData = new Uint8Array(0); // Reset for new audio
        this.audioChunks = []; // Queue of audio chunks to play

        try {
            const response = await fetch('/api/voice/tts/stream', {
                method: 'POST',
                headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
                body: `text=${encodeURIComponent(clean)}`
            });

            // #region agent log
            fetch('http://127.0.0.1:7243/ingest/c331841f-7e4f-4c50-9c5e-a68c9827234e',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'voice.js:185',message:'speakGoogle response',data:{status:response.status,ok:response.ok,pid},timestamp:Date.now(),sessionId:'debug-session',hypothesisId:'2'})}).catch(()=>{});
            // #endregion

            if (!response.ok) {
                if (onComplete) onComplete();
                return;
            }
            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let buffer = '';

            while (true) {
                if (this.playbackId !== pid) {
                    // #region agent log
                    fetch('http://127.0.0.1:7243/ingest/c331841f-7e4f-4c50-9c5e-a68c9827234e',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'voice.js:195',message:'speakGoogle cancelled due to playbackId mismatch',data:{current:this.playbackId,pid},timestamp:Date.now(),sessionId:'debug-session',hypothesisId:'2'})}).catch(()=>{});
                    // #endregion
                    await reader.cancel();
                    if (onComplete) onComplete();
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

                        // Add to current audio data
                        const combined = new Uint8Array(this.currentAudioData.length + bytes.length);
                        combined.set(this.currentAudioData);
                        combined.set(bytes, this.currentAudioData.length);
                        this.currentAudioData = combined;

                        // Try to decode and play incrementally
                        if (this.currentAudioData.length > 0) {
                            await this.tryDecodeAndPlayIncremental(pid);
                        }

                        // #region agent log
                        fetch('http://127.0.0.1:7243/ingest/c331841f-7e4f-4c50-9c5e-a68c9827234e',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'voice.js:205',message:'processed chunk',data:{bytesLen:bytes.length,totalLen:this.currentAudioData.length},timestamp:Date.now(),sessionId:'debug-session',hypothesisId:'2'})}).catch(()=>{});
                        // #endregion

                    } catch (e) {
                        // #region agent log
                        fetch('http://127.0.0.1:7243/ingest/c331841f-7e4f-4c50-9c5e-a68c9827234e',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'voice.js:230',message:'speakGoogle atob error',data:{err:e.message},timestamp:Date.now(),sessionId:'debug-session',hypothesisId:'2'})}).catch(()=>{});
                        // #endregion
                    }
                }
            }

            // Final attempt to decode any remaining data
            if (this.playbackId === pid && this.currentAudioData.length > 0) {
                await this.tryDecodeAndPlayIncremental(pid, true);
            }
            
            // Wait for all audio to finish playing
            const check = () => {
                if (this.playbackId !== pid) {
                    if (onComplete) onComplete();
                    return;
                }
                // In cool mode, we only check audioQueue (no synth.speaking check)
                if (this.audioQueue.length > 0 || this.currentSource) {
                    setTimeout(check, 100);
                } else {
                    // All audio finished
                    if (onComplete) onComplete();
                }
            };
            check();
        } catch (e) {
            // #region agent log
            fetch('http://127.0.0.1:7243/ingest/c331841f-7e4f-4c50-9c5e-a68c9827234e',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'voice.js:245',message:'speakGoogle fatal error',data:{err:e.message},timestamp:Date.now(),sessionId:'debug-session',hypothesisId:'2'})}).catch(()=>{});
            // #endregion
            if (onComplete) onComplete();
        }
    }

    async tryDecodeAndPlayIncremental(pid, isFinal = false) {
        if (this.playbackId !== pid || this.currentAudioData.length === 0) {
            return;
        }

        // For WAV files, we need at least the header (44 bytes) + some audio data
        if (!isFinal && this.currentAudioData.length < 1000) { // Need at least header + some data
            return;
        }

        try {
            if (this.audioContext.state === 'suspended') await this.audioContext.resume();

            // Try to decode the current WAV data
            const audioBuffer = await this.audioContext.decodeAudioData(
                this.currentAudioData.buffer.slice(0, this.currentAudioData.length)
            );

            // If successful, create and play audio source
            const source = this.audioContext.createBufferSource();
            source.buffer = audioBuffer;
            source.connect(this.audioContext.destination);
            this.currentSource = source;

            source.onended = () => {
                // #region agent log
                fetch('http://127.0.0.1:7243/ingest/c331841f-7e4f-4c50-9c5e-a68c9827234e',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'voice.js:280',message:'incremental audio ended',data:{pid},timestamp:Date.now(),sessionId:'debug-session',hypothesisId:'3'})}).catch(()=>{});
                // #endregion
                this.currentSource = null;
            };

            source.start();
            this.updateStatus('Озвучка (Gemini)...');

            // Reset current data after successful decode
            this.currentAudioData = new Uint8Array(0);

            // #region agent log
            fetch('http://127.0.0.1:7243/ingest/c331841f-7e4f-4c50-9c5e-a68c9827234e',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'voice.js:290',message:'started incremental WAV playback',data:{duration:audioBuffer.duration},timestamp:Date.now(),sessionId:'debug-session',hypothesisId:'3'})}).catch(()=>{});
            // #endregion

        } catch (e) {
            // If decoding failed and it's not final, just wait for more data
            if (!isFinal) {
                // #region agent log
                fetch('http://127.0.0.1:7243/ingest/c331841f-7e4f-4c50-9c5e-a68c9827234e',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'voice.js:285',message:'incremental WAV decode failed, waiting for more data',data:{err:e.message,dataLen:this.currentAudioData.length},timestamp:Date.now(),sessionId:'debug-session',hypothesisId:'3'})}).catch(()=>{});
                // #endregion
                return;
            }

            // If it's final and still failed, try to save the data for debugging
            console.error('[TTS] Final audio decoding error:', e);
            // Save failed audio data for debugging
            try {
                const blob = new Blob([this.currentAudioData], { type: 'audio/wav' });
                const url = URL.createObjectURL(blob);
                console.log('[TTS] Failed audio data saved as blob:', url);
            } catch (saveErr) {
                console.error('[TTS] Could not save debug audio:', saveErr);
            }

            // #region agent log
            fetch('http://127.0.0.1:7243/ingest/c331841f-7e4f-4c50-9c5e-a68c9827234e',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({location:'voice.js:285',message:'final incremental WAV decode error',data:{err:e.message,dataLen:this.currentAudioData.length},timestamp:Date.now(),sessionId:'debug-session',hypothesisId:'3'})}).catch(()=>{});
            // #endregion
            this.currentAudioData = new Uint8Array(0);
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
