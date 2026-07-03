import os
import re
import json
import numpy as np
import librosa
import soundfile as sf
import torch
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util

# Global variables for caching models to avoid reloading on every run in Streamlit
_asr_pipeline = None
_sentence_model = None

def get_asr_pipeline(model_name="openai/whisper-tiny"):
    """
    Load or retrieve cached Hugging Face Automatic Speech Recognition pipeline.
    Uses whisper-tiny by default for speed and low memory footprint.
    """
    global _asr_pipeline
    if _asr_pipeline is None:
        device = 0 if torch.cuda.is_available() else -1
        # Set cache dir in workspace if needed, or let transformers handle it
        _asr_pipeline = pipeline(
            "automatic-speech-recognition",
            model=model_name,
            device=device
        )
    return _asr_pipeline

def get_sentence_model(model_name="all-MiniLM-L6-v2"):
    """
    Load or retrieve cached Sentence-Transformer model.
    """
    global _sentence_model
    if _sentence_model is None:
        _sentence_model = SentenceTransformer(model_name)
    return _sentence_model

def load_concepts(concepts_path):
    """
    Load predefined concepts from the JSON database.
    """
    if os.path.exists(concepts_path):
        with open(concepts_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_concepts(concepts_path, concepts):
    """
    Save concepts to the JSON database.
    """
    with open(concepts_path, "w", encoding="utf-8") as f:
        json.dump(concepts, f, indent=2)

def transcribe_audio(audio_path, model_name="openai/whisper-tiny"):
    """
    Transcribe audio file to text using Hugging Face Whisper pipeline.
    """
    try:
        asr = get_asr_pipeline(model_name)
        # Load audio using librosa or soundfile to ensure correct sample rate for whisper
        y, sr = librosa.load(audio_path, sr=16000)
        # Pass the raw numpy array directly to whisper
        result = asr({"raw": y, "sampling_rate": sr})
        return result["text"].strip()
    except Exception as e:
        return f"Transcription error: {str(e)}"

def split_into_sentences(text):
    """
    Simple sentence splitter based on punctuation.
    """
    # Split by period, question mark, or exclamation mark followed by whitespace
    sentences = re.split(r'(?<=[.!?])\s+', text)
    # Filter out empty strings
    return [s.strip() for s in sentences if s.strip()]

def evaluate_concept_understanding(user_transcript, concept_data, similarity_threshold=0.45):
    """
    Evaluates understanding of a concept:
    1. Overall semantic similarity between user transcript and reference description.
    2. Check coverage of expected key points using sentence embeddings.
    3. Match key terms/keywords.
    """
    if not user_transcript or "error" in user_transcript.lower():
        return {
            "similarity_score": 0.0,
            "understanding_level": "N/A (No speech detected)",
            "key_points_status": [],
            "keywords_found": [],
            "keywords_missing": []
        }

    sentence_model = get_sentence_model()

    # 1. Overall Semantic Similarity
    ref_desc = concept_data.get("description", "")
    user_emb = sentence_model.encode(user_transcript, convert_to_tensor=True)
    ref_emb = sentence_model.encode(ref_desc, convert_to_tensor=True)
    overall_sim = float(util.cos_sim(user_emb, ref_emb).item())

    # Map overall similarity to qualitative level
    # Cosine similarity for MiniLM ranges around 0.3-1.0 for related texts
    if overall_sim >= 0.70:
        level = "Strong Understanding"
    elif overall_sim >= 0.45:
        level = "Moderate Understanding"
    else:
        level = "Poor Understanding"

    # 2. Key Points Coverage
    # Segment user transcript into sentences
    user_sentences = split_into_sentences(user_transcript)
    key_points = concept_data.get("key_points", [])
    
    key_points_status = []
    if key_points and user_sentences:
        # Encode user sentences and reference key points
        user_sentence_embs = sentence_model.encode(user_sentences, convert_to_tensor=True)
        key_point_embs = sentence_model.encode(key_points, convert_to_tensor=True)
        
        # Calculate similarity matrix: shape (num_key_points, num_user_sentences)
        sim_matrix = util.cos_sim(key_point_embs, user_sentence_embs)
        
        for idx, kp in enumerate(key_points):
            # Find the best matching user sentence
            max_sim_idx = int(torch.argmax(sim_matrix[idx]).item())
            best_sim = float(sim_matrix[idx][max_sim_idx].item())
            best_sentence = user_sentences[max_sim_idx]
            
            if best_sim >= similarity_threshold:
                status = "Covered"
            elif best_sim >= (similarity_threshold - 0.15):
                status = "Partially Covered"
            else:
                status = "Missed"
                best_sentence = ""
                
            key_points_status.append({
                "key_point": kp,
                "status": status,
                "similarity": best_sim,
                "matching_sentence": best_sentence
            })
    else:
        for kp in key_points:
            key_points_status.append({
                "key_point": kp,
                "status": "Missed",
                "similarity": 0.0,
                "matching_sentence": ""
            })

    # 3. Keyword Matching (Simple substring matching in transcript, case insensitive)
    keywords = concept_data.get("keywords", [])
    keywords_found = []
    keywords_missing = []
    
    clean_transcript = re.sub(r'[^\w\s]', '', user_transcript.lower())
    words = set(clean_transcript.split())
    
    for kw in keywords:
        # Check for multi-word or single-word keywords
        kw_clean = kw.lower().strip()
        if kw_clean in clean_transcript:
            keywords_found.append(kw)
        else:
            keywords_missing.append(kw)

    return {
        "similarity_score": overall_sim,
        "understanding_level": level,
        "key_points_status": key_points_status,
        "keywords_found": keywords_found,
        "keywords_missing": keywords_missing
    }

def analyze_audio_fluency(audio_path, user_transcript, pause_threshold_db=-35, min_pause_sec=0.4):
    """
    Evaluates speech fluency and features from audio:
    1. Loads audio and calculates duration.
    2. Calculates RMS energy levels over time.
    3. Detects silent intervals (pauses) based on RMS energy threshold.
    4. Computes Pause Ratio.
    5. Counts filler words in the transcript.
    6. Calculates speaking rate (WPM).
    7. Computes a composite fluency score.
    """
    # 1. Load Audio
    y, sr = librosa.load(audio_path, sr=None)
    duration = librosa.get_duration(y=y, sr=sr)
    
    # 2. RMS Energy calculation
    frame_length = 2048
    hop_length = 512
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    
    # Avoid zero division when converting to dB
    rms_safe = np.maximum(rms, 1e-10)
    rms_db = librosa.amplitude_to_db(rms_safe, ref=np.max)
    
    # Get timestamps for each frame
    times = librosa.frames_to_time(range(len(rms)), sr=sr, hop_length=hop_length)
    
    # 3. Pause Detection
    is_silent = rms_db < pause_threshold_db
    pauses = []
    in_pause = False
    pause_start = 0
    
    for idx, silent in enumerate(is_silent):
        t = times[idx]
        if silent and not in_pause:
            in_pause = True
            pause_start = t
        elif not silent and in_pause:
            in_pause = False
            pause_dur = t - pause_start
            if pause_dur >= min_pause_sec:
                pauses.append({"start": pause_start, "end": t, "duration": pause_dur})
    
    if in_pause:
        pause_dur = times[-1] - pause_start
        if pause_dur >= min_pause_sec:
            pauses.append({"start": pause_start, "end": times[-1], "duration": pause_dur})
            
    total_pause_time = sum([p["duration"] for p in pauses])
    pause_count = len(pauses)
    
    # Calculate Pause Ratio
    pause_ratio = (total_pause_time / duration) if duration > 0 else 0.0
    
    # 4. Count Filler Words in Transcript
    # Common fillers to track
    filler_words = ["um", "uh", "like", "so", "you know", "actually", "er", "ah"]
    filler_counts = {word: 0 for word in filler_words}
    
    # Clean transcript for search
    clean_text = re.sub(r'[^\w\s\']', ' ', user_transcript.lower())
    # Match multi-word fillers like "you know" as well
    for word in filler_words:
        # Create a regex to match the exact word/phrase boundary
        pattern = r'\b' + re.escape(word) + r'\b'
        matches = re.findall(pattern, clean_text)
        filler_counts[word] = len(matches)
        
    total_fillers = sum(filler_counts.values())
    
    # Calculate words in transcript
    words_list = clean_text.split()
    total_words = len(words_list)
    
    # Speaking rate (WPM)
    duration_min = duration / 60.0 if duration > 0 else 0.0
    wpm = (total_words / duration_min) if duration_min > 0 else 0.0
    
    # Filler word density (fillers per 100 words)
    filler_density = (total_fillers / total_words * 100) if total_words > 0 else 0.0
    
    # 5. Composite Fluency Scoring (0-100)
    # Rules:
    # - WPM: ideal 110-150. Deduct points if outside range.
    # - Pause ratio: ideal 0.10-0.25 (10%-25%). Deduct if too high (hesitation) or too low (no breathing space).
    # - Filler word density: ideal < 2. Deduct points proportionally to higher densities.
    
    fluency_score = 100.0
    
    # WPM scoring
    if wpm < 80:
        fluency_score -= min(30.0, (80 - wpm) * 0.75)  # Penalize speaking too slowly
    elif wpm > 160:
        fluency_score -= min(25.0, (wpm - 160) * 0.5)  # Penalize speaking too quickly
        
    # Pause ratio scoring
    if pause_ratio > 0.35:
        fluency_score -= min(30.0, (pause_ratio - 0.35) * 100)  # Penalize high hesitation pauses
    elif pause_ratio < 0.05 and duration > 5:
        fluency_score -= 15.0  # Penalize speaking without any pauses
        
    # Filler word density scoring
    if filler_density > 2.0:
        fluency_score -= min(30.0, (filler_density - 2.0) * 8.0)
        
    fluency_score = max(10.0, min(100.0, fluency_score))
    
    # Generate qualitative fluency feedback
    if fluency_score >= 85:
        fluency_feedback = "Excellent fluency: Smooth pacing, natural pauses, and minimal filler words."
    elif fluency_score >= 65:
        fluency_feedback = "Moderate fluency: Pacing is acceptable, but there are noticeable hesitations, long pauses, or frequent filler words."
    else:
        fluency_feedback = "Needs Improvement: Frequent pauses, high usage of fillers, or an irregular speech rate impacts clarity."

    return {
        "duration": duration,
        "total_words": total_words,
        "wpm": wpm,
        "rms_levels": rms_db.tolist(),
        "rms_times": times.tolist(),
        "pauses": pauses,
        "pause_count": pause_count,
        "total_pause_time": total_pause_time,
        "pause_ratio": pause_ratio,
        "filler_counts": filler_counts,
        "total_fillers": total_fillers,
        "filler_density": filler_density,
        "fluency_score": fluency_score,
        "fluency_feedback": fluency_feedback,
        "waveform_y": y.tolist() if len(y) < 160000 else y[::10].tolist(),  # downsample long waveforms for speed
        "waveform_sr": sr if len(y) < 160000 else sr // 10
    }
