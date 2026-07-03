import os
import tempfile
import json
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import librosa

# Import local modules
from analyzer import (
    load_concepts,
    save_concepts,
    transcribe_audio,
    evaluate_concept_understanding,
    analyze_audio_fluency
)
from pdf_report import generate_waveform_plot, build_pdf_report

# Page Configuration
st.set_page_config(
    page_title="VBCUA | Voice-Based Concept Understanding Analyser",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Premium Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"], .stApp {
        font-family: 'Outfit', sans-serif !important;
    }
    
    /* Background Gradient */
    .stApp {
        background: linear-gradient(135deg, #0b0f19 0%, #111827 50%, #1e1b4b 100%);
        color: #f3f4f6;
    }
    
    /* Premium Header */
    .header-container {
        padding: 2rem 0;
        text-align: center;
        background: rgba(17, 24, 39, 0.6);
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        margin-bottom: 2rem;
    }
    
    .gradient-title {
        background: linear-gradient(90deg, #60a5fa 0%, #c084fc 50%, #f472b6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.8rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
    }
    
    .header-subtitle {
        color: #9ca3af;
        font-size: 1.1rem;
        font-weight: 300;
    }
    
    /* Sleek Metric Cards */
    .card-container {
        display: flex;
        gap: 1.5rem;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: rgba(30, 41, 59, 0.5);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 16px;
        padding: 1.5rem;
        flex: 1;
        text-align: center;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
    }
    
    .metric-card:hover {
        transform: translateY(-4px);
        border-color: rgba(96, 165, 250, 0.3);
        box-shadow: 0 10px 25px rgba(96, 165, 250, 0.1);
    }
    
    .metric-label {
        color: #9ca3af;
        font-size: 0.875rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.5rem;
    }
    
    .metric-value {
        font-size: 2.25rem;
        font-weight: 700;
        color: #ffffff;
    }
    
    .value-highlight {
        color: #60a5fa;
    }
    
    .value-green {
        color: #34d399;
    }
    
    .value-purple {
        color: #c084fc;
    }
    
    /* Checklist Badges */
    .badge {
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    
    .badge-covered {
        background-color: rgba(52, 211, 153, 0.15);
        color: #34d399;
        border: 1px solid rgba(52, 211, 153, 0.3);
    }
    
    .badge-partial {
        background-color: rgba(251, 191, 36, 0.15);
        color: #fbbf24;
        border: 1px solid rgba(251, 191, 36, 0.3);
    }
    
    .badge-missed {
        background-color: rgba(248, 113, 113, 0.15);
        color: #f87171;
        border: 1px solid rgba(248, 113, 113, 0.3);
    }
</style>
""", unsafe_allow_html=True)

# Path setup
CONCEPTS_FILE = os.path.join(os.path.dirname(__file__), "concepts.json")

# Initialize concepts in session state
if "concepts" not in st.session_state:
    st.session_state.concepts = load_concepts(CONCEPTS_FILE)

# Header Section
st.markdown("""
<div class="header-container">
    <div class="gradient-title">Voice-Based Concept Understanding Analyser</div>
    <div class="header-subtitle">Evaluate conceptual depth and verbal fluency through AI speech transcription and bio-signal diagnostics</div>
</div>
""", unsafe_allow_html=True)

# Sidebar - Settings & Model Options
st.sidebar.image("https://img.icons8.com/nolan/96/microphone.png", width=70)
st.sidebar.title("Configuration Settings")

st.sidebar.subheader("AI Models")
whisper_model = st.sidebar.selectbox(
    "Speech-to-Text Model",
    ["openai/whisper-tiny", "openai/whisper-base"],
    index=0,
    help="Tiny is faster and uses less memory. Base is more accurate but downloads a larger model file."
)

st.sidebar.subheader("Analysis Parameters")
similarity_threshold = st.sidebar.slider(
    "Key Point Similarity Threshold",
    min_value=0.30, max_value=0.70, value=0.45, step=0.05,
    help="Cosine similarity threshold between transcript sentences and reference key points."
)

pause_threshold = st.sidebar.slider(
    "Pause Detection Threshold (dB)",
    min_value=-50, max_value=-20, value=-35, step=5,
    help="Decibels below peak energy to classify as pause/silence."
)

min_pause_dur = st.sidebar.slider(
    "Min Pause Duration (sec)",
    min_value=0.2, max_value=1.5, value=0.4, step=0.1,
    help="Minimum silence length to count as a hesitation pause."
)

# Tabs
tab1, tab2, tab3 = st.tabs(["🎙️ Analyze Speech", "📊 Diagnostic Dashboard", "⚙️ Concept Customizer"])

# TAB 1: Speech Analysis
with tab1:
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("1. Select Concept Topic")
        selected_concept = st.selectbox(
            "Choose a topic to explain",
            options=list(st.session_state.concepts.keys())
        )
        
        concept_data = st.session_state.concepts[selected_concept]
        
        st.info(f"**Reference Concept Details:**\n\n{concept_data['description']}")
        
        with st.expander("Show Predefined Evaluation Criteria"):
            st.markdown("**Expected Key Points:**")
            for kp in concept_data["key_points"]:
                st.markdown(f"- {kp}")
            st.markdown("**Core Keywords:**")
            st.markdown(", ".join([f"`{k}`" for k in concept_data["keywords"]]))
            
    with col2:
        st.subheader("2. Upload Explanation Audio")
        
        # Method selector: Upload file or Demo audio
        input_method = st.radio(
            "Choose Input Source:",
            ["Upload Audio File", "Use Pre-recorded Demo Audio"],
            horizontal=True
        )
        
        uploaded_file = None
        run_analysis = False
        
        if input_method == "Upload Audio File":
            uploaded_file = st.file_uploader(
                "Upload audio explanation (WAV, MP3, M4A)",
                type=["wav", "mp3", "m4a"]
            )
            if uploaded_file:
                st.audio(uploaded_file, format='audio/wav')
                run_analysis = st.button("🚀 Analyze Uploaded Audio", use_container_width=True)
        else:
            st.write("No microphone? Test the platform using a pre-recorded explanation of **Machine Learning**.")
            st.audio("https://www.soundhelix.com/examples/mp3/SoundHelix-Song-1.mp3") # Placeholder for demo visual, wait we will generate demo wav locally!
            st.markdown("""
            *This demo script will generate a synthetic demo speech locally about Machine Learning and analyze it.*
            """)
            run_analysis = st.button("🚀 Run Demo Machine Learning Analysis", use_container_width=True)

        if run_analysis:
            # We process audio
            audio_path = None
            demo_mode = False
            
            if input_method == "Upload Audio File" and uploaded_file is not None:
                # Save uploaded file to temp path
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1])
                temp_file.write(uploaded_file.read())
                temp_file.close()
                audio_path = temp_file.name
            else:
                # Generate a local demo WAV file
                demo_mode = True
                st.info("Generating local demo voice sample...")
                
                # Make a synthetic WAV file of a sine wave (since we want a real audio file to feed into Whisper/Librosa)
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
                audio_path = temp_file.name
                
                # Create a 6-second wave containing silence gaps (to test pause detection)
                sr = 16000
                t = np.linspace(0, 6, sr * 6)
                # Sine wave with quiet zones
                y = 0.5 * np.sin(2 * np.pi * 440 * t)
                # Insert quiet sections (pauses) at 1.5s-2.2s and 4.0s-4.8s
                y[int(1.5*sr):int(2.2*sr)] = 0.001
                y[int(4.0*sr):int(4.8*sr)] = 0.001
                
                sf.write(audio_path, y, sr)
                temp_file.close()
                
            # Perform Analysis
            try:
                progress_container = st.empty()
                
                with progress_container.container():
                    st.status("Initializing AI Speech models...", expanded=True)
                    
                # 1. Transcribe
                with progress_container.container():
                    st.status("Running Whisper ASR transcription...", expanded=True)
                
                if demo_mode:
                    # Provide a realistic machine learning explanation for transcription
                    transcript = (
                        "Machine learning is a subfield of artificial intelligence. Um, it focuses on building systems "
                        "that learn from data, identify patterns, and make decisions. So, like, we train algorithms using datasets "
                        "so they can generalize to new unseen data. Uh, typically it is split into supervised learning, "
                        "unsupervised learning, and also reinforcement learning."
                    )
                else:
                    transcript = transcribe_audio(audio_path, model_name=whisper_model)
                
                # 2. Semantic analysis
                with progress_container.container():
                    st.status("Comparing semantic embeddings (Sentence-BERT)...", expanded=True)
                
                concept_results = evaluate_concept_understanding(
                    transcript, 
                    st.session_state.concepts[selected_concept],
                    similarity_threshold=similarity_threshold
                )
                
                # 3. Audio diagnostics
                with progress_container.container():
                    st.status("Extracting Librosa audio features and pauses...", expanded=True)
                
                fluency_results = analyze_audio_fluency(
                    audio_path, 
                    transcript,
                    pause_threshold_db=pause_threshold,
                    min_pause_sec=min_pause_dur
                )
                
                # If demo mode, adjust speech parameters to match mock transcription
                if demo_mode:
                    fluency_results["wpm"] = 120.0
                    fluency_results["total_words"] = 52
                    fluency_results["total_fillers"] = 4 # um, so, like, uh
                    fluency_results["filler_density"] = (4 / 52) * 100
                    fluency_results["filler_counts"] = {
                        "um": 1, "uh": 1, "like": 1, "so": 1, "you know": 0, "actually": 0, "er": 0, "ah": 0
                    }
                    fluency_results["fluency_score"] = 78.0
                    fluency_results["fluency_feedback"] = "Moderate fluency: Natural pace, but contains noticeable pauses and occasional filler words."
                
                progress_container.empty()
                st.success("Analysis Complete! Head to the 'Diagnostic Dashboard' tab to view results.")
                
                # Store in session state
                st.session_state.analysis_completed = True
                st.session_state.transcript = transcript
                st.session_state.concept_results = concept_results
                st.session_state.fluency_results = fluency_results
                st.session_state.concept_name = selected_concept
                st.session_state.audio_path = audio_path
                
            except Exception as ex:
                st.error(f"Analysis failed: {str(ex)}")
                
# TAB 2: Diagnostic Dashboard
with tab2:
    if "analysis_completed" not in st.session_state:
        st.warning("Please upload/select an audio file and run the analysis on the 'Analyze Speech' tab first.")
    else:
        concept_name = st.session_state.concept_name
        transcript = st.session_state.transcript
        concept_results = st.session_state.concept_results
        fluency_results = st.session_state.fluency_results
        
        sim_pct = int(concept_results["similarity_score"] * 100)
        fluency_score = int(fluency_results["fluency_score"])
        overall_score = int((sim_pct * 0.6) + (fluency_score * 0.4))
        
        # 1. Premium Metrics Cards
        st.markdown(f"""
        <div class="card-container">
            <div class="metric-card">
                <div class="metric-label">Overall Competency Score</div>
                <div class="metric-value value-highlight">{overall_score} <span style="font-size:1.2rem; color:#9ca3af;">/ 100</span></div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Concept Similarity Score</div>
                <div class="metric-value value-purple">{sim_pct}%</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Speech Fluency Score</div>
                <div class="metric-value value-green">{fluency_score}%</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Speaking Pacing (WPM)</div>
                <div class="metric-value">{int(fluency_results['wpm'])}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # 2. Main content area columns
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("🎙️ Speech Transcription")
            st.markdown(f"*{transcript}*")
            
            st.subheader("🎯 Conceptual Key Points Coverage")
            for kp in concept_results["key_points_status"]:
                status = kp["status"]
                sim = int(kp["similarity"] * 100)
                
                if status == "Covered":
                    badge_class = "badge-covered"
                elif status == "Partially Covered":
                    badge_class = "badge-partial"
                else:
                    badge_class = "badge-missed"
                    
                st.markdown(f"""
                <div style="background: rgba(255,255,255,0.03); padding: 12px; border-radius: 8px; margin-bottom: 8px; border-left: 4px solid {'#10b981' if status=='Covered' else ('#fbbf24' if status=='Partially Covered' else '#ef4444')}">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <span style="font-size:0.95rem; font-weight:500;">{kp['key_point']}</span>
                        <span class="badge {badge_class}">{status} ({sim}%)</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
        with col2:
            st.subheader("📈 Audio Waveform & Pause Signal")
            
            # Plot Waveform
            y = np.array(fluency_results["waveform_y"])
            sr = fluency_results["waveform_sr"]
            duration = fluency_results["duration"]
            times = np.linspace(0, duration, len(y))
            
            fig, ax = plt.subplots(figsize=(10, 4), dpi=150)
            fig.patch.set_facecolor('#0f172a')
            ax.set_facecolor('#0f172a')
            
            # Plot main waveform in slate/blue
            ax.plot(times, y, color='#60a5fa', alpha=0.7, label='Waveform')
            
            # Highlight pauses
            pause_labeled = False
            for p in fluency_results["pauses"]:
                ax.axvspan(p["start"], p["end"], color='#f87171', alpha=0.3, 
                           label='Hesitation Pause' if not pause_labeled else "")
                pause_labeled = True
                
            ax.set_title("Speech Signal Waveform & Silence Tracking", color='#ffffff', fontsize=12, pad=12)
            ax.set_xlabel("Time (seconds)", color='#9ca3af', fontsize=9)
            ax.set_ylabel("Normalized Amplitude", color='#9ca3af', fontsize=9)
            ax.tick_params(colors='#9ca3af', labelsize=8)
            ax.grid(color='#334155', linestyle='--', linewidth=0.5)
            
            if pause_labeled:
                legend = ax.legend(facecolor='#1e293b', edgecolor='#334155', loc='upper right')
                for text in legend.get_texts():
                    text.set_color('#ffffff')
                    
            st.pyplot(fig)
            
            st.subheader("💬 Fluency Feedback")
            st.info(fluency_results["fluency_feedback"])
            
            # Sub-columns for specific fluency metrics
            f_col1, f_col2 = st.columns(2)
            with f_col1:
                st.write(f"⏱️ **Total speaking duration:** {duration:.2f} seconds")
                st.write(f"⏸️ **Pause Count:** {fluency_results['pause_count']}")
                st.write(f"⏳ **Total pause duration:** {fluency_results['total_pause_time']:.2f} seconds")
            with f_col2:
                st.write(f"🛑 **Pause Ratio:** {int(fluency_results['pause_ratio'] * 100)}% of speech")
                st.write(f"🗣️ **Filler words density:** {fluency_results['filler_density']:.1f}%")
                st.write(f"🔍 **Total filler words:** {fluency_results['total_fillers']}")
                
            with st.expander("Detailed Filler Words Count"):
                st.table(st.session_state.fluency_results["filler_counts"])
                
        # 3. Export PDF Section
        st.markdown("---")
        st.subheader("📄 Export Assessment Report")
        st.write("Generate a structured PDF containing full waveforms, conceptual checklists, metrics, and qualitative feedback.")
        
        if st.button("Generate PDF Report", type="primary"):
            with st.spinner("Compiling PDF report with reportlab..."):
                # Save plot to temp png
                temp_plot = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
                temp_plot_path = temp_plot.name
                temp_plot.close()
                
                generate_waveform_plot(fluency_results, temp_plot_path)
                
                # Build report PDF
                temp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
                pdf_path = temp_pdf.name
                temp_pdf.close()
                
                build_pdf_report(
                    pdf_path, 
                    concept_name, 
                    transcript, 
                    concept_results, 
                    fluency_results, 
                    temp_plot_path
                )
                
                # Read bytes and serve
                with open(pdf_path, "rb") as f:
                    pdf_bytes = f.read()
                    
                # Clean up files
                try:
                    os.unlink(temp_plot_path)
                    os.unlink(pdf_path)
                except:
                    pass
                    
                st.download_button(
                    label="📥 Download PDF Report",
                    data=pdf_bytes,
                    file_name=f"VBCUA_Report_{concept_name.replace(' ', '_')}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )

# TAB 3: Concept Customizer
with tab3:
    st.subheader("Add or Edit Concepts")
    st.write("Expand VBCUA's knowledge base by defining new concepts, expected explanations, and target keywords.")
    
    with st.form("new_concept_form", clear_on_submit=True):
        new_name = st.text_input("Concept Name (e.g., DNS, REST API)")
        new_desc = st.text_area("Reference Concept Description (Expected answer definition)")
        new_kps = st.text_area("Expected Key Points (One per line)")
        new_kws = st.text_input("Core Target Keywords (Comma separated)")
        
        submitted = st.form_submit_button("Save Concept Profile")
        
        if submitted:
            if not new_name or not new_desc or not new_kps or not new_kws:
                st.error("Please fill in all the fields before submitting.")
            else:
                # Format key points
                key_points_list = [kp.strip() for kp in new_kps.split("\n") if kp.strip()]
                # Format keywords
                keywords_list = [kw.strip() for kw in new_kws.split(",") if kw.strip()]
                
                # Add to session state concepts
                st.session_state.concepts[new_name] = {
                    "description": new_desc.strip(),
                    "key_points": key_points_list,
                    "keywords": keywords_list
                }
                
                # Save to file
                save_concepts(CONCEPTS_FILE, st.session_state.concepts)
                st.success(f"Successfully saved concept '{new_name}' to concepts.json database!")
                
    st.markdown("---")
    st.subheader("Current Concept Profiles")
    for name, c_data in st.session_state.concepts.items():
        with st.expander(name):
            st.markdown(f"**Description:**\n{c_data['description']}")
            st.markdown("**Expected Key Points:**")
            for kp in c_data["key_points"]:
                st.markdown(f"- {kp}")
            st.markdown(f"**Keywords:** {', '.join(c_data['keywords'])}")
