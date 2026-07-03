import os
import tempfile
import numpy as np
import soundfile as sf

from analyzer import (
    load_concepts,
    evaluate_concept_understanding,
    analyze_audio_fluency
)
from pdf_report import generate_waveform_plot, build_pdf_report

def main():
    print("=== Beginning VBCUA Verification Tests ===")
    
    # 1. Load Concept database
    concepts_path = os.path.join(os.path.dirname(__file__), "concepts.json")
    print(f"Loading concepts from database: {os.path.basename(concepts_path)}")
    concepts = load_concepts(concepts_path)
    assert len(concepts) > 0, "No concepts loaded!"
    print(f"Successfully loaded {len(concepts)} concepts: {list(concepts.keys())}")
    
    # 2. Setup mock audio file
    # We will generate a synthetic 5-second sine wave with silence gaps at 1.5s - 2.5s
    print("Generating synthetic WAV file...")
    temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    temp_wav_path = temp_wav.name
    temp_wav.close()
    
    sr = 16000
    duration = 5.0
    t = np.linspace(0, duration, int(sr * duration))
    # Sine wave
    y = 0.5 * np.sin(2 * np.pi * 440 * t)
    # Inject silence segment between 1.5 and 2.5 seconds
    y[int(1.5 * sr):int(2.5 * sr)] = 0.001
    
    sf.write(temp_wav_path, y, sr)
    print(f"Wav file saved to temp file: {os.path.basename(temp_wav_path)}")
    
    # 3. Test Audio analysis
    print("Running audio fluency analysis...")
    mock_transcript = (
        "Machine learning is a subset of artificial intelligence. Um, it focuses on building systems "
        "that learn from data and identify patterns. So, like, we train algorithms using datasets. "
        "Uh, it includes supervised, unsupervised, and reinforcement learning."
    )
    
    fluency_results = analyze_audio_fluency(
        temp_wav_path,
        mock_transcript,
        pause_threshold_db=-35,
        min_pause_sec=0.4
    )
    
    print(f"Fluency score: {fluency_results['fluency_score']:.1f}")
    print(f"Pause count: {fluency_results['pause_count']}")
    print(f"Pause ratio: {fluency_results['pause_ratio']:.2%}")
    print(f"Speaking rate: {fluency_results['wpm']:.1f} WPM")
    print(f"Total filler words found: {fluency_results['total_fillers']}")
    
    # 4. Test Concept evaluation
    print("Running semantic concept evaluation...")
    concept_data = concepts["Machine Learning"]
    concept_results = evaluate_concept_understanding(
        mock_transcript,
        concept_data,
        similarity_threshold=0.45
    )
    
    print(f"Overall Similarity: {concept_results['similarity_score']:.2%}")
    print(f"Understanding Level: {concept_results['understanding_level']}")
    print("\nKey Points Coverage Breakdown:")
    for kp in concept_results["key_points_status"]:
        print(f" - [{kp['status']}] {kp['key_point']} (Match: {kp['similarity']:.2f})")
        
    # 5. Test PDF Generation
    print("\nVerifying PDF report builder...")
    # Generate waveform image first
    temp_plot = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    temp_plot_path = temp_plot.name
    temp_plot.close()
    
    generate_waveform_plot(fluency_results, temp_plot_path)
    print(f"Temporary plot generated at temp path: {os.path.basename(temp_plot_path)}")
    
    # Generate PDF report
    temp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    temp_pdf_path = temp_pdf.name
    temp_pdf.close()
    
    build_pdf_report(
        temp_pdf_path,
        "Machine Learning",
        mock_transcript,
        concept_results,
        fluency_results,
        temp_plot_path
    )
    print(f"Temporary PDF compiled successfully at temp path: {os.path.basename(temp_pdf_path)}")
    
    # Clean up temp files
    os.unlink(temp_wav_path)
    os.unlink(temp_plot_path)
    os.unlink(temp_pdf_path)
    print("Cleaned up all temporary verification files.")
    print("=== All Verification Tests Passed Successfully! ===")

if __name__ == "__main__":
    main()
