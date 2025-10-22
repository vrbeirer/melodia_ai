from flask import Flask, jsonify, send_file, request, send_from_directory, render_template
from flask_cors import CORS
import numpy as np
import random
import os
from midiutil import MIDIFile
import pickle
from io import BytesIO


# Deep Learning imports - DISABLED FOR RENDER FREE TIER
DEEP_LEARNING_AVAILABLE = False
print("⚠️ TensorFlow disabled to save memory on free tier")

app = Flask(__name__)
CORS(app)

# AI Mood suggestions database
MOODS = [
    {"mood": "Calm Piano", "emoji": "🎹", "tempo": 80, "scale": "major"},
    {"mood": "Dreamy Lo-Fi", "emoji": "🌙", "tempo": 70, "scale": "minor"},
    {"mood": "Morning Acoustic", "emoji": "☀️", "tempo": 95, "scale": "major"},
    {"mood": "Ambient Chill", "emoji": "🌊", "tempo": 60, "scale": "minor"},
    {"mood": "Uplifting Pop", "emoji": "✨", "tempo": 120, "scale": "major"},
    {"mood": "Melancholic Rain", "emoji": "🌧️", "tempo": 65, "scale": "minor"},
    {"mood": "Epic Cinematic", "emoji": "🎬", "tempo": 110, "scale": "minor"},
    {"mood": "Jazz Lounge", "emoji": "🎷", "tempo": 90, "scale": "major"},
    {"mood": "Mystical Forest", "emoji": "🌲", "tempo": 75, "scale": "minor"},
]

# AI Insights for generated melodies
INSIGHTS = [
    "A gentle melody with flowing progressions and serene atmosphere",
    "Emotional depth with dynamic range and expressive phrasing",
    "Minimalist composition with elegant repetition patterns",
    "Complex harmonic structure with contemporary influences",
    "Uplifting progression that builds emotional resonance",
    "Contemplative tones with subtle melodic variations",
    "Rhythmic patterns that create engaging movement",
    "Balanced composition with classical undertones",
]

# Musical scales (MIDI note offsets from root)
SCALES = {
    "major": [0, 2, 4, 5, 7, 9, 11, 12],
    "minor": [0, 2, 3, 5, 7, 8, 10, 12],
}


def generate_melody_fallback(tempo=80, scale_type="major", num_notes=48):
    """Lightweight algorithmic melody generation"""
    midi = MIDIFile(1)
    track = 0
    channel = 0
    time = 0
    midi.addTempo(track, time, tempo)
    
    # Choose root note
    root_note = random.choice([60, 62, 64, 65])
    scale_notes = [root_note + offset for offset in SCALES[scale_type]]
    
    volume = 100
    previous_note = random.choice(scale_notes)
    
    for i in range(num_notes):
        # Weighted selection based on distance from previous note
        weights = []
        for note in scale_notes:
            distance = abs(note - previous_note)
            weight = max(1, 10 - distance)
            weights.append(weight)
        
        total_weight = sum(weights)
        probabilities = [w / total_weight for w in weights]
        current_note = np.random.choice(scale_notes, p=probabilities)
        
        # Occasionally add octave jumps
        if random.random() < 0.15:
            current_note += random.choice([-12, 12])
            current_note = max(48, min(84, current_note))
        
        # Varied note durations
        note_duration = random.choice([0.5, 0.75, 1.0, 1.5, 2.0])
        midi.addNote(track, channel, current_note, time, note_duration, volume)
        
        time += note_duration
        previous_note = current_note
    
    return midi


# ============= ROUTES =============

@app.route('/')
def index():
    """Serve the main index.html page"""
    return render_template('index.html')


@app.route('/suggest', methods=['GET'])
def suggest():
    """Return 3 random AI mood suggestions"""
    try:
        suggestions = random.sample(MOODS, 3)
        return jsonify({"suggestions": suggestions})
    except Exception as e:
        print(f"Error in suggest: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/generate', methods=['POST'])
def generate():
    """Generate a MIDI melody using algorithmic generation"""
    
    try:
        # Get parameters from request (with defaults)
        data = request.get_json() or {}
        tempo = data.get('tempo', random.choice([70, 80, 90, 100, 110, 120]))
        creativity = data.get('creativity', 1.0)
        length = data.get('length', 48)
        
        # Limit length for performance
        length = min(max(length, 24), 64)
        
        # Choose scale based on mood or random
        mood = random.choice(MOODS)
        scale_type = mood["scale"]
        
        print(f"⚙️ Generating melody: tempo={tempo}, length={length}, scale={scale_type}")
        
        # Use lightweight algorithmic generation
        midi = generate_melody_fallback(tempo, scale_type, length)
        
        # Generate filename
        filename = f"melody_{random.randint(1000, 9999)}.mid"
        
        # Save to BytesIO
        midi_bytes = BytesIO()
        midi.writeFile(midi_bytes)
        midi_bytes.seek(0)
        
        print(f"✓ Generated {filename}")
        
        # Return the file directly
        return send_file(
            midi_bytes,
            mimetype='audio/midi',
            as_attachment=True,
            download_name=filename
        )
    
    except Exception as e:
        print(f"❌ Error in generate: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/static/<filename>')
def download_file(filename):
    """Serve static MIDI files"""
    try:
        return send_file(os.path.join("static", filename), as_attachment=False)
    except Exception as e:
        print(f"Error serving file: {e}")
        return jsonify({"error": "File not found"}), 404


@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "mode": "algorithmic"})


if __name__ == '__main__':
    print("=" * 60)
    print("🎵 Melodia AI - Lightweight Music Generator")
    print("=" * 60)
    print("⚙️  Using algorithmic generation (optimized for Render)")
    print("💾 Memory footprint: <50MB")
    print("⚡ Generation speed: Instant")
    print("=" * 60)
    print("🌐 Server starting...")
    print("=" * 60)
    print("\nAvailable routes:")
    print(" • GET  /           → Main page")
    print(" • GET  /suggest    → AI mood suggestions")
    print(" • POST /generate   → Generate melody")
    print(" • GET  /health     → Health check")
    print("=" * 60)
    
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
