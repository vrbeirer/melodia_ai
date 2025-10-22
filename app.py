from flask import Flask, jsonify, send_file, request, send_from_directory, render_template
from flask_cors import CORS
import numpy as np
import random
import os
from midiutil import MIDIFile
import pickle
from io import BytesIO


# Deep Learning imports
try:
    from tensorflow import keras
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Activation
    from tensorflow.keras.utils import to_categorical
    DEEP_LEARNING_AVAILABLE = True
except ImportError:
    DEEP_LEARNING_AVAILABLE = False
    print("Warning: TensorFlow not available. Using mock generation.")

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


class MusicLSTM:
    """Deep Learning LSTM Model for Music Generation"""
    
    def __init__(self):
        self.model = None
        self.sequence_length = 50
        self.note_to_int = {}
        self.int_to_note = {}
        self.vocab_size = 0
        self.initialize_model()
    
    def create_training_data(self, scale_type="major", num_sequences=1000):
        """Generate synthetic training data based on musical scales"""
        notes = []
        root_notes = [60, 62, 64, 65, 67]
        
        for root in root_notes:
            scale = SCALES[scale_type]
            notes.extend([root + offset for offset in scale])
        
        notes = sorted(list(set(notes)))
        
        self.note_to_int = {note: i for i, note in enumerate(notes)}
        self.int_to_note = {i: note for i, note in enumerate(notes)}
        self.vocab_size = len(notes)
        
        sequences = []
        targets = []
        
        for _ in range(num_sequences):
            sequence = []
            current_note = random.choice(notes)
            
            for _ in range(self.sequence_length):
                sequence.append(self.note_to_int[current_note])
                
                available_notes = [n for n in notes if abs(n - current_note) <= 7]
                if available_notes:
                    weights = [max(1, 8 - abs(n - current_note)) for n in available_notes]
                    total = sum(weights)
                    probabilities = [w / total for w in weights]
                    current_note = np.random.choice(available_notes, p=probabilities)
                else:
                    current_note = random.choice(notes)
            
            next_note = random.choice([n for n in notes if abs(n - current_note) <= 5])
            sequences.append(sequence)
            targets.append(self.note_to_int[next_note])
        
        return np.array(sequences), np.array(targets)
    
    def build_model(self):
        """Build LSTM neural network architecture"""
        model = Sequential([
            LSTM(
                256,
                input_shape=(self.sequence_length, 1),
                return_sequences=True,
                activation='tanh'
            ),
            Dropout(0.3),
            LSTM(256, return_sequences=True, activation='tanh'),
            Dropout(0.3),
            LSTM(256, activation='tanh'),
            Dropout(0.3),
            Dense(256, activation='relu'),
            Dense(self.vocab_size, activation='softmax')
        ])
        
        model.compile(
            loss='sparse_categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
        )
        
        return model
    
    def initialize_model(self):
        """Initialize or load pre-trained model"""
        model_path = "lstm_music_model.h5"
        vocab_path = "vocab.pkl"
        
        if DEEP_LEARNING_AVAILABLE:
            if os.path.exists(model_path) and os.path.exists(vocab_path):
                try:
                    self.model = load_model(model_path)
                    with open(vocab_path, 'rb') as f:
                        vocab_data = pickle.load(f)
                        self.note_to_int = vocab_data['note_to_int']
                        self.int_to_note = vocab_data['int_to_note']
                        self.vocab_size = vocab_data['vocab_size']
                    print("✓ Loaded pre-trained LSTM model")
                except Exception as e:
                    print(f"Error loading model: {e}. Training new model...")
                    self.train_model()
            else:
                self.train_model()
    
    def train_model(self, epochs=50, batch_size=64):
        """Train the LSTM model on synthetic data"""
        print("🎵 Training LSTM model for music generation...")
        
        X_major, y_major = self.create_training_data("major", 500)
        X_minor, y_minor = self.create_training_data("minor", 500)
        
        X = np.concatenate([X_major, X_minor])
        y = np.concatenate([y_major, y_minor])
        
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        X = X / float(self.vocab_size)
        
        self.model = self.build_model()
        
        self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            verbose=0
        )
        
        self.model.save("lstm_music_model.h5")
        with open("vocab.pkl", 'wb') as f:
            pickle.dump({
                'note_to_int': self.note_to_int,
                'int_to_note': self.int_to_note,
                'vocab_size': self.vocab_size
            }, f)
        
        print("✓ Model trained and saved successfully!")
    
    def generate_notes(self, num_notes=64, temperature=1.0):
        """Generate musical notes using the trained LSTM model"""
        if self.model is None or not self.int_to_note:
            return None
        
        available_notes = list(self.int_to_note.keys())
        start_sequence = [random.choice(available_notes) for _ in range(self.sequence_length)]
        
        generated_notes = []
        pattern = start_sequence.copy()
        
        for _ in range(num_notes):
            x = np.reshape(pattern, (1, len(pattern), 1))
            x = x / float(self.vocab_size)
            
            prediction = self.model.predict(x, verbose=0)[0]
            
            prediction = np.log(prediction + 1e-10) / temperature
            exp_preds = np.exp(prediction)
            prediction = exp_preds / np.sum(exp_preds)
            
            index = np.random.choice(len(prediction), p=prediction)
            
            note = self.int_to_note[index]
            generated_notes.append(note)
            
            pattern.append(index)
            pattern = pattern[1:]
        
        return generated_notes


# Initialize LSTM model (global instance)
music_lstm = None
if DEEP_LEARNING_AVAILABLE:
    music_lstm = MusicLSTM()


def generate_melody_lstm(tempo=80, num_notes=64):
    """Generate melody using LSTM model"""
    if music_lstm and music_lstm.model:
        temperature = random.uniform(0.8, 1.2)
        notes = music_lstm.generate_notes(num_notes, temperature)
        
        midi = MIDIFile(1)
        track = 0
        channel = 0
        time = 0
        midi.addTempo(track, time, tempo)
        
        volume = 100
        
        for note in notes:
            duration = random.choice([0.5, 1.0, 1.5])
            midi.addNote(track, channel, note, time, duration, volume)
            time += duration
        
        return midi
    else:
        return None


def generate_melody_fallback(tempo=80, scale_type="major"):
    """Fallback melody generation without deep learning"""
    midi = MIDIFile(1)
    track = 0
    channel = 0
    time = 0
    midi.addTempo(track, time, tempo)
    
    root_note = random.choice([60, 62, 64, 65])
    scale_notes = [root_note + offset for offset in SCALES[scale_type]]
    
    num_notes = 48
    duration = 1.0
    volume = 100
    
    previous_note = random.choice(scale_notes)
    
    for i in range(num_notes):
        weights = []
        for note in scale_notes:
            distance = abs(note - previous_note)
            weight = max(1, 10 - distance)
            weights.append(weight)
        
        total_weight = sum(weights)
        probabilities = [w / total_weight for w in weights]
        current_note = np.random.choice(scale_notes, p=probabilities)
        
        if random.random() < 0.2:
            current_note += random.choice([-12, 12])
            current_note = max(48, min(84, current_note))
        
        note_duration = random.choice([0.5, 1.0, 1.5, 2.0])
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
    suggestions = random.sample(MOODS, 3)
    return jsonify({"suggestions": suggestions})


@app.route('/generate', methods=['POST'])
def generate():
    """Generate a MIDI melody using Deep Learning LSTM"""
    
    # Get parameters from request (with defaults)
    data = request.get_json() or {}
    tempo = data.get('tempo', random.choice([60, 80, 100, 120]))
    creativity = data.get('creativity', 1.0)
    length = data.get('length', 64)
    
    mood = random.choice(MOODS)
    scale_type = mood["scale"]
    
    # Try LSTM generation with custom parameters
    midi = generate_melody_lstm(tempo, length)
    
    # Fallback to algorithmic generation
    if midi is None:
        print("⚠ Using fallback generation method")
        midi = generate_melody_fallback(tempo, scale_type)
    
    # Generate filename
    filename = f"melody_{random.randint(1000, 9999)}.mid"
    
    # Save to BytesIO instead of disk
    from io import BytesIO
    midi_bytes = BytesIO()
    midi.writeFile(midi_bytes)
    midi_bytes.seek(0)
    
    insight = random.choice(INSIGHTS)
    recommendation = random.choice([m["mood"] for m in MOODS])
    
    # Return the file directly
    return send_file(
        midi_bytes,
        mimetype='audio/midi',
        as_attachment=True,
        download_name=filename
    )



@app.route('/static/<filename>')
def download_file(filename):
    """Serve generated MIDI files"""
    return send_file(os.path.join("static", filename), as_attachment=False)


@app.route('/retrain', methods=['POST'])
def retrain():
    """Endpoint to retrain the model with different parameters"""
    if not DEEP_LEARNING_AVAILABLE or music_lstm is None:
        return jsonify({"error": "Deep learning not available"}), 400
    
    data = request.get_json() or {}
    epochs = data.get('epochs', 50)
    
    music_lstm.train_model(epochs=epochs)
    
    return jsonify({"message": "Model retrained successfully", "epochs": epochs})


if __name__ == '__main__':
    print("=" * 60)
    print("🎵 AI Melody Studio - Deep Learning Backend")
    print("=" * 60)
    if DEEP_LEARNING_AVAILABLE:
        print("✓ TensorFlow/Keras available")
        print("✓ LSTM model initialized")
        print("✓ Deep learning music generation enabled")
    else:
        print("⚠ TensorFlow not found - using algorithmic generation")
        print("  Install with: pip install tensorflow")
    print("=" * 60)
    print("🌐 Server running")
    print("=" * 60)
    print("\nAvailable routes:")
    print(" • / → Main page")
    print(" • /suggest → Get mood suggestions")
    print(" • /generate → Generate melody")
    print("=" * 60)
    app.run(debug=False, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))


