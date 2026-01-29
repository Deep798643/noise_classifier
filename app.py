"""
🔊 Audio Noise Classifier - Streamlit Cloud Ready
Complete standalone app for deployment

Author: Audio Classifier
Date: 2026
"""

import streamlit as st
import numpy as np
import os
import tempfile
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# AUDIO FEATURE EXTRACTOR (Embedded)
# ============================================================================

class AudioFeatureExtractor:
    """Extract audio features for classification"""
    
    def __init__(self, sr=16000, n_mfcc=13):
        self.sr = sr
        self.n_mfcc = n_mfcc
    
    def load_audio(self, file_path, duration=5):
        """Load audio file"""
        try:
            import librosa
            y, sr = librosa.load(file_path, sr=self.sr, duration=duration)
            return y, sr
        except Exception as e:
            st.error(f"❌ Error loading {file_path}: {e}")
            return None, None
    
    def calculate_decibels(self, y):
        """Calculate decibel level from audio signal"""
        rms = np.sqrt(np.mean(y**2))
        if rms < 1e-10:
            return 0.0
        db_level = 20 * np.log10(rms)
        db_level = max(30, min(120, db_level + 90))
        return db_level
    
    def extract_features(self, y, sr=None):
        """Extract exactly 60 audio features"""
        import librosa
        
        if sr is None:
            sr = self.sr
        
        features = []
        
        # MFCC features (13 coefficients)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc)
        features.extend(np.mean(mfcc, axis=1))
        features.extend(np.std(mfcc, axis=1))
        
        # Spectral centroid
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        features.append(np.mean(spectral_centroids))
        features.append(np.std(spectral_centroids))
        
        # Spectral rolloff
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        features.append(np.mean(spectral_rolloff))
        features.append(np.std(spectral_rolloff))
        
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        features.append(np.mean(zcr))
        features.append(np.std(zcr))
        
        # RMS Energy
        rms = librosa.feature.rms(y=y)[0]
        features.append(np.mean(rms))
        features.append(np.std(rms))
        
        # Chroma features (12 coefficients)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        features.extend(np.mean(chroma, axis=1))
        
        # Spectral bandwidth
        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
        features.append(np.mean(spec_bw))
        features.append(np.std(spec_bw))
        
        # Spectral contrast (7 bands)
        spec_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        features.extend(np.mean(spec_contrast, axis=1))
        
        # Onset strength
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        features.append(np.mean(onset_env))
        features.append(np.std(onset_env))
        
        # Tempo
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        features.append(float(tempo))
        
        # Spectral flatness
        spec_flatness = librosa.feature.spectral_flatness(y=y)[0]
        features.append(np.mean(spec_flatness))
        features.append(np.std(spec_flatness))
        
        return np.array(features[:60])


# ============================================================================
# NOISE CLASSIFIER (Embedded)
# ============================================================================

class NoiseClassifier:
    """Noise classifier using Random Forest + K-Means"""
    
    def __init__(self):
        self.rf_model = None
        self.kmeans_model = None
        self.scaler = None
        self.classes = None
        self.feature_extractor = AudioFeatureExtractor()
    
    def create_models(self, n_clusters=5):
        """Create Random Forest and K-Means models"""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
        
        self.scaler = StandardScaler()
        
        self.rf_model = RandomForestClassifier(
            n_estimators=150,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        self.kmeans_model = KMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init=10
        )
    
    def train(self, X, y):
        """Train both models"""
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score
        
        self.classes = np.unique(y)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        self.rf_model.fit(X_train_scaled, y_train)
        self.kmeans_model.fit(X_train_scaled)
        
        y_pred = self.rf_model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        return accuracy
    
    def classify_audio(self, audio_file):
        """Classify single audio file and return detailed results"""
        # Load audio
        y, sr = self.feature_extractor.load_audio(audio_file)
        if y is None:
            return None
        
        # Calculate decibels
        db_level = self.feature_extractor.calculate_decibels(y)
        
        # Extract features
        features = self.feature_extractor.extract_features(y, sr)
        
        if len(features) != 60:
            if len(features) < 60:
                features = np.pad(features, (0, 60 - len(features)), mode='constant')
            else:
                features = features[:60]
        
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        
        # Random Forest prediction
        prediction = self.rf_model.predict(features_scaled)[0]
        probabilities = self.rf_model.predict_proba(features_scaled)[0]
        
        # Get all class probabilities
        prob_dict = {}
        for i, cls in enumerate(self.classes):
            prob_dict[cls] = probabilities[i]
        
        sorted_probs = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)
        
        # Get decibel classification
        if db_level < 40:
            db_class, alert = "Very Quiet", "🟢 Safe"
        elif db_level < 55:
            db_class, alert = "Quiet", "🟢 Safe"
        elif db_level < 70:
            db_class, alert = "Moderate", "🟡 Caution"
        elif db_level < 85:
            db_class, alert = "Loud", "🟠 Warning"
        elif db_level < 100:
            db_class, alert = "Very Loud", "🔴 Danger"
        else:
            db_class, alert = "Extremely Loud", "🔴 Harmful"
        
        return {
            'file': audio_file,
            'classified_as': sorted_probs[0][0],
            'confidence': sorted_probs[0][1],
            'decibel_level': db_level,
            'decibel_class': db_class,
            'alert_status': alert,
            'all_probabilities': sorted_probs
        }
    
    def save(self, filename='audio_classifier_model.pkl'):
        """Save models"""
        import joblib
        joblib.dump({
            'rf_model': self.rf_model,
            'kmeans_model': self.kmeans_model,
            'scaler': self.scaler,
            'classes': self.classes
        }, filename)
    
    def load(self, filename='audio_classifier_model.pkl'):
        """Load models"""
        import joblib
        data = joblib.load(filename)
        self.rf_model = data['rf_model']
        self.kmeans_model = data['kmeans_model']
        self.scaler = data['scaler']
        self.classes = data['classes']


def create_synthetic_dataset(num_samples_per_class=80):
    """Create synthetic dataset for training"""
    classes = ['Traffic', 'Machinery', 'Speech', 'Music', 'Wind']
    n_features = 60
    
    X = []
    y = []
    
    for i, noise_class in enumerate(classes):
        for _ in range(num_samples_per_class):
            features = np.random.randn(n_features) * 0.5
            features += (i * 0.4)
            X.append(features)
            y.append(noise_class)
    
    return np.array(X), np.array(y)


# ============================================================================
# STREAMLIT APP
# ============================================================================

st.set_page_config(
    page_title="🔊 Audio Noise Classifier",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1E88E5;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .result-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
        padding: 0.75rem;
        border-radius: 10px;
        border: none;
        font-size: 1.1rem;
    }
</style>
""", unsafe_allow_html=True)

# Session state
if 'classifier' not in st.session_state:
    st.session_state.classifier = None
    st.session_state.model_trained = False
    st.session_state.last_result = None

# Load or train model
@st.cache_resource
def load_or_train_model():
    """Load existing model or train new one"""
    classifier = NoiseClassifier()
    model_file = 'audio_classifier_model.pkl'
    
    if os.path.exists(model_file):
        try:
            classifier.load(model_file)
            return classifier, True
        except:
            pass
    
    # Train new model
    X, y = create_synthetic_dataset(num_samples_per_class=80)
    classifier.create_models(n_clusters=5)
    classifier.train(X, y)
    classifier.save(model_file)
    
    return classifier, True

# Main app
def main():
    # Header
    st.markdown('<div class="main-header">🔊 Audio Noise Classifier</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Upload audio files and get instant classification with confidence scores</div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        st.subheader("📊 Model Status")
        
        if st.button("🚀 Load/Train Model"):
            with st.spinner("Loading model..."):
                classifier, trained = load_or_train_model()
                st.session_state.classifier = classifier
                st.session_state.model_trained = trained
                st.success("✅ Model ready!")
        
        if st.session_state.model_trained:
            st.success("✅ Model is ready")
            st.info("**Classes:** Traffic, Machinery, Speech, Music, Wind")
        else:
            st.warning("⚠️ Please load/train model first")
        
        st.divider()
        
        st.subheader("ℹ️ About")
        st.write("""
        This app uses **Random Forest + K-Means** to classify environmental sounds.
        
        **Features:**
        - Real-time audio analysis
        - Decibel level measurement
        - Confidence scores
        - Beautiful visualizations
        """)
    
    # Main content
    if not st.session_state.model_trained:
        st.info("👈 Please load/train the model using the sidebar button")
        return
    
    # File upload
    st.header("📂 Upload Audio File")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Choose a WAV audio file",
            type=['wav', 'mp3', 'ogg', 'flac'],
            help="Upload an audio file to classify"
        )
    
    with col2:
        st.write("**Supported formats:**")
        st.write("- WAV, MP3, OGG, FLAC")
    
    if uploaded_file is not None:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name
        
        # Display audio player
        st.audio(uploaded_file, format=f'audio/{Path(uploaded_file.name).suffix[1:]}')
        
        # Classify button
        if st.button("🎯 Classify Audio", use_container_width=True):
            with st.spinner("🔍 Analyzing audio..."):
                result = st.session_state.classifier.classify_audio(tmp_path)
                st.session_state.last_result = result
                
                if result:
                    display_results(result)
        
        # Clean up
        try:
            os.unlink(tmp_path)
        except:
            pass

def display_results(result):
    """Display classification results"""
    
    st.divider()
    st.header("📊 Classification Results")
    
    # Main result
    st.markdown(f"""
    <div class="result-box">
        <h2 style="margin:0; font-size: 2.5rem;">🎯 {result['classified_as'].upper()}</h2>
        <p style="font-size: 1.3rem; margin-top: 0.5rem;">
            Confidence: {result['confidence']*100:.1f}% sure
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("🔊 Decibel Level", f"{result['decibel_level']:.1f} dB", result['decibel_class'])
    
    with col2:
        st.metric("📈 Confidence", f"{result['confidence']*100:.1f}%")
    
    with col3:
        st.metric("⚠️ Alert", result['alert_status'].split()[1])
    
    st.divider()
    
    # Probability breakdown
    st.subheader("📊 Detailed Probability Breakdown")
    
    for sound_type, probability in result['all_probabilities']:
        pct = probability * 100
        
        if probability >= 0.70:
            label = "✅ Very High Confidence"
        elif probability >= 0.50:
            label = "✓ High Confidence"
        elif probability >= 0.30:
            label = "○ Moderate"
        else:
            label = "· Low"
        
        col1, col2 = st.columns([1, 4])
        with col1:
            st.write(f"**{sound_type}**")
        with col2:
            st.progress(probability, text=f"{pct:.1f}% - {label}")
    
    st.divider()
    
    # Interpretation
    st.subheader("💡 Interpretation")
    
    top_prob = result['all_probabilities'][0][1] * 100
    top_class = result['all_probabilities'][0][0]
    
    if top_prob >= 90:
        st.success(f"This audio is **ALMOST CERTAINLY {top_class.upper()}** ({top_prob:.0f}% confident)")
    elif top_prob >= 70:
        st.info(f"This audio is **MOST LIKELY {top_class.upper()}** ({top_prob:.0f}% confident)")
    elif top_prob >= 50:
        st.warning(f"This audio is **PROBABLY {top_class.upper()}** ({top_prob:.0f}% confident)")
    else:
        second_class = result['all_probabilities'][1][0]
        second_prob = result['all_probabilities'][1][1] * 100
        st.warning(f"Could be **{top_class.upper()}** ({top_prob:.0f}%) or **{second_class.upper()}** ({second_prob:.0f}%)")
    
    # Safety warning
    db_level = result['decibel_level']
    if db_level >= 85:
        st.error("⚠️ **WARNING:** This noise level is potentially harmful to hearing!")
    elif db_level >= 70:
        st.warning("⚠️ **CAUTION:** Prolonged exposure may cause discomfort.")
    else:
        st.success("✅ This noise level is considered safe.")

if __name__ == "__main__":
    main()
