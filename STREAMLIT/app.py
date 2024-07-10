import streamlit as st
import numpy as np
import librosa
from tensorflow.keras.models import load_model  # Correct import statement


# Function to extract MFCC from audio file
def extract_mfcc(audio_path, num_mfcc=13, max_pad_len=216):
    y, sr = librosa.load(audio_path)  # Ensure the path is correct
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=num_mfcc)

    # Pad or truncate MFCCs to a fixed length
    if mfccs.shape[1] > max_pad_len:
        mfccs = mfccs[:, :max_pad_len]
    else:
        pad_width = max_pad_len - mfccs.shape[1]
        mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')

    return mfccs


# Function to predict genre from audio file
def predict_genre(model, audio_file, num_mfcc=13, max_pad_len=216):
    try:
        audio_path = f"./user_audio/{audio_file.name}"
        with open(audio_path, 'wb') as f:
            f.write(audio_file.getbuffer())

        mfcc = extract_mfcc(audio_path, num_mfcc, max_pad_len)
        mfcc = mfcc[np.newaxis, ..., np.newaxis]  # Add batch and channel dimensions

        # Check model input shape
        input_shape = model.input_shape

        prediction = model.predict(mfcc)
        predicted_genre = np.argmax(prediction)
        return predicted_genre
    except Exception as e:
        st.error(f"Error predicting genre: {e}")
        return None


# Load model
def load_custom_model(model_path):
    model = load_model(model_path)
    return model


# Main function to run the Streamlit app
def main():
    st.title('Music Genre Classification')
    st.sidebar.title('Genre Explanation')

    # Sidebar with genre explanations
    st.sidebar.markdown("""
    This app classifies the genre of music based on audio files in WAV format. It uses a Convolutional Neural Network (CNN) trained on MFCC features.

    ### Genres
    - **Blues** 🎸: Evokes emotion, typically with guitar and harmonica.
    - **Classical** 🎻: Orchestral compositions known for complexity and elegance.
    - **Country** 🤠: Songs about rural life, often with guitar, banjo, and storytelling.
    - **Disco** 🕺: Upbeat dance music with a distinctive bassline and rhythm.
    - **Hip-hop** 🎤: Urban music with rap vocals and sampled beats.
    - **Jazz** 🎷: Improvised melodies with swing and blues influences.
    - **Metal** 🎸: Heavy guitar riffs, aggressive vocals, and amplified sound.
    - **Pop** 🎤: Popular music characterized by catchy melodies and hooks.
    - **Reggae** 🎶: Jamaican music known for its rhythmic accents and social commentary.
    - **Rock** 🎸: Guitar-driven music with strong beats and energetic performances.
    """)

    # File uploader
    uploaded_file = st.file_uploader("Choose a WAV file", type="wav")

    if uploaded_file is not None:
        st.audio(uploaded_file, format='audio/wav')

        # Handle prediction
        if st.button('Predict Genre'):
            model_path = 'music_genre_classification_model.h5'
            model = load_custom_model(model_path)
            if model:
                predicted_genre = predict_genre(model, uploaded_file)
                if predicted_genre is not None:
                    genres = ['Blues 🎸', 'Classical 🎻', 'Country 🤠', 'Disco 🕺', 'Hip-hop 🎤',
                              'Jazz 🎷', 'Metal 🎸', 'Pop 🎤', 'Reggae 🎶', 'Rock 🎸']
                    st.write(f"Predicted Genre: {genres[predicted_genre]}")


if __name__ == '__main__':
    main()
