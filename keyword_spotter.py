import os
import pathlib
import time
import sys

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import tensorflow_model_optimization as tfmot

import librosa
import pyaudio

# ------------------------------------
# 1. Ustawienia i stałe
# ------------------------------------

SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

TARGET_WORDS = ["yes", "no"]

SAMPLE_RATE = 16000
AUDIO_LENGTH_SECONDS = 1.0
N_SAMPLES = int(SAMPLE_RATE * AUDIO_LENGTH_SECONDS)

N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128

CONFIDENCE_THRESHOLD = 0.85
DEBOUNCE_TIME = 2.0  # sekundy

FORMAT = pyaudio.paInt16
CHANNELS = 1

# ------------------------------------
# 2. Akwizycja danych
# ------------------------------------


def load_data():
    data_dir = pathlib.Path("data/mini_speech_commands_extracted/mini_speech_commands")

    if not data_dir.exists():
        print("Pobieranie i rozpakowywanie danych...")
        try:
            tf.keras.utils.get_file(
                "mini_speech_commands.zip",
                origin="http://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip",
                extract=True,
                cache_dir=".",
                cache_subdir="data",
            )
        except Exception as e:
            print(f"Błąd podczas pobierania danych: {e}")
            sys.exit(1)

    target_paths = []
    for word in TARGET_WORDS:
        word_path = data_dir / word
        target_paths.extend(list(word_path.glob("*.wav")))

    if not target_paths:
        print(f"Nie znaleziono plików dla słów: {TARGET_WORDS}")
        sys.exit(1)

    print(f"Liczba plików dla słów '{', '.join(TARGET_WORDS)}': {len(target_paths)}")
    return target_paths


# ------------------------------------
# 3. Preprocessing audio (spektrogramy)
# ------------------------------------


def get_label(file_path: str) -> str:
    parts = str(file_path).split(os.path.sep)
    return parts[-2]


def preprocess_audio(file_path: str):
    try:
        audio, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    except Exception:
        return None, None

    if len(audio) > N_SAMPLES:
        audio = audio[:N_SAMPLES]
    elif len(audio) < N_SAMPLES:
        audio = np.pad(audio, (0, N_SAMPLES - len(audio)), "constant")

    S = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
    )

    S_dB = librosa.power_to_db(S, ref=np.max)
    spectrogram = np.expand_dims(S_dB, axis=-1)

    return spectrogram.astype("float32"), get_label(file_path)


def process_all_data(target_paths):
    X = []
    Y_labels = []

    print("Rozpoczynanie przetwarzania i etykietowania...")

    for file_path in target_paths:
        spectrogram, label = preprocess_audio(str(file_path))
        if spectrogram is not None:
            X.append(spectrogram)
            Y_labels.append(label)

    if not X:
        print("Błąd: nie udało się przetworzyć żadnych plików audio.")
        sys.exit(1)

    X = np.array(X, dtype="float32")

    label_to_id = {label: i for i, label in enumerate(TARGET_WORDS)}
    Y_ids = np.array([label_to_id[label] for label in Y_labels])
    Y = to_categorical(Y_ids, num_classes=len(TARGET_WORDS))

    print(f"Kształt danych wejściowych X: {X.shape}")
    print(f"Kształt etykiet Y: {Y.shape}")

    return X, Y, label_to_id


# ------------------------------------
# 4. Budowa i trening modelu
# ------------------------------------


def build_cnn_model(input_shape, num_classes):
    model = Sequential(
        [
            Input(shape=input_shape),
            Conv2D(32, (3, 3), activation="relu", padding="same"),
            MaxPooling2D((2, 2)),
            Dropout(0.25),
            Conv2D(64, (3, 3), activation="relu", padding="same"),
            MaxPooling2D((2, 2)),
            Dropout(0.25),
            Conv2D(128, (3, 3), activation="relu", padding="same"),
            MaxPooling2D((2, 2)),
            Dropout(0.25),
            Flatten(),
            Dense(256, activation="relu"),
            Dropout(0.5),
            Dense(num_classes, activation="softmax"),
        ]
    )

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_pruned_cnn_model(input_shape, num_classes):
    base_model = build_cnn_model(input_shape, num_classes)

    pruning_params = {
        "pruning_schedule": tfmot.sparsity.keras.PolynomialDecay(
            initial_sparsity=0.0,
            final_sparsity=0.5,
            begin_step=0,
            end_step=1000,
        )
    }

    pruned_model = tfmot.sparsity.keras.prune_low_magnitude(
        base_model,
        **pruning_params,
    )

    pruned_model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return pruned_model


# ------------------------------------
# 5. Trening i zapis modelu
# ------------------------------------


def train_and_save_model():
    print("--- ROZPOCZĘCIE ETAPU TRENINGOWEGO ---")

    target_paths = load_data()
    X, Y, label_to_id = process_all_data(target_paths)

    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std[std == 0] = 1.0
    X_normalized = (X - mean) / std

    X_train, X_temp, Y_train, Y_temp = train_test_split(
        X_normalized,
        Y,
        test_size=0.3,
        random_state=SEED,
        stratify=Y,
    )
    X_val, X_test, Y_val, Y_test = train_test_split(
        X_temp,
        Y_temp,
        test_size=0.5,
        random_state=SEED,
        stratify=Y_temp,
    )

    print(f"\nZbiór treningowy: {X_train.shape[0]} próbek")

    input_shape = X_train.shape[1:]
    num_classes = len(TARGET_WORDS)

    model = build_pruned_cnn_model(input_shape, num_classes)

    print("-" * 30)
    model.summary()
    print("-" * 30)

    pruning_callbacks = [
        tfmot.sparsity.keras.UpdatePruningStep(),
        tfmot.sparsity.keras.PruningSummaries(log_dir="./pruning_logs"),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5),
    ]

    model.fit(
        X_train,
        Y_train,
        validation_data=(X_val, Y_val),
        batch_size=32,
        epochs=20,
        callbacks=pruning_callbacks,
    )

    model = tfmot.sparsity.keras.strip_pruning(model)

    loss, acc = model.evaluate(X_test, Y_test, verbose=0)
    print(f"\nDokładność na zbiorze testowym: {acc * 100:.2f}%")

    model.save("kws_cnn_model.h5")
    np.savez("kws_normalization_params.npz", mean=mean, std=std)

    print(
        "Model i parametry normalizacyjne zostały zapisane: "
        "kws_cnn_model.h5 i kws_normalization_params.npz"
    )
    print("--- KONIEC ETAPU TRENINGOWEGO ---\n")

    return label_to_id, mean, std


# ------------------------------------
# 6. Aplikacja czuwająca (mikrofon)
# ------------------------------------


def preprocess_live_audio(audio_chunk, mean, std):
    audio_chunk = (
        np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32) / 32768.0
    )

    S = librosa.feature.melspectrogram(
        y=audio_chunk,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
    )
    S_dB = librosa.power_to_db(S, ref=np.max)

    spectrogram = np.expand_dims(S_dB, axis=-1)
    normalized_spectrogram = (spectrogram - mean) / std

    return np.expand_dims(normalized_spectrogram, axis=0).astype("float32")


last_detection_time = 0.0


def start_listening(loaded_model, mean, std, id_to_label):
    global last_detection_time

    p = pyaudio.PyAudio()
    frames_per_buffer = N_SAMPLES

    def stream_callback(in_data, frame_count, time_info, status):
        global last_detection_time

        expected_bytes = frames_per_buffer * 2
        if len(in_data) == expected_bytes:
            processed_spectrogram = preprocess_live_audio(in_data, mean, std)

            predictions = loaded_model.predict(processed_spectrogram, verbose=0)[0]
            max_prob = float(np.max(predictions))
            predicted_id = int(np.argmax(predictions))

            current_time = time.time()
            if max_prob > CONFIDENCE_THRESHOLD:
                if current_time - last_detection_time > DEBOUNCE_TIME:
                    predicted_label = id_to_label[predicted_id]
                    print(
                        f"\nWykryto słowo: {predicted_label} "
                        f"(ufność: {max_prob * 100:.1f}%)"
                    )
                    last_detection_time = current_time
                else:
                    print(".", end="", flush=True)
            else:
                print("_", end="", flush=True)

        return in_data, pyaudio.paContinue

    print("-" * 50)
    print("Start nasłuchiwania: mów 'yes' lub 'no' co najmniej co 2 sekundy...")
    print(f"Czułość (próg ufności): {CONFIDENCE_THRESHOLD * 100:.0f}%")
    print("-" * 50)

    try:
        stream = p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=frames_per_buffer,
            stream_callback=stream_callback,
        )

        stream.start_stream()

        while stream.is_active():
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\nZatrzymano nasłuchiwanie.")
    except Exception as e:
        print(f"\nBłąd w strumieniu audio: {e}")
    finally:
        if "stream" in locals() and stream.is_active():
            stream.stop_stream()
            stream.close()
        p.terminate()


# ------------------------------------
# 7. Główna funkcja programu
# ------------------------------------

if __name__ == "__main__":
    MODEL_PATH = "kws_cnn_model.h5"
    NORM_PARAMS_PATH = "kws_normalization_params.npz"

    if not os.path.exists(MODEL_PATH) or not os.path.exists(NORM_PARAMS_PATH):
        print("Brak wytrenowanego modelu. Rozpoczynanie treningu...")
        try:
            label_to_id, mean, std = train_and_save_model()
        except Exception as e:
            print(f"Błąd podczas treningu: {e}")
            sys.exit(1)
    else:
        print("Wykryto wytrenowany model. Ładowanie...")

        # spójne z kierunkiem mapowania używanym w preprocessingu
        label_to_id = {label: i for i, label in enumerate(TARGET_WORDS)}

        norm_params = np.load(NORM_PARAMS_PATH)
        mean = norm_params["mean"]
        std = norm_params["std"]

    try:
        loaded_model = tf.keras.models.load_model(MODEL_PATH)
    except Exception as e:
        print(f"Błąd podczas ładowania modelu: {e}")
        sys.exit(1)

    id_to_label = {v: k for k, v in label_to_id.items()}
    start_listening(loaded_model, mean, std, id_to_label)
