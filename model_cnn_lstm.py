import numpy as np
import tensorflow as tf
from tqdm import tqdm
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.applications import DenseNet201, VGG16, EfficientNetV2B0
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.layers import Input, Dense, Embedding, LSTM, AdditiveAttention, Add, Concatenate, Dropout, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# Build tokenizer before using the model
class CNN_LSTM:
    """
    CNN-LSTM Model for Image Captioning

    This class implements a CNN-LSTM model architecture for image captioning tasks. It provides methods for 
    feature extraction, model building (with or without attention mechanism), and tokenizer initialization.

    Attributes:
        cnn_type (str): The type of CNN backbone used for feature extraction ('densenet', 'efficientnetv2', or 'vgg16').
        vocab_size (int): The size of the vocabulary.
        max_length (int): The maximum length of captions.
        input_shape (tuple): The shape of the input features extracted by the CNN.
        feature_extractor (Model): Pretrained CNN model for feature extraction.
        tokenizer (Tokenizer): Tokenizer fitted to the captions for processing text.
        model (Model): The built CNN-LSTM model.
    """

    def __init__(self, cnn_type="densenet", vocab_size=None, max_length=None):
        """
        Initializes the CNN-LSTM model with the specified CNN backbone and parameters.

        Args:
            cnn_type (str): The type of CNN backbone ('densenet', 'efficientnetv2', or 'vgg16').
            vocab_size (int, optional): Size of the vocabulary. Default is None.
            max_length (int, optional): Maximum length of captions. Default is None.
        """
        self.feature_extractor = self._build_feature_extractor()
        self.cnn_type = cnn_type
        if self.cnn_type == "densenet":
            self.input_shape = (1920,)
        elif self.cnn_type == "efficientnetv2":
            self.input_shape = (1280,)
        elif self.cnn_type == "vgg16":
            self.input_shape = (4096,)

    def _build_feature_extractor(self):
        """
        Builds the CNN feature extractor based on the selected CNN backbone.

        Returns:
            Model: A pretrained CNN model for feature extraction.
        """
        if self.cnn_type == "densenet":
            base_model = DenseNet201()
        elif self.cnn_type == "efficientnetv2":
            base_model = EfficientNetV2B0()
        elif self.cnn_type == "vgg16":
            base_model = VGG16()
        return Model(inputs=base_model.input, outputs=base_model.layers[-2].output)

    def extract_image_features(self, image_paths):
        """
        Extracts image features using the CNN feature extractor.

        Args:
            image_paths (list of str): List of image file paths.

        Returns:
            dict: A dictionary mapping image paths to their corresponding feature vectors.
        """
        image_features = {}
        for image in tqdm(image_paths):
            img = load_img(image, target_size=(224, 224))
            img = img_to_array(img) / 255.
            img = np.expand_dims(img, axis=0)
            feature = self.feature_extractor.predict(img, verbose=0)
            image_features[image] = feature
        return image_features

    def _build_model(self, attention=True):
        """
        Builds the CNN-LSTM model with or without an attention mechanism.

        Args:
            attention (bool): Whether to include an attention mechanism in the model. Default is True.

        Returns:
            Model: The compiled Keras model.
        """
        if attention:
            # CNN Feature Input
            image_input = Input(shape=self.input_shape, name="image_input")
            cnn_features = Dense(256, activation='relu')(image_input)

            # Sentence Input and Embedding
            sentence_input = Input(shape=(self.max_length,), name="sentence_input")
            sentence_embedding = Embedding(self.vocab_size, 256, mask_zero=True, name="sentence_embedding")(sentence_input)

            # LSTM for Caption
            lstm_output = LSTM(256)(sentence_embedding)
            lstm_output = Dropout(0.5)(lstm_output)

            # Attention Mechanism
            attention_output = AdditiveAttention(dropout=0.2, use_scale=True)([lstm_output, cnn_features])
            attention_output = Dropout(0.5)(attention_output)

            # Combine Attention and LSTM Output
            combined = Add()([attention_output, lstm_output])
            combined = Concatenate(axis=-1)([combined, cnn_features])
            combined = Dense(128, activation='relu')(combined)
            combined = Dropout(0.5)(combined)

            # Vocabulary Prediction
            output = Dense(self.vocab_size, activation="softmax", name="output_layer")(combined)

            # Build Model
            model = Model(inputs=[image_input, sentence_input], outputs=output)

        else:
            # CNN Feature Input
            image_input = Input(shape=self.input_shape, name="image_input")
            img_features = Dense(256, activation='relu')(image_input)
            img_features_reshaped = Reshape((1, 256), input_shape=(256,))(img_features)

            # Sentence Input and Embedding
            sentence_input = Input(shape=(self.max_length,), name="sentence_input")
            sentence_features = Embedding(self.vocab_size, 256, mask_zero=False, name="sentence_embedding")(sentence_input)

            # Merge Features
            merged = Concatenate(axis=1)([img_features_reshaped, sentence_features])
            sentence_features = LSTM(256)(merged)
            x = Dropout(0.5)(sentence_features)
            x = Add()([x, img_features])
            x = Dense(128, activation='relu')(x)
            x = Dropout(0.5)(x)

            # Vocabulary Prediction
            output = Dense(self.vocab_size, activation='softmax', name="output_layer")(x)

            # Build Model
            model = Model(inputs=[image_input, sentence_input], outputs=output)

        optimizer = Adam()
        model.compile(loss='categorical_crossentropy', optimizer=optimizer)
        self.model = model
        return model

    def initialize_tokenizer(self, captions):
        """
        Initializes and fits a tokenizer to the provided captions. Also updates `vocab_size` and `max_length`.

        Args:
            captions (list of str): List of text captions for training.

        Returns:
            Tokenizer: The fitted tokenizer.
        """
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(captions)
        self.vocab_size = len(tokenizer.word_index) + 1
        self.max_length = max(len(caption.split()) for caption in captions)
        self.tokenizer = tokenizer
        self.model = self._build_model()
        return tokenizer
    