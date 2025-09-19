"""
Complete MTG Card Recognition System - AI Model Approach
This file contains everything needed to recognize Magic: The Gathering cards from images using AI.
"""

import cv2
import numpy as np
import requests
import json
import os
import time
from PIL import Image
import re
from datetime import datetime

# Import TensorFlow/Keras for AI model
try:
    import tensorflow as tf
    from tensorflow import keras
    from sklearn.model_selection import train_test_split
    
    # Create aliases for compatibility with TensorFlow 2.20.0
    applications = tf.keras.applications
    models = tf.keras.models
    TF_AVAILABLE = True
except ImportError:
    print("Error: TensorFlow is required for the AI model approach.")
    print("Please install it with: pip install tensorflow")
    exit(1)

class MTGDataPreparer:
    def __init__(self, data_dir="mtg_dataset"):
        self.data_dir = data_dir
        self.images_dir = os.path.join(data_dir, "images")
        self.labels_dir = os.path.join(data_dir, "labels")
        self.metadata_file = os.path.join(data_dir, "metadata.json")
        
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.labels_dir, exist_ok=True)
        
    def download_card_images(self, limit=500):
        """Download card images from Scryfall API"""
        print("Downloading card images from Scryfall...")
        
        # Get all cards
        response = requests.get("https://api.scryfall.com/cards/search?q=game:paper")
        cards_data = response.json()
        
        cards = cards_data['data']
        while cards_data.get('has_more') and len(cards) < limit:
            next_page = requests.get(cards_data['next_page'])
            cards_data = next_page.json()
            cards.extend(cards_data['data'])
        
        metadata = {}
        downloaded = 0
        
        for card in cards[:limit]:
            try:
                card_id = card['id']
                card_name = card['name']
                
                # Skip digital-only cards
                if not card.get('image_uris'):
                    continue
                
                # Download image
                image_url = card['image_uris']['normal']
                image_response = requests.get(image_url)
                
                if image_response.status_code == 200:
                    # Save image
                    image_path = os.path.join(self.images_dir, f"{card_id}.jpg")
                    with open(image_path, 'wb') as f:
                        f.write(image_response.content)
                    
                    # Store metadata
                    metadata[card_id] = {
                        'name': card_name,
                        'set': card.get('set_name', ''),
                        'type': card.get('type_line', ''),
                        'mana_cost': card.get('mana_cost', ''),
                        'colors': card.get('colors', []),
                    }
                    
                    downloaded += 1
                    print(f"Downloaded {downloaded}/{limit}: {card_name}")
                    
                    # Be respectful to the API
                    time.sleep(0.1)
                    
            except Exception as e:
                print(f"Error downloading {card.get('name', 'unknown')}: {e}")
        
        # Save metadata
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Downloaded {downloaded} card images")
        return metadata
    
    def create_augmented_dataset(self, augmentations_per_image=5):
        """Create augmented versions of card images"""
        print("Creating augmented dataset...")
        
        augmented_dir = os.path.join(self.data_dir, "augmented")
        os.makedirs(augmented_dir, exist_ok=True)
        
        image_files = [f for f in os.listdir(self.images_dir) if f.endswith('.jpg')]
        
        for img_file in image_files:
            img_path = os.path.join(self.images_dir, img_file)
            image = cv2.imread(img_path)
            
            if image is None:
                continue
                
            card_id = os.path.splitext(img_file)[0]
            
            # Save original
            aug_path = os.path.join(augmented_dir, f"{card_id}_0.jpg")
            cv2.imwrite(aug_path, image)
            
            # Create augmented versions
            for i in range(augmentations_per_image):
                augmented = self._augment_image(image)
                aug_path = os.path.join(augmented_dir, f"{card_id}_{i+1}.jpg")
                cv2.imwrite(aug_path, augmented)
        
        print("Augmentation complete")
    
    def _augment_image(self, image):
        """Apply random augmentations to an image"""
        # Random rotation (-15 to 15 degrees)
        angle = np.random.uniform(-15, 15)
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, rotation_matrix, (width, height))
        
        # Random perspective transformation
        if np.random.random() > 0.5:
            pts1 = np.float32([[0,0], [width,0], [0,height], [width,height]])
            pts2 = np.float32([
                [np.random.randint(0, width*0.1), np.random.randint(0, height*0.1)],
                [np.random.randint(width*0.9, width), np.random.randint(0, height*0.1)],
                [np.random.randint(0, width*0.1), np.random.randint(height*0.9, height)],
                [np.random.randint(width*0.9, width), np.random.randint(height*0.9, height)]
            ])
            matrix = cv2.getPerspectiveTransform(pts1, pts2)
            rotated = cv2.warpPerspective(rotated, matrix, (width, height))
        
        # Random brightness/contrast adjustment
        alpha = np.random.uniform(0.8, 1.2)  # Contrast control
        beta = np.random.randint(-30, 30)    # Brightness control
        adjusted = cv2.convertScaleAbs(rotated, alpha=alpha, beta=beta)
        
        # Random blur
        if np.random.random() > 0.7:
            kernel_size = np.random.choice([3, 5])
            adjusted = cv2.GaussianBlur(adjusted, (kernel_size, kernel_size), 0)
        
        return adjusted
    
    def prepare_training_data(self, test_size=0.2, val_size=0.1):
        """Prepare data for training"""
        print("Preparing training data...")
        
        # Load metadata
        with open(self.metadata_file, 'r') as f:
            metadata = json.load(f)
        
        # Create label mapping
        card_ids = list(metadata.keys())
        label_to_id = {card_id: i for i, card_id in enumerate(card_ids)}
        id_to_label = {i: card_id for i, card_id in enumerate(card_ids)}
        
        # Get all image paths
        augmented_dir = os.path.join(self.data_dir, "augmented")
        image_files = [f for f in os.listdir(augmented_dir) if f.endswith('.jpg')]
        
        # Create dataset
        X_paths = []
        y_labels = []
        
        for img_file in image_files:
            card_id = img_file.split('_')[0]  # Extract card ID from filename
            if card_id in label_to_id:
                X_paths.append(os.path.join(augmented_dir, img_file))
                y_labels.append(label_to_id[card_id])
        
        # Split dataset
        X_train, X_test, y_train, y_test = train_test_split(
            X_paths, y_labels, test_size=test_size, random_state=42
        )

        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=val_size, random_state=42
        )
        
        # Save dataset splits
        splits = {
            'train': list(zip(X_train, y_train)),
            'val': list(zip(X_val, y_val)),
            'test': list(zip(X_test, y_test)),
            'label_mapping': id_to_label,
            'metadata': metadata
        }
        
        with open(os.path.join(self.data_dir, "dataset_splits.json"), 'w') as f:
            json.dump(splits, f, indent=2)
        
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        print(f"Test samples: {len(X_test)}")
        
        return splits

class MTGDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, image_paths, labels, batch_size=32, img_size=(224, 224), augment=False):
        self.image_paths = image_paths
        self.labels = labels
        self.batch_size = batch_size
        self.img_size = img_size
        self.augment = augment
        self.on_epoch_end()
    
    def __len__(self):
        return int(np.ceil(len(self.image_paths) / self.batch_size))
    
    def __getitem__(self, index):
        batch_paths = self.image_paths[index*self.batch_size:(index+1)*self.batch_size]
        batch_labels = self.labels[index*self.batch_size:(index+1)*self.batch_size]
        
        batch_images = []
        for path in batch_paths:
            image = self.load_image(path)
            if self.augment:
                image = self.augment_image(image)
            batch_images.append(image)
        
        return np.array(batch_images), np.array(batch_labels)
    
    def on_epoch_end(self):
        # Shuffle data at the end of each epoch
        if self.augment:
            indices = np.arange(len(self.image_paths))
            np.random.shuffle(indices)
            self.image_paths = [self.image_paths[i] for i in indices]
            self.labels = [self.labels[i] for i in indices]
    
    def load_image(self, path):
        # Always load as color (3 channels)
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is None:
            return np.zeros((self.img_size[0], self.img_size[1], 3))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.img_size)
        image = image / 255.0  # Normalize to [0, 1]
        return image
    
    def augment_image(self, image):
        # Random flipping
        if np.random.random() > 0.5:
            image = cv2.flip(image, 1)
        
        # Random rotation
        if np.random.random() > 0.7:
            angle = np.random.uniform(-10, 10)
            height, width = image.shape[:2]
            center = (width // 2, height // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            image = cv2.warpAffine(image, rotation_matrix, (width, height))
        
        # Random brightness/contrast
        alpha = np.random.uniform(0.9, 1.1)
        beta = np.random.uniform(-0.1, 0.1)
        image = np.clip(alpha * image + beta, 0, 1)
        
        return image

class MTGModel:
    def __init__(self, num_classes, input_shape=(224, 224, 3)):
        # Always use 3 channels for EfficientNetB0
        self.num_classes = num_classes
        self.input_shape = (224, 224, 3)
        self.model = None
    
    def build_model(self, base_model_name="EfficientNetB0"):
        """Build a CNN model for card recognition"""
        # Create base model with transfer learning
        if base_model_name == "EfficientNetB0":
            base_model = applications.EfficientNetB0(
                weights='imagenet',
                include_top=False,
                input_shape=self.input_shape
            )
        elif base_model_name == "ResNet50":
            base_model = applications.ResNet50(
                weights='imagenet',
                include_top=False,
                input_shape=self.input_shape
            )
        else:
            base_model = applications.MobileNetV2(
                weights='imagenet',
                include_top=False,
                input_shape=self.input_shape
            )
        
        # Freeze base model layers
        base_model.trainable = False
        
        # Add custom head
        inputs = tf.keras.Input(shape=self.input_shape)
        x = base_model(inputs, training=False)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        x = tf.keras.layers.Dense(512, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        outputs = tf.keras.layers.Dense(self.num_classes, activation='softmax')(x)
        
        self.model = tf.keras.Model(inputs, outputs)
        
        return self.model
    
    def compile_model(self, learning_rate=0.001):
        """Compile the model"""
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
    
    def unfreeze_layers(self, unfreeze_layers=10):
        """Unfreeze some layers for fine-tuning"""
        if self.model is None:
            raise ValueError("Model must be built first")
        
        # Unfreeze the top layers of the base model
        base_model = self.model.layers[1]  # The base model is the second layer
        base_model.trainable = True
        
        # Freeze all layers except the last few
        for layer in base_model.layers[:-unfreeze_layers]:
            layer.trainable = False
        
        # Recompile with lower learning rate
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

class MTGTrainer:
    def __init__(self, data_dir="mtg_dataset"):
        self.data_dir = data_dir
        self.splits = None
        self.model = None
        self.history = None
        
    def load_data(self):
        """Load dataset splits"""
        splits_path = os.path.join(self.data_dir, "dataset_splits.json")
        with open(splits_path, 'r') as f:
            self.splits = json.load(f)
        
        return self.splits
    
    def train(self, batch_size=32, epochs_initial=10, epochs_fine_tune=10):
        """Train the model"""
        if self.splits is None:
            self.load_data()
        
        # Prepare data generators
        train_paths, train_labels = zip(*self.splits['train'])
        val_paths, val_labels = zip(*self.splits['val'])
        
        train_gen = MTGDataGenerator(
            list(train_paths), list(train_labels), 
            batch_size=batch_size, augment=True
        )
        val_gen = MTGDataGenerator(
            list(val_paths), list(val_labels), 
            batch_size=batch_size, augment=False
        )
        
        num_classes = len(self.splits['label_mapping'])
        
        # Build and compile model
        mtg_model = MTGModel(num_classes)
        model = mtg_model.build_model("EfficientNetB0")
        mtg_model.compile_model()
        
        # Callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(factor=0.2, patience=3),
            tf.keras.callbacks.ModelCheckpoint(
                os.path.join(self.data_dir, "best_model.h5"),
                save_best_only=True,
                monitor='val_accuracy'
            )
        ]
        
        # Initial training
        print("Starting initial training...")
        history_initial = model.fit(
            train_gen,
            epochs=epochs_initial,
            validation_data=val_gen,
            callbacks=callbacks,
            verbose=1
        )
        
        # Fine-tuning
        print("Starting fine-tuning...")
        mtg_model.unfreeze_layers(20)
        
        history_fine_tune = model.fit(
            train_gen,
            epochs=epochs_fine_tune,
            validation_data=val_gen,
            callbacks=callbacks,
            verbose=1
        )
        
        # Combine histories
        self.history = {
            'initial': history_initial.history,
            'fine_tune': history_fine_tune.history
        }
        
        self.model = model
        return model, self.history
    
    def evaluate(self):
        """Evaluate the model on test set"""
        if self.splits is None:
            self.load_data()
        
        if self.model is None:
            raise ValueError("Model must be trained first")
        
        test_paths, test_labels = zip(*self.splits['test'])
        test_gen = MTGDataGenerator(
            list(test_paths), list(test_labels), 
            batch_size=32, augment=False
        )
        
        results = self.model.evaluate(test_gen, verbose=1)
        print(f"Test loss: {results[0]}, Test accuracy: {results[1]}")
        
        return results

class ImagePreprocessor:
    def __init__(self):
        # Parameters that can be adjusted
        self.blur_kernel_size = (5, 5)
        self.canny_low_threshold = 50
        self.canny_high_threshold = 150
        self.min_contour_area = 5000  # Minimum area to consider as a card
        self.contrast_alpha = 1.5  # Contrast control (1.0-3.0)
        self.contrast_beta = 0  # Brightness control (0-100)
        
    def load_image(self, image_path):
        """Load an image from file"""
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        return image
    
    def resize_image(self, image, max_dimension=1000):
        """Resize image while maintaining aspect ratio"""
        height, width = image.shape[:2]
        
        if max(height, width) > max_dimension:
            scale = max_dimension / max(height, width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            image = cv2.resize(image, (new_width, new_height))
            
        return image
    
    def adjust_brightness_contrast(self, image):
        """Adjust image contrast and brightness"""
        adjusted = cv2.convertScaleAbs(image, alpha=self.contrast_alpha, beta=self.contrast_beta)
        return adjusted
    
    def preprocess_for_contour_detection(self, image):
        """
        Prepare image for contour detection by:
        1. Converting to grayscale
        2. Applying Gaussian blur
        3. Detecting edges
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, self.blur_kernel_size, 0)
        
        # Detect edges using Canny edge detector
        edges = cv2.Canny(blurred, self.canny_low_threshold, self.canny_high_threshold)
        
        # Dilate edges to close gaps
        dilated = cv2.dilate(edges, None, iterations=2)
        
        return dilated
    
    def find_card_contours(self, edges):
        """
        Find contours in the edge image and identify potential card contours
        """
        # Find contours in the edge map
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        card_contours = []
        
        for contour in contours:
            # Calculate contour area
            area = cv2.contourArea(contour)
            
            # Skip small contours
            if area < self.min_contour_area:
                continue
                
            # Calculate perimeter
            perimeter = cv2.arcLength(contour, True)
            
            # Approximate the contour to reduce the number of points
            epsilon = 0.02 * perimeter  # Approximation accuracy
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Check if the approximated contour has 4 points (quadrilateral)
            if len(approx) == 4:
                # Check if the contour is convex (like a card should be)
                if cv2.isContourConvex(approx):
                    card_contours.append(approx)
        
        # Sort contours by area (largest first)
        card_contours = sorted(card_contours, key=cv2.contourArea, reverse=True)
        
        return card_contours
    
    def order_points(self, points):
        """
        Arrange points in consistent order:
        top-left, top-right, bottom-right, bottom-left
        """
        rect = np.zeros((4, 2), dtype="float32")
        s = points.sum(axis=1)
        rect[0] = points[np.argmin(s)]  # top-left
        rect[2] = points[np.argmax(s)]  # bottom-right
        diff = np.diff(points, axis=1)
        rect[1] = points[np.argmin(diff)]  # top-right
        rect[3] = points[np.argmax(diff)]  # bottom-left
        return rect
    
    def four_point_transform(self, image, points):
        """
        Apply perspective transform to get a bird's-eye view of the card
        """
        # Obtain a consistent order of the points
        rect = self.order_points(points)
        (tl, tr, br, bl) = rect
        
        # Compute the width of the new image
        width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        max_width = max(int(width_a), int(width_b))
        
        # Compute the height of the new image
        height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        max_height = max(int(height_a), int(height_b))
        
        # Define destination points
        dst = np.array([
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1]], dtype="float32")
        
        # Compute the perspective transform matrix
        M = cv2.getPerspectiveTransform(rect, dst)
        
        # Apply the perspective transformation
        warped = cv2.warpPerspective(image, M, (max_width, max_height))
        
        return warped
    
    def process_image(self, image_path):
        """
        Process an image to extract the card
        """
        # Load and resize image
        image = self.load_image(image_path)
        image = self.resize_image(image)
        
        # Adjust brightness and contrast
        enhanced = self.adjust_brightness_contrast(image)
        
        # Preprocess for contour detection
        edges = self.preprocess_for_contour_detection(enhanced)
        
        # Find card contours
        card_contours = self.find_card_contours(edges)
        
        if not card_contours:
            print("No card contours found")
            return None
        
        # Extract the first (largest) card
        card_contour = card_contours[0]
        
        # Apply perspective transform
        extracted_card = self.four_point_transform(image, card_contour.reshape(4, 2))
        
        return extracted_card

class MTGCardRecognizer:
    def __init__(self, model_path, metadata_path):
        self.model = tf.keras.models.load_model(model_path)
        
        with open(metadata_path, 'r') as f:
            data = json.load(f)
            self.metadata = data['metadata']
            self.label_mapping = data['label_mapping']
        
        self.id_to_label = {v: k for k, v in self.label_mapping.items()}
        self.img_size = (224, 224)
        self.preprocessor = ImagePreprocessor()
        
    def preprocess_image(self, image):
        """Preprocess image for model prediction"""
        # Always convert to RGB (3 channels)
        if image.shape[-1] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.img_size)
        image = image / 255.0
        image = np.expand_dims(image, axis=0)
        
        return image
    
    def predict(self, image):
        """Predict card from image"""
        processed_image = self.preprocess_image(image)
        predictions = self.model.predict(processed_image)
        
        # Get top 5 predictions
        top_5_indices = np.argsort(predictions[0])[-5:][::-1]
        top_5_confidences = predictions[0][top_5_indices]
        top_5_card_ids = [self.id_to_label[idx] for idx in top_5_indices]
        
        results = []
        for card_id, confidence in zip(top_5_card_ids, top_5_confidences):
            card_info = self.metadata.get(card_id, {}).copy()
            card_info['confidence'] = float(confidence)
            card_info['card_id'] = card_id
            results.append(card_info)
        
        return results
    
    def recognize_from_file(self, image_path):
        """Recognize card from image file"""
        # First extract the card from the image
        card_image = self.preprocessor.process_image(image_path)
        
        if card_image is None:
            raise ValueError("No card found in the image")
        
        # Then predict using the model
        return self.predict(card_image)

def main():
    """Main function to demonstrate the MTG card recognition"""
    print("MTG Card Recognition System - AI Model Approach")
    print("=" * 50)
    
    # Check if we need to train or can use existing model
    model_path = "mtg_dataset/best_model.h5"
    metadata_path = "mtg_dataset/dataset_splits.json"
    
    if not os.path.exists(model_path) or not os.path.exists(metadata_path):
        print("Model not found. Please train the model first.")
        print("This will download card images and train the AI model.")
        print("This may take a while...")
        
        response = input("Do you want to train the model now? (y/n): ")
        if response.lower() != 'y':
            print("Exiting. Please train the model first or place a trained model in mtg_dataset/")
            return
        
        # Step 1: Prepare data
        preparer = MTGDataPreparer("mtg_dataset")
        
        # Download images
        #preparer.download_card_images(limit=500)
        
        # Create augmented dataset
        #preparer.create_augmented_dataset(augmentations_per_image=5)
        
        # Prepare training data
        preparer.prepare_training_data(test_size=0.15, val_size=0.15)
        
        # Step 2: Train model
        trainer = MTGTrainer("mtg_dataset")
        trainer.load_data()
        
        model, history = trainer.train(
            batch_size=32,
            epochs_initial=15,
            epochs_fine_tune=10
        )
        
        # Step 3: Evaluate
        trainer.evaluate()
        
        # Step 4: Save final model
        model.save("mtg_dataset/mtg_card_recognizer.h5")
        print("Model trained and saved successfully!")
    
    # Now use the trained model
    recognizer = MTGCardRecognizer(
        "mtg_dataset/mtg_card_recognizer.h5",
        "mtg_dataset/dataset_splits.json"
    )
    
    # Test with an image
    test_image_path = input("Enter the path to your MTG card image: ").strip().strip('"')
    
    if not os.path.exists(test_image_path):
        print(f"Error: File '{test_image_path}' not found.")
        return
    
    print(f"Recognizing card in {test_image_path}...")
    
    try:
        results = recognizer.recognize_from_file(test_image_path)
        
        print("\nTop 5 predictions:")
        print("=" * 50)
        for i, result in enumerate(results, 1):
            print(f"{i}. {result.get('name', 'Unknown')} (Confidence: {result.get('confidence', 0):.2%})")
            print(f"   Set: {result.get('set', 'Unknown')}")
            print(f"   Type: {result.get('type', 'Unknown')}")
            print(f"   Mana Cost: {result.get('mana_cost', 'Unknown')}")
            print()
            
    except Exception as e:
        print(f"Error recognizing card: {e}")

if __name__ == "__main__":
    main()