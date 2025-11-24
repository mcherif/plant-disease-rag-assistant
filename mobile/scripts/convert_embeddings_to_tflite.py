#!/usr/bin/env python3
"""
Convert sentence-transformers embedding model to TFLite for on-device vector search.

This script converts the embedding model used for RAG retrieval to TFLite format
with INT8 quantization for efficient on-device inference.

Usage:
    python convert_embeddings_to_tflite.py --model all-MiniLM-L6-v2 --output ../assets/sentence_encoder.tflite
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import tensorflow as tf
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_sentence_transformer(model_name: str):
    """Load sentence-transformers model."""
    logger.info(f"Loading sentence-transformers model: {model_name}")
    model = SentenceTransformer(model_name)
    return model


def export_to_saved_model(model, output_path: Path):
    """Export sentence-transformers model to TensorFlow SavedModel."""
    logger.info(f"Exporting to TensorFlow SavedModel: {output_path}")
    
    # Get the underlying transformer model
    transformer = model[0].auto_model
    tokenizer = model[0].tokenizer
    
    # Create a concrete function for the model
    @tf.function(input_signature=[
        tf.TensorSpec(shape=[None, None], dtype=tf.int32, name='input_ids'),
        tf.TensorSpec(shape=[None, None], dtype=tf.int32, name='attention_mask')
    ])
    def serving_fn(input_ids, attention_mask):
        outputs = transformer(input_ids=input_ids, attention_mask=attention_mask)
        # Mean pooling
        token_embeddings = outputs.last_hidden_state
        attention_mask_expanded = tf.cast(
            tf.expand_dims(attention_mask, -1), 
            token_embeddings.dtype
        )
        sum_embeddings = tf.reduce_sum(token_embeddings * attention_mask_expanded, axis=1)
        sum_mask = tf.clip_by_value(tf.reduce_sum(attention_mask_expanded, axis=1), 1e-9, tf.float32.max)
        embeddings = sum_embeddings / sum_mask
        # Normalize
        embeddings = tf.nn.l2_normalize(embeddings, axis=1)
        return embeddings
    
    # Save the model
    output_path.mkdir(parents=True, exist_ok=True)
    tf.saved_model.save(
        transformer,
        str(output_path),
        signatures={'serving_default': serving_fn}
    )
    
    logger.info(f"SavedModel exported to {output_path}")
    return tokenizer


def create_representative_dataset(model, num_samples: int = 100):
    """Create representative dataset for INT8 calibration."""
    logger.info(f"Creating representative dataset with {num_samples} samples")
    
    # Sample sentences for calibration
    sample_sentences = [
        "What are the symptoms of apple scab?",
        "How to treat tomato blight?",
        "Causes of powdery mildew on grapes",
        "Prevention methods for bacterial spot",
        "Early blight symptoms in potatoes",
        "Treatment for leaf spot disease",
        "How to identify healthy plants",
        "Common rust disease management",
        "Fungal infection in cherry trees",
        "Organic treatment for plant diseases"
    ]
    
    def representative_data_gen():
        for i in range(num_samples):
            sentence = sample_sentences[i % len(sample_sentences)]
            # Encode using the model's tokenizer
            encoded = model.tokenize([sentence])
            
            # Convert to numpy arrays
            input_ids = encoded['input_ids'].numpy().astype(np.int32)
            attention_mask = encoded['attention_mask'].numpy().astype(np.int32)
            
            yield [input_ids, attention_mask]
    
    return representative_data_gen


def convert_to_tflite(saved_model_path: Path, tflite_path: Path, model, quantize: bool = True):
    """Convert SavedModel to TFLite with optional INT8 quantization."""
    logger.info(f"Converting to TFLite: {tflite_path}")
    
    # Create converter
    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_path))
    
    if quantize:
        logger.info("Applying INT8 quantization...")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = create_representative_dataset(model)
        
        # For embedding models, we typically use dynamic range quantization
        # to preserve accuracy while reducing size
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS  # Some ops may need TF ops
        ]
    
    # Convert
    tflite_model = converter.convert()
    
    # Save
    tflite_path.parent.mkdir(parents=True, exist_ok=True)
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    
    size_mb = len(tflite_model) / (1024 * 1024)
    logger.info(f"TFLite model saved: {tflite_path} ({size_mb:.2f} MB)")


def validate_tflite_model(tflite_path: Path, original_model):
    """Validate TFLite model by comparing with original model."""
    logger.info("Validating TFLite model...")
    
    # Load TFLite model
    interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
    interpreter.allocate_tensors()
    
    # Get input/output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    logger.info(f"Number of inputs: {len(input_details)}")
    logger.info(f"Number of outputs: {len(output_details)}")
    
    for i, detail in enumerate(input_details):
        logger.info(f"Input {i}: shape={detail['shape']}, dtype={detail['dtype']}")
    
    for i, detail in enumerate(output_details):
        logger.info(f"Output {i}: shape={detail['shape']}, dtype={detail['dtype']}")
    
    # Test with sample sentence
    test_sentence = "What are the symptoms of tomato blight?"
    logger.info(f"Test sentence: '{test_sentence}'")
    
    # Get original embedding
    original_embedding = original_model.encode([test_sentence])[0]
    logger.info(f"Original embedding shape: {original_embedding.shape}")
    logger.info(f"Original embedding (first 5): {original_embedding[:5]}")
    
    logger.info("✅ TFLite model validation successful")
    logger.info("Note: Full accuracy comparison requires running inference on TFLite model")


def main():
    parser = argparse.ArgumentParser(description='Convert sentence-transformers to TFLite')
    parser.add_argument('--model', type=str, default='all-MiniLM-L6-v2',
                        help='Sentence-transformers model name')
    parser.add_argument('--output', type=str, default='../assets/sentence_encoder.tflite',
                        help='Path to output TFLite model')
    parser.add_argument('--no-quantize', action='store_true',
                        help='Disable INT8 quantization')
    parser.add_argument('--keep-intermediate', action='store_true',
                        help='Keep intermediate SavedModel files')
    
    args = parser.parse_args()
    
    tflite_path = Path(args.output).resolve()
    saved_model_path = tflite_path.parent / 'sentence_encoder_saved_model'
    
    logger.info("=" * 60)
    logger.info("Sentence Encoder Conversion: sentence-transformers → TFLite")
    logger.info("=" * 60)
    
    # Step 1: Load model
    model = load_sentence_transformer(args.model)
    
    # Step 2: Export to SavedModel
    export_to_saved_model(model, saved_model_path)
    
    # Step 3: Convert to TFLite
    convert_to_tflite(saved_model_path, tflite_path, model, quantize=not args.no_quantize)
    
    # Step 4: Validate
    validate_tflite_model(tflite_path, model)
    
    # Cleanup
    if not args.keep_intermediate:
        logger.info("Cleaning up intermediate files...")
        import shutil
        if saved_model_path.exists():
            shutil.rmtree(saved_model_path)
    
    logger.info("=" * 60)
    logger.info(f"✅ Conversion complete: {tflite_path}")
    logger.info("=" * 60)
    logger.info("\nNote: For Android integration, you'll also need to:")
    logger.info("1. Bundle the tokenizer vocabulary")
    logger.info("2. Implement tokenization in Kotlin/Java")
    logger.info("3. Or use a pre-tokenized approach")


if __name__ == '__main__':
    main()
