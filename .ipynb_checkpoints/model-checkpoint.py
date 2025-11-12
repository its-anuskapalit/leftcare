import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    GlobalAveragePooling2D, Dense, Dropout, Conv2D, 
    Layer, Reshape, Activation, Multiply, Concatenate
)
from tensorflow.keras.applications import MobileNetV2
NUM_KNOWN_CLASSES = 16 
NUM_PROTOTYPE_CLASSES = 2 
INPUT_SHAPE = (224, 224, 3) 
BACKBONE_FEATURE_SIZE = (7, 7) 

class SymptomAttention(Layer):
    def __init__(self, **kwargs):
        super(SymptomAttention, self).__init__(**kwargs)
        self.conv_spatial = None
        self.conv_channel = None

    def build(self, input_shape):
        channels = input_shape[-1]
        self.conv_spatial = Conv2D(
            1, (1, 1), padding='same', activation=None, name='W1_spatial_conv'
        )
        self.conv_channel = Conv2D(
            1, (1, 1), padding='same', activation=None, name='W2_channel_conv'
        )
        
        super(SymptomAttention, self).build(input_shape)

    def call(self, inputs):
        F = inputs
        spatial_term = self.conv_spatial(F)
        Fc = GlobalAveragePooling2D()(F)
        Fc_expanded = Reshape((1, 1, Fc.shape[-1]))(Fc)
        channel_term = self.conv_channel(Fc_expanded)
        A_logits = tf.add(spatial_term, channel_term)
        A = Activation('softmax', name='attention_map_output')(A_logits)
        return A

def prototypical_head(input_tensor, embedding_dim=128):
   
    x = GlobalAveragePooling2D(name='proto_pool')(input_tensor)
    x = Dense(256, activation='relu', name='proto_dense_1')(x)
    # The final layer generates the low-dimensional embedding vector (z)
    z = Dense(embedding_dim, activation=None, name='prototypical_embedding_output')(x)
    return z
def build_multi_task_leafcare_model(
    input_shape=INPUT_SHAPE, 
    num_known_classes=NUM_KNOWN_CLASSES,
    embedding_dim=128
):
    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False, 
        weights='imagenet'
    )
    base_model.trainable = True
    F = base_model.output 
    shared_features = Conv2D(
        512, (1, 1), padding='same', name='shared_conv_512', activation='relu'
    )(F)
    
    pooled_features = GlobalAveragePooling2D(name='shared_pool')(shared_features)
    
    classification_output = Dense(
        num_known_classes, 
        activation='softmax', 
        name='classification_output'
    )(pooled_features)
    
    regression_dense = Dense(128, activation='relu', name='regression_dense')(pooled_features)
    health_output = Dense(
        1, 
        activation='sigmoid', 
        name='health_regression_output'
    )(regression_dense) 
    attention_output = SymptomAttention(name='symptom_attention_head')(F) 
    embedding_output = prototypical_head(F, embedding_dim=embedding_dim)

    
    model = Model(
        inputs=base_model.input,
        outputs=[
            classification_output, 
            health_output, 
            attention_output,
            embedding_output
        ],
        name='LeafCare_MultiTask_Model'
    )
    
    return model

# --- Example Usage ---

if __name__ == '__main__':
    leafcare_model = build_multi_task_leafcare_model(num_known_classes=18, embedding_dim=128)
    
    print("\n--- LeafCare Multi-Task Model Summary ---")
    leafcare_model.summary()
    
    losses = {
        'classification_output': 'categorical_crossentropy',
        'health_regression_output': 'mse', 
        'symptom_attention_head': 'binary_crossentropy', 
        'prototypical_embedding_output': None
    }
    
    leafcare_model.compile(
        optimizer='adam', 
        loss=losses, 
        metrics={'classification_output': ['accuracy'], 'health_regression_output': ['mae']}
    )
    print("\nModel built and compiled successfully.")