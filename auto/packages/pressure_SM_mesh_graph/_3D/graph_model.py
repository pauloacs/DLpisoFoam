import numpy as np
import tensorflow as tf


class MeshGraphShifter(tf.keras.Model):
    def __init__(self, graph, input_channels, width=24, steps=4, predict_velocity=False,
                 velocity_output_mode='direct'):
        super().__init__()
        if velocity_output_mode not in ('direct', 'shift_source', 'shift_source_component'):
            raise ValueError('Unsupported velocity_output_mode.')
        if velocity_output_mode != 'direct' and not predict_velocity:
            raise ValueError('Shift-source velocity output requires predict_velocity=True.')
        self.velocity_output_mode = velocity_output_mode
        self.source = tf.constant(graph['source'], dtype=tf.int32)
        self.target = tf.constant(graph['target'], dtype=tf.int32)
        self.edge_weights = tf.constant(graph['edge_weights'][:, None], dtype=tf.float32)
        self.node_count = int(graph['coarse_count'])
        length = max(float(np.linalg.norm(np.ptp(graph['coarse_centres'], axis=0))), 1e-12)
        offsets = graph['edge_offsets'] / length
        self.edge_features = tf.constant(np.column_stack((offsets, np.linalg.norm(offsets, axis=1))), dtype=tf.float32)
        self.encoder = tf.keras.layers.Dense(width, activation='swish')
        self.edge_encoder = tf.keras.layers.Dense(width, activation='tanh')
        self.node_updates = [tf.keras.layers.Dense(width, activation='swish') for _ in range(steps)]
        self.norms = [tf.keras.layers.LayerNormalization() for _ in range(steps)]
        self.shift = tf.keras.layers.Dense(3, activation='tanh', kernel_initializer='zeros')
        self.source_head = tf.keras.layers.Dense(1, kernel_initializer='zeros')
        self.velocity_head = tf.keras.layers.Dense(3, kernel_initializer='zeros') if predict_velocity and velocity_output_mode == 'direct' else None
        self.velocity_shift_head = (tf.keras.layers.Dense(3, activation='tanh', kernel_initializer='zeros')
                        if predict_velocity and velocity_output_mode in ('shift_source', 'shift_source_component') else None)
        self.velocity_source_head = (tf.keras.layers.Dense(3, kernel_initializer='zeros')
                         if predict_velocity and velocity_output_mode == 'shift_source' else None)
        self.velocity_component_shift_head = (tf.keras.layers.Dense(9, activation='tanh', kernel_initializer='zeros')
                 if predict_velocity and velocity_output_mode == 'shift_source_component' else None)
        self.velocity_component_source_head = (tf.keras.layers.Dense(3, kernel_initializer='zeros')
                 if predict_velocity and velocity_output_mode == 'shift_source_component' else None)
        self(tf.zeros((self.node_count, input_channels), dtype=tf.float32))

    def call(self, inputs, training=False):
        hidden = self.encoder(inputs)
        edge_encoding = self.edge_encoder(self.edge_features)
        for update, norm in zip(self.node_updates, self.norms):
            messages = (tf.gather(hidden, self.target) + edge_encoding) * self.edge_weights
            local = tf.math.unsorted_segment_sum(messages, self.source, self.node_count)
            context = tf.broadcast_to(tf.reduce_mean(hidden, axis=0, keepdims=True), tf.shape(hidden))
            hidden = norm(hidden + update(tf.concat((hidden, local, context), axis=1)))
        outputs = [self.shift(hidden), self.source_head(hidden)]
        if self.velocity_head is not None:
            outputs.append(self.velocity_head(hidden))
        if self.velocity_shift_head is not None:
            outputs.append(self.velocity_shift_head(hidden))
        if self.velocity_source_head is not None:
            outputs.append(self.velocity_source_head(hidden))
        if self.velocity_component_shift_head is not None:
            outputs.extend((self.velocity_component_shift_head(hidden), self.velocity_component_source_head(hidden)))
        return tf.concat(outputs, axis=1)


def reconstruct(outputs, gradient, target_scale):
    return -tf.reduce_sum(outputs[:, :3] * gradient, axis=1) + outputs[:, 3] * target_scale


def reconstruct_velocity_shift(outputs, gradient, length_scale):
    gradient = tf.convert_to_tensor(gradient, dtype=outputs.dtype)
    displacement = outputs[:, 4:7] * float(length_scale)
    source = outputs[:, 7:10]
    return -tf.einsum('nci,ni->nc', gradient, displacement) + source


def reconstruct_velocity_component_shift(outputs, gradient, length_scale, correction_factor=.25):
    gradient = tf.convert_to_tensor(gradient, dtype=outputs.dtype)
    shared_displacement = outputs[:, 4:7] * float(length_scale)
    component_displacement = tf.reshape(outputs[:, 7:16], (-1, 3, 3))
    component_displacement *= float(length_scale) * float(correction_factor)
    source = outputs[:, 16:19]
    shared_shift = -tf.einsum('nci,ni->nc', gradient, shared_displacement)
    component_shift = -tf.reduce_sum(gradient * component_displacement, axis=2)
    return shared_shift + component_shift + source