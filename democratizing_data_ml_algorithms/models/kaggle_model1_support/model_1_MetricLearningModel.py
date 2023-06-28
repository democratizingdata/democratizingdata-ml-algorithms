import tensorflow as tf


class MetricLearningModel(tf.keras.Model):
    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.main_model = None
        self.support_dense = tf.keras.layers.Dense(units=768, activation=None)
        self.config = config
        self.K = 3

    def _compute_avg_embeddings(self, sequence_embeddings, attentions_mask, K=3):
        embeddings = tf.reduce_mean(
            attentions_mask * sequence_embeddings, axis=1
        )  # [B * K, F]
        if K > 1:
            embeddings = tf.reshape(
                embeddings,
                (-1, K, self.support_dense.units),
            )
            embeddings = tf.reduce_mean(embeddings, axis=1)  # [B, F]
        return embeddings

    def call(
        self,
        inputs,
        training=False,
        sequence_labels=None,
        mask_embeddings=None,
        nomask_embeddings=None,
        use_only_mask=False,
    ):
        output_hidden_states = self.main_model(
            input_ids=inputs[0], attention_mask=inputs[1], training=training
        )[-2]
        concat_hidden_states = tf.concat(
            output_hidden_states[-1:], axis=-1
        )  # [B * K, T, F]
        concat_hidden_states = self.support_dense(
            concat_hidden_states
        )  # [B * K, T, 768]
        sequence_embeddings = concat_hidden_states[:, 0, :]  # [B * K, 768]
        if sequence_labels is not None:
            sequence_labels = tf.cast(
                sequence_labels, dtype=concat_hidden_states.dtype
            )[..., None]
            mask_embeddings = self._compute_avg_embeddings(
                concat_hidden_states,
                tf.where(sequence_labels == -100, 0.0, sequence_labels),
                self.K,
            )
            nomask_embeddings = self._compute_avg_embeddings(
                concat_hidden_states,
                1.0 - tf.where(sequence_labels == -100, 1.0, sequence_labels),
                K=self.K,
            )
            return sequence_embeddings, mask_embeddings, nomask_embeddings
        else:
            attention_mask = tf.cast(inputs[1], concat_hidden_states.dtype)[
                ..., None
            ]  # [B, T, 1]
            normed_mask_embeddings = tf.nn.l2_normalize(mask_embeddings, axis=1)[
                ..., None
            ]
            normed_nomask_embeddings = tf.nn.l2_normalize(nomask_embeddings, axis=1)[
                ..., None
            ]
            normed_hidden_states = tf.nn.l2_normalize(concat_hidden_states, axis=-1)
            mask_cosine_similarity = tf.matmul(
                normed_hidden_states, normed_mask_embeddings
            )  # [B, T, 1]
            nomask_cosine_similarity = tf.matmul(
                normed_hidden_states, normed_nomask_embeddings
            )  # [B, T, 1]
            mask_attentions = tf.nn.sigmoid(10.0 * mask_cosine_similarity)  # [B, T, 1]
            nomask_attentions = tf.nn.sigmoid(
                10.0 * nomask_cosine_similarity
            )  # [B, T, 1]

            # average attention
            if use_only_mask:
                attentions = mask_attentions
            else:
                attentions = 0.5 * (mask_attentions + (1.0 - nomask_attentions))

            attentions *= attention_mask

            # compute mask and nomask embeddings
            mask_embeddings = self._compute_avg_embeddings(
                concat_hidden_states,
                tf.where(attention_mask == 0, 0.0, attentions),
                K=1,
            )
            nomask_embeddings = self._compute_avg_embeddings(
                concat_hidden_states,
                1.0 - tf.where(attention_mask == 0, 1.0, attentions),
                K=1,
            )
            return sequence_embeddings, mask_embeddings, nomask_embeddings, attentions
