import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD

from tensorflow.keras.layers import Layer


@tf.custom_gradient
def kmean_operator(x, centroids):
    """
    Operator for a k-mean layer.

    :param x: The input tensor.
    :param centroids: The k centroids.

    :return: The centroid ids and values associated with ``x``.
    """
    # See https://stackoverflow.com/questions/56657993/how-to-create-a-keras-layer-with-a-custom-gradient-in-tf2-0
    # for more details on how to create a keras layer with a custom
    # gradient.

    # Compute the distance between each instance of x and each
    # centroid. The result is a tensor having one row per instance and
    # one column per centroid where t[i,j] is the distance between the
    # i-th instance and the j-th centroid.
    centroid_distance = tf.math.reduce_sum(
        tf.math.squared_difference(x, tf.expand_dims(centroids, 1)), axis=-1
    )

    # Get the centroid id that is the closest to each instance.
    x_centroid_ids = tf.argmin(centroid_distance, axis=0)
    x_centroids = tf.gather(centroids, x_centroid_ids)

    def gradient(x_centroid_ids_derivative, x_centroids_derivative):
        """
        Compute the gradient of this operation.

        :param x_centroid_ids_derivative: Derivative of ``x_centroid_ids``.
        :param x_centroids_derivative: Derivative of ``x_centroids``.
        :return: The partial derivative of ``x`` and ``centroids``.
        """
        del x_centroid_ids_derivative

        # Count the number of time each centroid was used
        _, idx, count = tf.unique_with_counts(x_centroid_ids)

        # Scale the derivative of each instance by the number of time
        # its centroid is updated. This is the value that will be
        # propagated to the previous layers.
        dy_dx = tf.math.divide(
            x_centroids_derivative,
            tf.expand_dims(tf.cast(tf.gather(count, idx), tf.float32), 1),
        )

        # Accumulate the derivative for each centroid. This will be
        # used to update the value of the centroids.
        dy_dcentroids = tf.scatter_nd(
            tf.expand_dims(x_centroid_ids, 1), dy_dx, centroids.shape
        )

        return dy_dx, dy_dcentroids

    # The ``gradient`` variable is _swallowed_ by the
    # tf.custom_gradient operator. The function only returns the
    # centroid ids and the centroid values to its caller.
    return [x_centroid_ids, x_centroids], gradient


class KMeanLayer(Layer):
    """
    A layer for K-Mean clustering.

    This layer map its input to the closest cluster, and return the cluster's centroid.
    """

    def __init__(
        self,
        k: int,
        centroids_initializer,
        name="KMeanLayer",
        debug=True,
        **kwargs,
    ):
        """
        :param k: The number of clusters.
        :param clusters_initializer: The function that will initialize the cluster values.
        :param debug: Whether to output debug metrics or not (this will slow down training).
        """
        super().__init__(name=name, **kwargs)
        self.k = k
        self.clusters_initializer = centroids_initializer
        self.debug = debug

    def build(self, input_shape=None):
        """
        Build the layer.

        This method will be called automatically, you don't need to call it.
        """
        self.clusters = self.add_weight(
            name="clusters",
            shape=(self.k, input_shape[-1]),
            initializer=self.clusters_initializer,
            dtype=tf.float32,
        )

    def call(self, inputs, training: bool = False):
        """
        Apply the layer to its inputs.
        """
        centroid_ids, centroids = kmean_operator(inputs, self.clusters)

        updated_centroids, _ = tf.unique(centroid_ids)

        if self.debug:
            # The update rate helps us understand if clusters are spreaded
            # evenly. The small update rate indicates that a small amount
            # of clusters are attracting most of the data points.
            self.add_metric(
                tf.shape(updated_centroids)[0] / self.k,
                name=f"{self.name}/centroids_update_rate",
                aggregation="mean",
            )

            self.add_metric(
                tf.sqrt(
                    tf.math.reduce_mean(tf.math.squared_difference(inputs, centroids))
                ),
                name=f"{self.name}/mean_distance",
                aggregation="mean",
            )

        return centroid_ids, centroids

    def top_k(self, inputs, k: int):
        """
        Map each input to its ``k`` closest cluster.

        :param inputs: The inputs.
        :param k: The number of clusters to map each input to.
        :return: A 3-tuple made of (k cluster ids, k centroids, k distance)
        """
        clusters = tf.expand_dims(self.clusters, 1)

        # Compute the distance between each instance and each cluster
        cluster_distance = tf.sqrt(
            tf.math.reduce_sum(tf.math.squared_difference(inputs, clusters), axis=-1)
        )

        # Compute the k closest centroids for each instance. Note that
        # we multiply the distance by -1 because top-k looks for the
        # largest value and we are interested in the smallest distance
        distances, indices = tf.math.top_k(-1 * tf.transpose(cluster_distance), k=k)

        return indices, tf.gather(self.clusters, indices), -1 * distances

    def get_config(self):
        # Needed by keras when serializing/deserializing the layer.
        config = {
            "k": self.k,
            "embeddings_initializer": tf.keras.initializers.serialize(
                self.clusters_initializer
            ),
        }

        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


class KMeanModel(Model):
    """
    A KMean model.
    """

    def __init__(
        self,
        k: int,
        centroids_initializer,
        name="KMeanModel",
        debug=True,
        **kwargs,
    ):
        """
        :param k: The number of centroids.
        :param centroids_initializer: Initalizer for centroids. Make sure that it is spreaded well into the input space, otherwise only a handful of centroids might be 'active'.
        :param debug: Whether to output debugging metrics (this will slow down training).
        """
        super().__init__(name=name, **kwargs)
        self.kmean_layer = KMeanLayer(k, centroids_initializer, debug=debug)

    def call(self, inputs, training=False):
        # return the id of the cluster that is the closest to the training data
        ids, centroids = self.kmean_layer(inputs, training=training)

        if training:
            self.add_loss(
                tf.math.reduce_mean(tf.math.squared_difference(inputs, centroids))
            )

        return ids

    def top_k(self, inputs, k: int):
        """
        :param k: Number of centroids to return.
        :return: A tuple with the closest centroid ids, centroids and distance.
        """
        return self.kmean_layer.top_k(inputs, k)


class PQModel(Model):
    def __init__(
        self,
        c: int,
        m: int,
        centroids_initializer,
        debug=True,
        name="PQModel",
        **kwargs,
    ):
        """
        :param input_dim: Size of the input vectors.
        :param c: Number of entry in each code book.
        :param m: Number of sub-quantizers.
        :param centroids_initializer: Initalizer for centroids. Make sure that it is spreaded well into the input space, otherwise only a handful of centroids might be 'active'.
        :param debug: Whether to output debugging metrics (this will slow down training).
        """
        super().__init__(name=name, **kwargs)
        self.c = c
        self.m = m
        self.sub_quantizers = [
            KMeanLayer(
                self.c, centroids_initializer, name=f"{self.name}/{i}", debug=debug
            )
            for i in range(self.m)
        ]

    def call(self, inputs, training=False):
        input_dim = inputs.shape[-1]
        sub_quantizer_input_dim = input_dim // self.m
        all_pq_codes = []
        all_values = []
        from_index = 0
        for q in self.sub_quantizers:
            to_index = min(from_index + sub_quantizer_input_dim, input_dim)
            pq_codes, values = q(inputs[:, from_index:to_index], training=training)
            all_pq_codes.append(tf.expand_dims(pq_codes, axis=1))
            all_values.append(values)
            from_index = to_index

        if training:
            values = tf.keras.layers.Concatenate(axis=1)(all_values)
            self.add_loss(
                tf.math.reduce_mean(tf.math.squared_difference(inputs, values))
            )

        return tf.keras.layers.Concatenate(axis=1)(all_pq_codes)

    def decode(self, codes) -> tf.Tensor:
        """
        Decode vectors that were PQ encoded.

        :param codes: The PQ codes to decode.
        """
        values = []
        for (i, q) in enumerate(self.sub_quantizers):
            values.append(tf.gather(q.weights[0], codes[:, i]))
        return tf.concat(values, axis=1)


class CQPQ(Model):
    def __init__(
        self,
        k,
        m,
        n,
        cq_centroids_initializer,
        pq_centroids_initializer,
        debug=True,
        name="CQPQ",
        **kwargs,
    ):
        """
        :param k: Number of coarse centroids.
        :param m: Number of sub-quantizers.
        :param n: Number of entries in each PQ code book.
        :param cq_centroids_initializer: Initalizer for coarse centroids. Make sure that it is spreaded well into the input space, otherwise only a handful of centroids might be 'active'.
        :param pq_centroids_initializer: Initalizer for pq centroids. Make sure that it is spreaded well into the input space, otherwise only a handful of centroids might be 'active'.
        :param debug: Whether to output debugging metrics (this will slow down training).
        """
        super().__init__()
        self.cq = KMeanLayer(
            k,
            name=f"{name}/cq",
            centroids_initializer=cq_centroids_initializer,
            debug=debug,
        )
        self.pq = PQModel(
            n,
            m,
            name=f"{name}/pq",
            centroids_initializer=pq_centroids_initializer,
            debug=debug,
        )

    def call(self, inputs, training=False):
        # one centroid id per input instance
        centroid_ids, centroids = self.cq(inputs)
        delta = tf.math.subtract(inputs, centroids)
        pq_delta = self.pq(delta)

        # loss between the end-to-end quantization and the inputs
        if training:
            decoded_delta = self.pq.decode(pq_delta)
            self.add_metric(
                tf.sqrt(
                    tf.math.reduce_mean(
                        tf.math.squared_difference(inputs, centroids + decoded_delta)
                    )
                ),
                name="cqpq_distance",
            )

            self.add_loss(
                tf.math.reduce_mean(
                    tf.math.squared_difference(inputs, centroids + decoded_delta)
                )
            )

        return centroid_ids, pq_delta

    def encode(self, query_embeddings):
        return self(query_embeddings)

    def n_probe(self, query_embeddings, n_probes):
        # top_k
        # gather centroids
        return self.cq.top_k(query_embeddings, k=n_probes)

    def score(self, query_delta, document_codes):
        document_delta = self.pq.decode(document_codes)
        # return the euclidean distance between the query and the document
        return tf.sqrt(tf.reduce_sum(tf.square(query_delta - document_delta), axis=1))


def test_cqpq():
    """
    Make sure that CQPQ models can de trained and used for inference.
    """
    data = np.random.default_rng(seed=42).random((4096, 64))

    # create a model with the right shape, loss, and optimizer

    # model = CQPQ(
    #     k=1024,
    #     m=16,
    #     n=128,
    #     cq_centroids_initializer=tf.random_uniform_initializer(0, 1),
    #     pq_centroids_initializer=tf.random_uniform_initializer(0, 1),
    #     debug=False,
    # )
    model = KMeanModel(1024, tf.random_uniform_initializer(0, 1), debug=False)

    model.compile(optimizer=SGD(learning_rate=1))
    model.fit(data, batch_size=256, epochs=32)
    return model
