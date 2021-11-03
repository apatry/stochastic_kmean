import numpy as np

from stochastic_kmean.playground import *


def test_train_kmean():
    """
    Make sure that k-mean models can be trained and used for inference.
    """
    # create a dataset of 256 2D points
    data = np.random.default_rng(seed=42).random((256, 2))

    # train a kmean model and make sure it can be used
    model = KMeanModel(k=16, centroids_initializer=tf.random_uniform_initializer(0, 1))
    model.compile()
    assert True, "model compilation was successful"

    model.fit(data, batch_size=32, epochs=32)
    assert True, "k-mean training was successful"

    for cluster in np.array(model(data)):
        assert isinstance(cluster, np.int64), "clusters are integers"
        assert 0 <= cluster < 16, "clusters are within range"


def test_kmean_inference():
    # hardcode clusters so that clusters[i] is (i, i)
    clusters = tf.convert_to_tensor([(float(i), float(i)) for i in range(11)])
    model = KMeanModel(
        k=clusters.shape[0], centroids_initializer=tf.random_uniform_initializer(0, 1)
    )
    model.build(input_shape=(2,))
    model.set_weights([clusters])

    assert model([0.0, 0.0]) == tf.constant(
        [0], dtype=np.int64
    ), "The closest cluster is picked up"

    assert model([6.6, 7.1]) == tf.constant(
        [7], dtype=np.int64
    ), "The closest cluster is picked up"

    assert np.array_equal(
        model.top_k([6.6, 7.1], 3)[0], [[7, 6, 8]]
    ), "Top-k cluster ids are picked-up"


def test_product_quantization():
    """
    Make sure PQ models can be trained and used for inference.
    """
    data = np.random.default_rng(seed=42).random((256, 8))

    # train a pq model and make sure it can be used for inference
    model = PQModel(
        c=16, m=4, centroids_initializer=tf.random_uniform_initializer(0, 1)
    )
    model.compile()
    assert True, "model compilation was successful"

    model.fit(data, batch_size=32, epochs=32)
    assert True, "product quantization training was successful"

    codes = np.array(model(data))
    assert codes.shape == (data.shape[0], 4), "m codes per instance are returned"
    assert [
        code for code in codes.flatten() if code < 0 or code >= 16
    ] == [], "codes are all within range"

    roundtrip = model.decode(model(data))
    assert roundtrip.shape == data.shape
    assert np.array_equal(
        model.decode(model(roundtrip)), roundtrip
    ), "The output of a PQ encoder can be re-encoded without loss"


def test_cqpq():
    """
    Make sure that CQPQ models can de trained and used for inference.
    """
    data = np.random.default_rng(seed=42).random((256, 8))

    # create a model with the right shape, loss, and optimizer
    model = CQPQ(
        k=16,
        m=4,
        n=8,
        cq_centroids_initializer=tf.random_uniform_initializer(0, 1),
        pq_centroids_initializer=tf.random_uniform_initializer(-0.1, 0.1),
    )
    model.compile()
    assert True, "model compilation was successful"

    model.fit(data, batch_size=32, epochs=32)
    assert True, "product quantization training was successful"

    model(data)
    centroid_ids, pq_codes = model.encode(data)
    assert centroid_ids.shape == (data.shape[0],)
    assert pq_codes.shape == (data.shape[0], model.pq.m)

    centroid_ids, centroids, distances = model.n_probe(data, 3)
    assert centroid_ids.shape == (data.shape[0], 3)
    assert centroids.shape == (data.shape[0], 3, data.shape[1])
    assert distances.shape == (data.shape[0], 3)

    scores = model.score(data - centroids[:, 0, :], pq_codes)
    assert scores.shape == (data.shape[0],)
