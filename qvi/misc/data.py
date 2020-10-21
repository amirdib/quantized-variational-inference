import numpy as np
import sklearn
import tensorflow as tf
def shuffle_split(samples, train_split, val_split):
    '''
    Shuffles the data and performs a train-validation-test split.
    Test = 1 - (train + val).
    
    :param samples: Samples from a dataset / data distribution.
    :param train: Portion of the samples used for training (float32, 0<=train<1).
    :param val: Portion of the samples used for validation (float32, 0<=val<1).
    :return train_data, val_data, test_data: 
    '''

    if train_split + val_split > 1:
        raise Exception('train_split plus val_split has to be smaller or equal to one.')

    batch_size = len(samples)
    np.random.shuffle(samples)
    n_train = int(round(train_split * batch_size))
    n_val = int(round((train_split + val_split) * batch_size))
    train_data = tf.cast(samples[0:n_train], dtype=tf.float32)
    val_data = tf.cast(samples[n_train:n_val], dtype=tf.float32)
    test_data = tf.cast(samples[n_val:batch_size], dtype=tf.float32)

    return train_data, val_data, test_data

def generate_2d_data(data, rng=None, batch_size=1000):
    if rng is None:
        rng = np.random.RandomState()

    if data == "swissroll":
        data = sklearn.datasets.make_swiss_roll(n_samples=batch_size, noise=1.0)[0]
        data = data.astype("float32")[:, [0, 2]]
        data /= 5
        return data, np.max(data)


    elif data == "circles":
        data = sklearn.datasets.make_circles(n_samples=batch_size, factor=.5, noise=0.08)[0]
        data = data.astype("float32")
        data *= 3
        return data, np.max(data)

    elif data == "rings":
        n_samples4 = n_samples3 = n_samples2 = batch_size // 4
        n_samples1 = batch_size - n_samples4 - n_samples3 - n_samples2

        # so as not to have the first point = last point, we set endpoint=False
        linspace4 = np.linspace(0, 2 * np.pi, n_samples4, endpoint=False)
        linspace3 = np.linspace(0, 2 * np.pi, n_samples3, endpoint=False)
        linspace2 = np.linspace(0, 2 * np.pi, n_samples2, endpoint=False)
        linspace1 = np.linspace(0, 2 * np.pi, n_samples1, endpoint=False)

        circ4_x = np.cos(linspace4)
        circ4_y = np.sin(linspace4)
        circ3_x = np.cos(linspace4) * 0.75
        circ3_y = np.sin(linspace3) * 0.75
        circ2_x = np.cos(linspace2) * 0.5
        circ2_y = np.sin(linspace2) * 0.5
        circ1_x = np.cos(linspace1) * 0.25
        circ1_y = np.sin(linspace1) * 0.25

        X = np.vstack([
            np.hstack([circ4_x, circ3_x, circ2_x, circ1_x]),
            np.hstack([circ4_y, circ3_y, circ2_y, circ1_y])
        ]).T * 3.0
        X = util_shuffle(X, random_state=rng)

        # Add noise
        X = X + rng.normal(scale=0.08, size=X.shape)

        return X.astype("float32"), np.max(X)

    elif data == "moons":
        data = sklearn.datasets.make_moons(n_samples=batch_size, noise=0.1)[0]
        data = data.astype("float32")
        data = data * 2 + np.array([-1, -0.2], dtype="float32")
        return data, np.max(data)

    elif data == "4gaussians":
        scale = 4.
        centers = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        centers = [(scale * x, scale * y) for x, y in centers]

        dataset = []
        for i in range(batch_size):
            point = rng.randn(2) * 0.5
            idx = rng.randint(4)
            center = centers[idx]
            point[0] += center[0]
            point[1] += center[1]
            dataset.append(point)
        dataset = np.array(dataset, dtype="float32")
        dataset /= 1.414
        return dataset, np.max(dataset)
    
    elif data == "8gaussians":
        scale = 4.
        centers = [(1, 0), (-1, 0), (0, 1), (0, -1), (1. / np.sqrt(2), 1. / np.sqrt(2)),
                   (1. / np.sqrt(2), -1. / np.sqrt(2)), (-1. / np.sqrt(2),
                                                         1. / np.sqrt(2)), (-1. / np.sqrt(2), -1. / np.sqrt(2))]
        centers = [(scale * x, scale * y) for x, y in centers]

        dataset = []
        for i in range(batch_size):
            point = rng.randn(2) * 0.5
            idx = rng.randint(8)
            center = centers[idx]
            point[0] += center[0]
            point[1] += center[1]
            dataset.append(point)
        dataset = np.array(dataset, dtype="float32")
        dataset /= 1.414
        return dataset, np.max(dataset)

    elif data == "pinwheel":
        radial_std = 0.3
        tangential_std = 0.1
        num_classes = 5
        num_per_class = batch_size // 5
        rate = 0.25
        rads = np.linspace(0, 2 * np.pi, num_classes, endpoint=False)

        features = rng.randn(num_classes*num_per_class, 2) \
            * np.array([radial_std, tangential_std])
        features[:, 0] += 1.
        labels = np.repeat(np.arange(num_classes), num_per_class)

        angles = rads[labels] + rate * np.exp(features[:, 0])
        rotations = np.stack([np.cos(angles), -np.sin(angles), np.sin(angles), np.cos(angles)])
        rotations = np.reshape(rotations.T, (-1, 2, 2))
        data = 2 * rng.permutation(np.einsum("ti,tij->tj", features, rotations))
        return data, np.max(data)

    elif data == "2spirals":
        n = np.sqrt(np.random.rand(batch_size // 2, 1)) * 540 * (2 * np.pi) / 360
        d1x = -np.cos(n) * n + np.random.rand(batch_size // 2, 1) * 0.5
        d1y = np.sin(n) * n + np.random.rand(batch_size // 2, 1) * 0.5
        x = np.vstack((np.hstack((d1x, d1y)), np.hstack((-d1x, -d1y)))) / 3
        x += np.random.randn(*x.shape) * 0.1
        return np.array(x, dtype='float32'), np.max(x)

    elif data == "checkerboard":
        x1 = np.random.rand(batch_size) * 4 - 2
        x2_ = np.random.rand(batch_size) - np.random.randint(0, 2, batch_size) * 2
        x2 = x2_ + (np.floor(x1) % 2)
        data = np.concatenate([x1[:, None], x2[:, None]], 1) * 2
        return np.array(data, dtype="float32"), np.max(data)

    elif data == "line":
        x = rng.rand(batch_size) * 5 - 2.5
        y = x
        data = np.stack((x, y), 1)
        return data, np.max(data)
    elif data == "cos":
        x = rng.rand(batch_size) * 5 - 2.5
        y = np.sin(x) * 2.5
        data = np.stack((x, y), 1)
        return data, np.max(data)
    elif data == "tum":
        mesh = np.zeros((400,400), dtype="float32")
        mesh[20:140, 100:130] = 1
        mesh[70:100, 120:300] = 1
        mesh[140:170, 100:300] = 1
        mesh[170:220, 250:300] = 1
        mesh[220:250, 100:300] = 1
        mesh[250:370, 100:130] = 1
        mesh[280:310, 120:300] = 1
        mesh[340:370, 120:300] = 1

        index = np.argwhere(mesh == 1)

        coordinates = index - 200
        coordinates[:,1] *= -1
        coordinates = coordinates / 50

        index_2 = np.random.randint(len(coordinates), size=batch_size)

        dataset = np.array(coordinates[index_2,:], dtype="float32")
        return dataset, np.max(dataset)
    else:
        data = generate_2d_data("8gaussians", rng, batch_size)
        return data, np.max(data)
