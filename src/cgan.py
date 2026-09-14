"""Conditional GAN for generating CDA crops, ported to the project's shared splits.

Provenance
----------
The original implementation (``$HOME/src/gans.py`` on the HPC) is **not** in the migrated
archive -- only its driver survives, at
``from_HPC/models_new/CGAN_Final/CGAN_GPU.py``. The architecture here is reconstructed
from ``from_HPC/GANs/0001_initial_GAN/initial_GAN.py``, which is the last complete
implementation that survived, and parameterised with the search grid the lost driver
used.

Conditioning follows that file exactly: the generator concatenates a one-hot label onto
the latent vector, and the discriminator projects the one-hot to a 64x64 plane and
concatenates it as a fourth channel.

What changed, and why
---------------------
The old driver split the data itself with ``train_val_test(..., 0.8, 0.1)``, a split
unrelated to the project's shared one. Measured against ``medoid_64``, **161 of the 308
held-out test crops (52.3%) were in that GAN's training set**. Any classifier trained on
its output was therefore trained on more than half the test set at one remove, so no test
number from those runs is interpretable. This module never sees anything but the ``train``
split.
"""

import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from scipy import linalg
from tensorflow.keras import layers

IMG_SIZE = 64
IMG_CHANNELS = 3
NUM_CLASSES = 7


def build_generator(latent_dim: int, filters: list[int]) -> tf.keras.Model:
    """Label-conditioned generator: [noise ; one-hot] -> 64x64x3 in [-1, 1].

    Starts at 8x8 and upsamples by three stride-2 transposed convolutions
    (8 -> 16 -> 32 -> 64), then a stride-1 layer to the colour channels, as in
    initial_GAN.py. ``filters`` sets the four widths.
    """
    noise_in = layers.Input(shape=(latent_dim,))
    label_in = layers.Input(shape=(NUM_CLASSES,))
    x = layers.Concatenate()([noise_in, label_in])

    x = layers.Dense(8 * 8 * filters[0], use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Reshape((8, 8, filters[0]))(x)

    for f in filters[1:]:
        x = layers.Conv2DTranspose(f, (5, 5), strides=(2, 2), padding="same",
                                   use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)

    out = layers.Conv2DTranspose(IMG_CHANNELS, (5, 5), strides=(1, 1), padding="same",
                                 use_bias=False, activation="tanh")(x)
    return tf.keras.Model([noise_in, label_in], out, name="generator")


def build_discriminator(filters: list[int], dropout: float = 0.3) -> tf.keras.Model:
    """Label-conditioned discriminator: the one-hot becomes a 4th image channel.

    ``dropout`` is one of the few direct controls on discriminator strength. The original
    pinned it at 0.3; it is swept here because a discriminator that wins too easily
    starves the generator of gradient.
    """
    image_in = layers.Input(shape=(IMG_SIZE, IMG_SIZE, IMG_CHANNELS))
    label_in = layers.Input(shape=(NUM_CLASSES,))

    lab = layers.Dense(IMG_SIZE * IMG_SIZE)(label_in)
    lab = layers.Reshape((IMG_SIZE, IMG_SIZE, 1))(lab)
    x = layers.Concatenate(axis=-1)([image_in, lab])

    for f in filters:
        x = layers.Conv2D(f, (5, 5), strides=(2, 2), padding="same")(x)
        x = layers.LeakyReLU()(x)
        x = layers.Dropout(dropout)(x)

    x = layers.Flatten()(x)
    out = layers.Dense(1)(x)  # logit; losses use from_logits=True
    return tf.keras.Model([image_in, label_in], out, name="discriminator")


def preprocess_images(images: np.ndarray) -> np.ndarray:
    """Scale to [-1, 1] to match the generator's tanh output.

    base_dataset stores crops already divided by 256, i.e. in [0, 1), so this maps that
    range rather than the 0-255 range initial_GAN.py assumed.
    """
    return (np.asarray(images, dtype="float32") * 2.0) - 1.0


def postprocess_images(images: np.ndarray) -> np.ndarray:
    """Inverse of preprocess_images, back to the [0, 1) scale the classifiers expect."""
    return np.clip((np.asarray(images, dtype="float32") + 1.0) / 2.0, 0.0, 1.0)


class FrechetCDADistance:
    """Frechet distance between real and generated crops in a trained CDA CNN's features.

    This is **not** FID. FID is defined on InceptionV3 pool3 features; those weights are an
    ImageNet download the compute nodes cannot make, and ImageNet features are a poor
    description of 64x64 leaf crops anyway. The same Frechet formula is applied to the last
    pooling layer of one of this project's own trained CNNs, so the number is comparable
    *across runs of this script* and with the original runs, but not with any published FID.

    The original implementation is ``extract_fid_features`` /``calculate_FID_score`` in
    ``Bulletproof_Models/src/gans.py``. It did the same thing -- a project CNN's features,
    not Inception's -- while labelling the result ``best_fid``. This class matches its
    feature choice exactly (the LAST POOLING LAYER, found by walking the layers in
    reverse) so the numbers stay comparable with those runs, and only corrects the name.
    """

    def __init__(self, feature_model_path: str | Path):
        base = tf.keras.models.load_model(feature_model_path)
        pool_types = (tf.keras.layers.MaxPooling2D,
                      tf.keras.layers.AveragePooling2D,
                      tf.keras.layers.GlobalAveragePooling2D)
        last_pool = next((l for l in reversed(base.layers) if isinstance(l, pool_types)), None)
        if last_pool is None:
            raise ValueError(f"{feature_model_path} has no pooling layer to take features from")
        self.model = tf.keras.Model(base.inputs, last_pool.output)
        self.name = Path(feature_model_path).parent.name
        self.layer = last_pool.name

    def _stats(self, images: np.ndarray, batch: int = 128):
        feats = self.model.predict(images, batch_size=batch, verbose=0)
        feats = feats.reshape(len(feats), -1)
        return feats.mean(0), np.cov(feats, rowvar=False)

    def __call__(self, real: np.ndarray, fake: np.ndarray) -> float:
        mu1, s1 = self._stats(real)
        mu2, s2 = self._stats(fake)
        diff = mu1 - mu2
        # No disp= kwarg: it was removed in scipy 1.18 (the HPC container), while the
        # local env still had 1.17 where it was accepted. Calling sqrtm(A) plainly
        # returns just the matrix in both, so this works either side.
        covmean = linalg.sqrtm(s1.dot(s2))
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        return float(diff.dot(diff) + np.trace(s1) + np.trace(s2) - 2 * np.trace(covmean))


def sample(generator, latent_dim: int, n_per_class: int, seed: int | None = None):
    """Generate ``n_per_class`` images for each of the 7 classes, in [0, 1)."""
    rng = tf.random.Generator.from_seed(seed) if seed is not None else tf.random.get_global_generator()
    imgs, labs = [], []
    for c in range(NUM_CLASSES):
        z = rng.normal([n_per_class, latent_dim])
        y = tf.keras.utils.to_categorical([c] * n_per_class, NUM_CLASSES)
        imgs.append(generator([z, tf.convert_to_tensor(y, tf.float32)], training=False).numpy())
        labs.append(np.full(n_per_class, c, dtype=int))
    return postprocess_images(np.concatenate(imgs)), np.concatenate(labs)


def discriminator_accuracy(discriminator, real_images, real_labels, fake_images, fake_labels):
    """Fraction of real/fake calls the discriminator gets right (0.5 = no better than chance)."""
    r = discriminator([real_images, real_labels], training=False).numpy().ravel()
    f = discriminator([fake_images, fake_labels], training=False).numpy().ravel()
    return float(((r > 0).sum() + (f <= 0).sum()) / (len(r) + len(f)))


def train_cgan(
    train_images,
    train_labels,
    params: dict,
    epochs: int,
    out_dir: Path,
    val_images=None,
    fcd: "FrechetCDADistance | None" = None,
    fcd_every: int = 50,
    sample_every: int = 250,
    seed: int = 42,
    patience: int = 4,
    disc_acc_window: tuple[float, float] = (0.35, 0.65),
    max_seconds: float | None = None,
):
    """Train one conditional GAN config and return its history.

    The best epoch is chosen by the Frechet CDA distance, not by either loss. GAN losses
    are adversarial and move against each other, so neither is a quality measure -- a
    generator loss can fall simply because the discriminator got worse.

    Early stopping follows ``check_early_stopping`` in the recovered
    ``Bulletproof_Models/src/gans.py``: stop after ``patience`` evaluations with no
    improvement in the Frechet distance, but ONLY while the discriminator's accuracy sits
    inside ``disc_acc_window``. That guard matters -- if the discriminator is winning
    (accuracy near 1.0) the generator is still starved of gradient and may yet recover, so
    stopping there would abandon a run that had not really plateaued. In the original run
    ``best_epoch_num`` ranged from 28 to 4,944, so without this many configs train for
    thousands of epochs past their best.
    """
    out_dir = Path(out_dir)
    (out_dir / "samples").mkdir(parents=True, exist_ok=True)
    (out_dir / "models").mkdir(parents=True, exist_ok=True)

    tf.keras.utils.set_random_seed(seed)
    latent_dim = params["latent_dim"]
    gen = build_generator(latent_dim, params["generator_filters"])
    disc = build_discriminator(params["discriminator_filters"],
                               dropout=params.get("disc_dropout", 0.3))
    # beta_1: Keras defaults to 0.9, the DCGAN convention is 0.5. High beta_1 keeps
    # pushing the generator along a direction the discriminator has already moved away
    # from, which is a well-known source of GAN instability. The original pinned it at
    # the Keras default across all 48 configs, so the grid could not find it.
    beta_1 = params.get("beta_1", 0.9)
    lr_g = params["learning_rate"]
    # Two time-scale update rule: the discriminator may run at a different rate. A ratio
    # of 1.0 reproduces the original's single shared learning rate.
    lr_d = lr_g * params.get("lr_ratio_d", 1.0)
    g_opt = tf.keras.optimizers.Adam(lr_g, beta_1=beta_1)
    d_opt = tf.keras.optimizers.Adam(lr_d, beta_1=beta_1)
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    # One-sided label smoothing (Salimans et al. 2016, arXiv:1606.03498 sec. 3.4): the
    # discriminator's target for REAL images becomes (1 - eps) rather than 1, and the fake
    # target stays at 0. Their reason for leaving the negative target alone is that with
    # positive target alpha and negative target beta the optimal discriminator is
    # (alpha*p_data + beta*p_model) / (p_data + p_model); a non-zero beta puts p_model in
    # the numerator, and then "in areas where p_data is approximately zero and p_model is
    # large, erroneous samples from p_model have no incentive to move nearer to the data".
    real_target = 1.0 - params.get("label_smoothing", 0.0)

    x = preprocess_images(train_images)
    y = tf.keras.utils.to_categorical(np.asarray(train_labels).astype(int), NUM_CLASSES)
    ds = (tf.data.Dataset.from_tensor_slices((x, y))
          .shuffle(len(x), seed=seed)
          .batch(params["batch_size"], drop_remainder=True)
          .prefetch(tf.data.AUTOTUNE))

    @tf.function
    def step(images, labels):
        noise = tf.random.normal([tf.shape(images)[0], latent_dim])
        with tf.GradientTape() as gt, tf.GradientTape() as dt:
            fake = gen([noise, labels], training=True)
            real_out = disc([images, labels], training=True)
            fake_out = disc([fake, labels], training=True)
            g_loss = bce(tf.ones_like(fake_out), fake_out)
            d_loss = (bce(tf.fill(tf.shape(real_out), real_target), real_out)
                      + bce(tf.zeros_like(fake_out), fake_out))
        g_opt.apply_gradients(zip(gt.gradient(g_loss, gen.trainable_variables),
                                  gen.trainable_variables))
        d_opt.apply_gradients(zip(dt.gradient(d_loss, disc.trainable_variables),
                                  disc.trainable_variables))
        return g_loss, d_loss

    # The original compared generated images against the VALIDATION split
    # (gans.py main(): ``val_images_fid = val_images / 256.0``), not the training split.
    # Keeping that: comparing against held-out real images makes the distance a
    # generalisation measure rather than a memorisation one.
    history, best = [], {"fcd": np.inf, "epoch": -1}
    wait, stopped_at = 0, None
    real_ref = np.asarray(val_images if val_images is not None else train_images,
                          dtype="float32")

    t_start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        gl = dl = n = 0.0
        for xb, yb in ds:
            a, b = step(xb, yb)
            gl += float(a); dl += float(b); n += 1
        rec = {"epoch": epoch, "gen_loss": gl / n, "disc_loss": dl / n,
               "epoch_seconds": time.perf_counter() - t0}

        if fcd is not None and (epoch % fcd_every == 0 or epoch == epochs):
            fake_imgs, fake_labs = sample(gen, latent_dim, 64, seed=epoch)
            rec["fcd"] = fcd(real_ref, fake_imgs)
            rec["disc_acc"] = discriminator_accuracy(
                disc,
                preprocess_images(real_ref[:len(fake_imgs)]),
                y[:len(fake_imgs)],
                preprocess_images(fake_imgs),
                tf.keras.utils.to_categorical(fake_labs, NUM_CLASSES),
            )
            if rec["fcd"] < best["fcd"]:
                best = {"fcd": rec["fcd"], "epoch": epoch}
                wait = 0
                gen.save(out_dir / "models" / "generator_best.keras")
                disc.save(out_dir / "models" / "discriminator_best.keras")
            else:
                wait += 1
                lo, hi = disc_acc_window
                if wait >= patience and lo <= rec["disc_acc"] <= hi:
                    stopped_at = epoch
                    history.append(rec)
                    print(f"  early stop at epoch {epoch}: no Frechet improvement for "
                          f"{wait} evaluations, discriminator accuracy "
                          f"{rec['disc_acc']:.3f} inside [{lo}, {hi}]", flush=True)
                    break

        if epoch % sample_every == 0 or epoch == epochs:
            imgs, labs = sample(gen, latent_dim, 4, seed=0)
            np.savez_compressed(out_dir / "samples" / f"epoch_{epoch:05d}.npz",
                                images=imgs, labels=labs)
        history.append(rec)
        # Wall-clock budget. The early-stopping rule above only fires while the
        # discriminator's accuracy is inside the window, so a config whose discriminator
        # has won outright (accuracy near 1.0) never stops and runs every epoch. That is
        # the expensive failure mode, and with a fixed grid it sets the total cost. Break
        # out here instead so the results and history CSVs are still written -- a SLURM
        # timeout would leave generator_best.keras on disk but no summary row.
        if max_seconds is not None and time.perf_counter() - t_start >= max_seconds:
            stopped_at = epoch
            print(f"  wall-clock budget of {max_seconds / 3600:.2f} h reached at epoch "
                  f"{epoch}; stopping cleanly (best epoch {best['epoch']})", flush=True)
            break
        if epoch % 100 == 0:
            print(f"  epoch {epoch}: gen {rec['gen_loss']:.3f} disc {rec['disc_loss']:.3f}"
                  + (f" fcd {rec.get('fcd', float('nan')):.2f}" if "fcd" in rec else ""),
                  flush=True)

    gen.save(out_dir / "models" / "generator_final.keras")
    best["stopped_at"] = stopped_at
    best["epochs_run"] = history[-1]["epoch"]
    return history, best
