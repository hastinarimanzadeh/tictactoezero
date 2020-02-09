import glob, time, os, json
import tensorflow as tf
import numpy as np

import models

if __name__ == "__main__":
    batch_size = 200
    while True:
        model = models.NeuralNetworkModel()
        optimizer = tf.keras.optimizers.Adam()

        checkpoint_directory = "./training_checkpoints"
        checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")
        checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
        status = checkpoint.restore(
                    tf.train.latest_checkpoint(checkpoint_directory)).expect_partial()

        directory_name = "./data/{}/".format(checkpoint.save_counter.numpy())
        files = glob.glob(os.path.join(directory_name, "*.json"))

        if len(files) > batch_size:
            history = []
            for file in files:
                with open(file) as f:
                    h = json.load(f)
                    history.extend(
                        [(np.array(x, dtype=np.float32),
                            np.array(p, dtype=np.float32), v) for x, p, v in h])

            states, policies, values = zip(*history)
            with tf.GradientTape() as tape:
                 estimated_policies, estimated_values = model(np.array(states))
                 losses = model.loss(estimated_policies, estimated_values,
                                    np.array(policies), np.array(values))
            print("loss:", np.mean(losses), "batch size:", len(history))
            grads = tape.gradient(losses, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            checkpoint.save(file_prefix=checkpoint_prefix)


        time.sleep(30)
