import json, random, os
import tensorflow as tf

import models, tictactoe

if __name__ == "__main__":
    while True:
        model = models.NeuralNetworkModel()
        checkpoint_directory = "./training_checkpoints"
        checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")
        checkpoint = tf.train.Checkpoint(model=model)
        status = checkpoint.restore(
            tf.train.latest_checkpoint(checkpoint_directory)).expect_partial()

        board = tictactoe.TicTacToe()
        history = models.compare_models(model, model, board)

        directory_name = "./data/{}/".format(checkpoint.save_counter.numpy())
        os.makedirs(directory_name, exist_ok=True)
        run_id = random.randint(0, 2**20-1)
        filename = os.path.join(directory_name,
                        "gamedata-{:05x}.json".format(run_id))

        with open(filename, 'w') as file:
            json.dump(
                [(x.tolist(), p.tolist(), v) for x, p, v in history],
                file)
