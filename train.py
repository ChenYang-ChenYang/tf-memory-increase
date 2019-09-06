import tensorflow as tf
from model import TestModel

def get_batch(batch_size, inputs, targets):
    sindex = 0
    eindex = batch_size
    while eindex < len(inputs):
        batch_input = inputs[sindex:eindex]
        batch_target = targets[sindex:eindex]
        temp = eindex
        eindex = eindex + batch_size
        sindex = temp
        yield batch_input, batch_target

if __name__ == '__main__':
    embedding_size = 256
    input_steps = 30
    hidden_size = 50
    batch_size = 16
    epoch_num = 10
    vocab_size = 500
    intent_size = 60

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            tf.config.experimental.set_visible_devices(gpus[1], 'GPU')
        except RuntimeError as e:
            print(e)
    device = '/GPU:0'

    num_samples = 12800
    inputs = tf.random.uniform((num_samples, input_steps), -0.1, 0.1, dtype=tf.float32, name="input")
    targets = tf.random.uniform((num_samples, intent_size), -0.1, 0.1, dtype=tf.float32, name="output")

    with tf.device(device):
        model = TestModel(vocab_size, embedding_size, hidden_size, batch_size, intent_size)
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

        for epoch in range(epoch_num):
            for step, batch in enumerate(get_batch(batch_size, inputs, targets)):
                encoder_inputs = batch[0]
                intent_targets = batch[1]

                with tf.GradientTape() as tape:
                    intent_logits = model(encoder_inputs)
                    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(intent_targets, intent_logits)
                    loss = tf.reduce_mean(cross_entropy)
                    if step % 50 == 0:
                        print('epoch:{}, step:{}， intent loss:{:.2f}'.format(epoch, step, loss))
                    tf.random.set_seed(1)

                grads = tape.gradient(loss, model.trainable_variables)
                grads_norm, _ = tf.clip_by_global_norm(grads, 5)
                optimizer.apply_gradients(zip(grads_norm, model.trainable_variables))
