import argparse
import scipy.io
import scipy.misc
from nst_utils import *
import tensorflow as tf

STYLE_LAYERS = [
    ('conv1_1', 0.2),
    ('conv2_1', 0.2),
    ('conv3_1', 0.2),
    ('conv4_1', 0.2),
    ('conv5_1', 0.2)]


# print(model)
class StyleTransfer:

    def __init__(self):
        None

    def compute_content_cost(self, a_C, a_G):
        """
        Computes the content cost

        Arguments:
        a_C -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing content of the image C
        a_G -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing content of the image G

        Returns:
        J_content -- scalar that you compute using equation 1 above.
        """

        m, n_H, n_W, n_C = a_G.get_shape().as_list()

        a_C_unrolled = tf.reshape(a_C, (n_H * n_W, n_C))
        a_G_unrolled = tf.reshape(a_G, (n_H * n_W, n_C))

        J_content_cost = tf.reduce_sum(tf.squared_difference(a_C_unrolled, a_G_unrolled)) / (4 * n_H * n_W * n_C)

        return J_content_cost

    def gram_matrix(self, A):
        """
        Argument:
        A -- matrix of shape (n_C, n_H*n_W)

        Returns:
        GA -- Gram matrix of A, of shape (n_C, n_C)
        """

        GA = tf.matmul(A, tf.transpose(A))

        return GA

    def compute_layer_style_cost(self, a_S, a_G):
        """
        Arguments:
        a_S -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing style of the image S
        a_G -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing style of the image G

        Returns:
        J_style_layer -- tensor representing a scalar value, style cost defined above by equation (2)
        """

        m, n_H, n_W, n_C = a_G.get_shape().as_list()

        a_S = tf.reshape(tf.transpose(a_S), (n_C, n_H * n_W))
        a_G = tf.reshape(tf.transpose(a_G), (n_C, n_H * n_W))

        GS = self.gram_matrix(a_S)
        GG = self.gram_matrix(a_G)

        J_style_layer = tf.reduce_sum(tf.squared_difference(GS, GG)) / tf.to_float(
            4 * (n_C * n_C) * ((n_H * n_W) * (n_H * n_W)))

        return J_style_layer

    def compute_style_cost(self, sess, model, STYLE_LAYERS):
        """
        Computes the overall style cost from several chosen layers

        Arguments:
        model -- our tensorflow model
        STYLE_LAYERS -- A python list containing:
                            - the names of the layers we would like to extract style from
                            - a coefficient for each of them

        Returns:
        J_style -- tensor representing a scalar value, style cost defined above by equation (2)
        """

        J_style_cost = 0
        for layer_name, coeff in STYLE_LAYERS:
            out = model[layer_name]

            a_S = sess.run(out)
            a_G = out

            J_style_layer_cost = self.compute_layer_style_cost(a_S, a_G)

            J_style_cost += coeff * J_style_layer_cost

            return J_style_cost

    def total_cost(self, J_content, J_style, alpha=10, beta=40):
        """
        Computes the total cost function

        Arguments:
        J_content -- content cost coded above
        J_style -- style cost coded above
        alpha -- hyperparameter weighting the importance of the content cost
        beta -- hyperparameter weighting the importance of the style cost

        Returns:
        J -- total cost as defined by the formula above.
        """

        J_total_cost = alpha * J_content + beta * J_style

        return J_total_cost

    def main(self, args: argparse.Namespace):

        input_content_image = scipy.misc.imread(args.content_image)
        input_style_image = scipy.misc.imread(args.style_image)

        # TODO: Switch to TensorFlow 2.0
        tf.reset_default_graph()
        sess = tf.InteractiveSession()

        # Preprocess Images
        content_image = scipy.misc.imresize(input_content_image, (CONFIG.IMAGE_HEIGHT, CONFIG.IMAGE_WIDTH))
        style_image = scipy.misc.imresize(input_style_image, (CONFIG.IMAGE_HEIGHT, CONFIG.IMAGE_WIDTH))

        content_image = reshape_and_normalize_image(content_image)
        style_image = reshape_and_normalize_image(style_image)

        # Load model
        model = load_vgg_model(args.pretrained_model)

        # Run the model with the content image
        sess.run(model['input'].assign(content_image))

        # Select layer conv4_2
        out = model['conv4_2']

        # Compute content cost
        a_C = sess.run(out)
        a_G = out

        J_content_cost = self.compute_content_cost(a_C, a_G)

        # Compute style cost
        sess.run(model['input'].assign(style_image))

        J_style_cost = self.compute_style_cost(sess, model, STYLE_LAYERS)

        J_total_cost = self.total_cost(J_content_cost, J_style_cost, alpha=10, beta=40)

        with tf.device("/gpu:0"):
            # Optimizer
            optimizer = tf.train.AdamOptimizer(2.0)
            train_op = optimizer.minimize(J_total_cost)

        def model_nn(sess, num_iterations=200):

            generated_image = generate_noise_image(content_image)

            sess.run(tf.global_variables_initializer())
            sess.run(model['input'].assign(generated_image))

            for itr in range(num_iterations):

                sess.run(train_op)

                generated_image = sess.run(model['input'])

                if itr % 20 == 0:
                    Jt, Jc, Js = sess.run([J_total_cost, J_content_cost, J_style_cost])

                    print("Itr: {} Total cost: {} Content cost: {} Style cost: {}".format(itr, Jt, Jc, Js))
                    save_image("Images/GeneratedImages/" + str(itr) + ".png", generated_image)

            generated_image = save_image("Images/GeneratedImages/" + "FinalImage" + ".png", generated_image)

            return generated_image

        generated_image = model_nn(sess, num_iterations=args.epochs)

        # Display ContentImage + StyleImage = StyleTransferedContentImage
        f = plt.figure()
        f.add_subplot(1, 3, 1)

        plt.imshow(input_content_image)
        f.add_subplot(1, 3, 2)
        plt.imshow(input_style_image)

        f.add_subplot(1, 3, 3)
        plt.imshow(generated_image)

        plt.show()


def parse_arguments() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('--content-image', type=str)
    parser.add_argument('--style-image', type=str)
    parser.add_argument('--pretrained-model', type=str)
    parser.add_argument('--epochs', type=int, default=1000,
                        help='number of epochs to train (default: 1000)')
    return parser


if __name__ == '__main__':
    parser = parse_arguments()

    style = StyleTransfer()
    style.main(parser.parse_args())
