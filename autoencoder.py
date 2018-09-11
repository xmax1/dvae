# Copyright 2018 D-Wave Systems Inc.
# DVAE# licensed to authorized users only under the applicable license
# agreement.  See LICENSE.

import tensorflow as tf
import copy

from dist_util import FactorialBernoulliUtil, MixtureNormal, Spike_and_Exp, MixtureGeneric
from nets import FeedForwardNetwork


class SimpleDecoder:
    def __init__(self, num_output, num_latent_units, config_recon, output_dist_util=FactorialBernoulliUtil):
        """ This function creates hierarchical decoder using a series of fully connected neural networks.   
        
        Args:
            num_output: number of output in the final output tensor. This can be equal to the length of x (the observed
              random variable).
            num_latent_units: number of latent units used in the prior.
            config_recon: a dictionary containing the hyper-parameters of the reconstruct network. See below for the keys required in the dictionary.
            output_dist_util: optional class indicating the distribution type of the output of the network.
              Only used to determine how outputs of the network should be "split". Default is FactorialBernoulliUtil,
              which has one parameter and so requires no splitting.
        """
        self.num_latent_units = num_latent_units
        self.num_output = num_output
        self.output_dist_util = output_dist_util

        # The final likelihood function p(x|z). The following makes the network that generate the output
        # used for the likelihood function.
        num_input = self.num_latent_units
        num_output = self.num_output * self.output_dist_util.num_param
        num_det_hiddens = [config_recon['num_det_units']] * config_recon['num_det_layers']
        weight_decay_recon = config_recon['weight_decay_dec']
        name = config_recon['name']
        use_batch_norm = config_recon['batch_norm']
        with tf.name_scope("decoder_network"):
            self.net = FeedForwardNetwork(
                num_input=num_input, num_hiddens=num_det_hiddens, num_output=num_output, name='%s_output' % name,
                weight_decay_coeff=weight_decay_recon, output_split=self.output_dist_util.num_param, use_batch_norm=use_batch_norm)

    def generator(self, prior_samples):
        """ This function generates samples using ancestral sampling from decoder. It accepts
        the samples from prior. This function can be used when samples from the model are being generated.
        
        Args:
            prior_samples:  A tensor containing samples from p(z).

        Returns:
            The output of likelihood function measured using the generated samples. 
        """
        return self.reconstruct(prior_samples, is_training=False)

    def reconstruct(self, post_samples, is_training):
        """ Given all the samples from the approximate posterior this function creates a network for
         p(x|z). It's output can be fed into a dist util object to create a distribution.
        
        Args:
            post_samples: A tensor containing samples for q(z | x) or p(z).
            is_training: A boolean indicating whether we are building a training graph or evaluation graph.

        Returns:
            output_dist: a FactorialBernoulliUtil object containing the logit probability of output.
        """
        hiddens = self.net.build_network(post_samples, is_training)
        return hiddens

    def get_weight_decay(self):
        """ Returns the weight decay loss for the decoder networks.
        
        Returns:
            wd_loss: a scalar tensor containing weight decay loss.
        """
        return self.net.get_weight_decay_loss()


class SimpleEncoder:
    def __init__(self, num_input, config, dist_util, dist_util_param={}):
        """ This function creates hierarchical encoder using a series of fully connected neural networks.   

        Args:
            num_input: number of input that will be fed to the networks. This can be equal to the length of x (the 
             observed random variable).
            config: a dictionary containing the hyper-parameters of the encoder. See below for the keys required in the dictionary.
            dist_util: is a class used for creating parameters of the posterior.
            dist_util_param: parameters that will be passed to the dist util when creating the prior objects.
        """
        self.num_input = num_input
        # number of latent layers (levels in the hierarchy)
        self.num_latent_layers = 0 if config is None else config['num_latent_layers']
        # the following keys are extracted to form the encoder.
        if self.num_latent_layers > 0:
            self.num_latent_units = config['num_latent_units']    # number of latent units per layer.
            self.num_det_units = config['num_det_units_enc']      # number of dererministic units in each layer.
            self.num_det_layers = config['num_det_layers_enc']    # number of deterministic layers in each conditional p(z_i | z_{k<i})
            self.weight_decay = config['weight_decay_enc']        # weight decay coefficient.
            self.name = config['name']                            # name used for variable scopes.
            self.use_batch_norm = config['batch_norm']
        self.nets = []
        self.dist_util = dist_util
        self.dist_util_param = dist_util_param

        # Define all the networks required for the autoregressive posterior.
        for i in range(self.num_latent_layers):
            num_det_hiddens = [self.num_det_units] * self.num_det_layers
            num_input = self.num_input + i * self.num_latent_units
            num_output = self.num_latent_units * self.dist_util.num_param
            with tf.name_scope("latent_layer_%02i" % (i+1)):
                network = FeedForwardNetwork(
                    num_input=num_input, num_hiddens=num_det_hiddens, num_output=num_output, name='%s_enc_%d' % (self.name, i),
                    weight_decay_coeff=self.weight_decay, output_split=self.dist_util.num_param, use_batch_norm=self.use_batch_norm)
                self.nets.append(network)

    def hierarchical_posterior(self, input, is_training):
        """ This function defines a hierarchical approximate posterior distribution. The length of the output is equal 
            to num_latent_layers and each element in the list is a DistUtil object containing posterior distribution 
            for the group of latent units in each hierarchy level. 

        Args:
            input: a tensor containing input tensor.
            is_training: A boolean indicating whether we are building a training graph or evaluation graph.

        Returns:
            posterior: a list of DistUtil objects containing posterior parameters.
            post_samples: A list of samples from all the levels in the hierarchy, i.e. q(z_k| z_{0<i<k}, x).
        """
        posterior = []
        post_samples = []

        for i in range(self.num_latent_layers):
            network_input = tf.concat(axis=-1, values=[input] + post_samples)  # concat x, z0, z1, ...
            network = self.nets[i]
            param = network.build_network(network_input, is_training)                      # create network
            # In the evaluation, we will use Bernoulli instead of continuous relaxations.
            if not is_training and self.dist_util in {MixtureNormal, Spike_and_Exp, MixtureGeneric}:
                posterior_dist = FactorialBernoulliUtil([param[0]])
            else:
                # define a specific scale parameter for each random variable independent of x (used for Gaussian Int.)
                if self.dist_util == MixtureNormal and not self.dist_util_param['isotropic']:
                    with tf.variable_scope('%s_mixture_%d' % (self.name, i), reuse=not is_training):
                        shape = [1, self.num_latent_units]
                        s_var = 0.2 * tf.get_variable(name='s', shape=shape, initializer=tf.ones_initializer)
                        dist_util_param = copy.deepcopy(self.dist_util_param)
                        dist_util_param['s'] = tf.abs(s_var) + 5e-2
                        if is_training:
                            tf.summary.histogram('posterior_s', s_var)
                else:
                    dist_util_param = self.dist_util_param
                posterior_dist = self.dist_util(param, dist_util_param)                    # init posterior dist.

            samples = posterior_dist.reparameterize(is_training)                      # reparameterize
            posterior.append(posterior_dist)
            post_samples.append(samples)

        return posterior, post_samples

    def get_weight_decay(self):
        """ Returns the weight decay loss for all the encoder networks.

        Returns:
            wd_loss: a scalar tensor containing weight decay loss.
        """
        wd_loss = 0
        for net in self.nets:
            wd_loss += net.get_weight_decay_loss()

        return wd_loss

