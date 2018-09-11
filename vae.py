# Copyright 2018 D-Wave Systems Inc.
# DVAE# licensed to authorized users only under the applicable license
# agreement.  See LICENSE.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

from autoencoder import SimpleDecoder, SimpleEncoder
from dist_util import FactorialBernoulliUtil, Spike_and_Exp, MixtureNormal, MixtureGeneric
from smoothing_util import Normal, PowerLaw, Exponential, ExponentialUniform
from util import get_global_step_var, Print, repeat_input_iw
from rbm import RBM, GuassianIntRBM, MarginalRBMType1Generic


class VAE:
    def __init__(self, num_input, config, config_recon, config_train):
        """  This function initializes an instance of the VAE class. 
        Args:
            num_input: the length of observed random variable (x).
            config: a dictionary containing config. for the (hierarchical) posterior distribution and prior over z. 
            config_recon: a dictionary containing config. for the reconstruct function in the decoder p(x | z).
            config_train: a dictionary containing config. training (hyperparameters).
        """
        np.set_printoptions(threshold=10)
        Print(str(config))
        Print(str(config_recon))
        Print(str(config_train))

        self.num_input = num_input
        self.config = config              # configuration dictionary for approx post and prior on z
        self.config_recon = config_recon  # configuration dictionary for reconstruct function p(x | z)
        self.config_train = config_train  # configuration dictionary for training hyper-parameters

        # bias term on the visible node
        with tf.name_scope("bias-visable-node"):
            self.train_bias = -np.log(1. / np.clip(self.config_train['mean_x'], 0.001, 0.999) - 1.).astype(np.float32)

        self.dist_type = config['dist_type']  # flag indicating whether we have rbm prior.
        tf.summary.scalar('beta', config['beta'])
        # define DistUtil classes that will be used in posterior and prior.
        if self.dist_type == "dvae_spike_exp":                                        # DVAE (spike-exp)
            dist_util = Spike_and_Exp
            dist_util_param = {'beta': self.config['beta']}
            tf.summary.scalar('posterior/beta', dist_util_param['beta'])
        elif self.dist_type == "dvaepp_exp":                                          # DVAE++ (exp)
            dist_util = MixtureGeneric
            self.smoothing_dist = Exponential(params={'beta': self.config['beta']})
            dist_util_param = {'smoothing_dist': self.smoothing_dist}
            tf.summary.scalar('posterior/beta', self.smoothing_dist.beta)
        elif self.dist_type == "dvaepp_power":                                        # DVAE++ (power)
            dist_util = MixtureGeneric
            self.smoothing_dist = PowerLaw(params={'beta': self.config['beta']})
            dist_util_param = {'smoothing_dist': self.smoothing_dist}
            tf.summary.scalar('posterior/lambda', self.smoothing_dist._lambda)
        elif self.dist_type == "dvaes_gi":                                            # DVAE# (Gaussian int)
            MixtureNormal.num_param = 2  # more parameters for
            dist_util = MixtureNormal
            dist_util_param = {'isotropic': False, 'delta_mu_scale': 0.5}
        elif self.dist_type == "dvaes_gauss":                                         # DVAE# (Gaussian)
            dist_util = MixtureGeneric
            self.smoothing_dist = Normal(params={'beta': self.config['beta']})
            dist_util_param = {'smoothing_dist': self.smoothing_dist}
            tf.summary.scalar('posterior/sigma', self.smoothing_dist.sigma)
        elif self.dist_type == "dvaes_exp":                                           # DVAE# (exp)
            dist_util = MixtureGeneric
            self.smoothing_dist = Exponential(params={'beta': self.config['beta']})
            dist_util_param = {'smoothing_dist': self.smoothing_dist}
            tf.summary.scalar('posterior/beta', self.smoothing_dist.beta)
        elif self.dist_type == "dvaes_unexp":                                         # DVAE# (uniform+exp)
            dist_util = MixtureGeneric
            self.smoothing_dist = ExponentialUniform(params={'beta': self.config['beta'], 'eps': 0.05})
            dist_util_param = {'smoothing_dist': self.smoothing_dist}
            tf.summary.scalar('posterior/beta', self.smoothing_dist.beta)
            tf.summary.scalar('posterior/eps', self.smoothing_dist.eps)
        elif self.dist_type == "dvaes_power":                                         # DVAE# (power)
            dist_util = MixtureGeneric
            self.smoothing_dist = PowerLaw(params={'beta': self.config['beta']})
            dist_util_param = {'smoothing_dist': self.smoothing_dist}
            tf.summary.scalar('posterior/lambda', self.smoothing_dist._lambda)
        else:
            raise ValueError('self.dist_type=%s is unknown' % self.dist_type)

        # define p(z)
        self.prior = self.define_prior()

        # create encoder for the first level.
        with tf.name_scope("encoder"):
            self.encoder = SimpleEncoder(num_input=num_input, config=config, dist_util=dist_util,
                                     dist_util_param=dist_util_param)

        # create encoder and decoder for lower layers.
        num_latent_units = self.config['num_latent_units'] * self.config['num_latent_layers']
        with tf.name_scope('decoder'):
            self.decoder = SimpleDecoder(num_latent_units=num_latent_units, num_output=num_input, config_recon=config_recon)

    def should_compute_log_z(self):
        return isinstance(self.prior, RBM)

    def define_prior(self):
        """ Defines the prior distribution over z. The prior will be an RBM or Normal prior based on self.dist_type.
         
        Returns:
            a DistUtil object representing the prior distribution.
        """
        # set up the rbm
        with tf.name_scope("rbm_prior"):
            num_var1 = self.config['num_latent_units'] * self.config['num_latent_layers'] // 2
            wd = self.config['weight_decay']
            if self.dist_type == 'dvaes_gi':
                rbm_prior = GuassianIntRBM(num_var1=num_var1, num_var2=num_var1, num_samples=1000, weight_decay=wd,
                                           use_qupa=self.config['use_qupa'],
                                           minimum_lambda=self.config['beta'])
            elif self.dist_type in {'dvaes_gauss', 'dvaes_exp', 'dvaes_power', 'dvaes_unexp'}:
                rbm_prior = MarginalRBMType1Generic(num_var1=num_var1, num_var2=num_var1, num_samples=1000, weight_decay=wd,
                                                    use_qupa=self.config['use_qupa']
                                                    , smoothing_dist=self.smoothing_dist)
            elif self.dist_type in {'dvae_spike_exp', 'dvaepp_exp', 'dvaepp_power'}:
                rbm_prior = RBM(num_var1=num_var1, num_var2=num_var1, num_samples=1000, weight_decay=wd,
                                kld_term=self.dist_type, use_qupa=self.config['use_qupa'])
            else:
                raise NotImplementedError

            return rbm_prior

    def generate_samples(self, num_samples):
        """ It will randomly sample from the model using ancestral sampling. It first generates samples from p(z_0).
        Then, it generates samples from the hierarchical distributions p(z_j|z_{i < j}). Finally, it forms p(x | z_i).  
        
         Args:
             num_samples: an integer value representing the number of samples that will be generated by the model.
        """
        with tf.name_scope("generate-samples"):
            if isinstance(self.prior, RBM):
                prior_samples = self.prior.samples
                prior_samples = tf.slice(prior_samples, [0, 0], [num_samples, -1])
            else:
                raise NotImplementedError

            output_activations = self.decoder.generator(prior_samples)
            output_activations[0] = output_activations[0] + self.train_bias
            output_dist = FactorialBernoulliUtil(output_activations)
            output_samples = tf.nn.sigmoid(output_dist.logit_mu)
            return output_samples

    def neg_elbo(self, input, is_training, k=1, use_iw=False):
        """ Defines the core operations that are used for both training and evaluation.
        
        Args:
            input: a 2D tensor containing current batch. 
            is_training: a boolean representing whether we are building the train or test computation graph.
            k: number of samples used for evaluating the objective function
            use_iw: A boolean flag indicating whether importance weighted bound is computed or not.
             
        Returns:
            iw_loss: importance weighted loss
            neg_elbo: is a scalar tensor containing negative EBLO computed for the batch. For training batch the KL 
              coeff is applied.   
            output: a tensor representing p(x|z) that is created by a single sample z~q(z|x).  
            wd_loss: a scalar tensor containing weight decay loss for all the networks.
            log_iw: a tensor of length batch size, representing the importance weights log p(x, z) / q(z|x). This
             will be used in the test batch for evaluating Log Likelihood.
        """
        with tf.name_scope("negative-elbo"):
            # subtract mean from input
            encoder_input = input - self.config_train['mean_x']

            # repeat the input if K > 1
            if k > 1:
                encoder_input = repeat_input_iw(encoder_input, k)
                input = repeat_input_iw(input, k)

            # form the encoder for z -- p(zeta|x)
            with tf.name_scope("hierarchical_posterior"):
                posterior, post_samples = self.encoder.hierarchical_posterior(encoder_input, is_training)

                # convert list of samples to single tensor
                post_samples_concat = tf.concat(axis=-1, values=post_samples)

            # create features for the likelihood p(x|z)
            with tf.name_scope("decoder"):
                output_activations = self.decoder.reconstruct(post_samples_concat, is_training)

            # add data bias
            output_activations[0] = output_activations[0] + self.train_bias
            # form the output dist util.
            output_dist = FactorialBernoulliUtil(output_activations)
            # create the final output
            output = tf.nn.sigmoid(output_dist.logit_mu)

            # compute KL only for VAE case
            total_kl = 0.
            if not use_iw:
                total_kl = self.prior.kl_dist_from(posterior, post_samples, is_training)

            # expected log prob p(x| z)
            cost = - output_dist.log_prob_per_var(input)
            cost = tf.reduce_sum(cost, axis=1)

            # weight decay loss
            with tf.name_scope("weight_decay_loss"):
                enc_wd_loss = self.encoder.get_weight_decay()
                dec_wd_loss = self.decoder.get_weight_decay()
                prior_wd_loss = self.prior.get_weight_decay() if isinstance(self.prior, RBM) else 0
                wd_loss = enc_wd_loss + dec_wd_loss + prior_wd_loss
                if is_training:
                    tf.summary.scalar('weigh decay/encoder', enc_wd_loss)
                    tf.summary.scalar('weigh decay/decoder', dec_wd_loss)
                    tf.summary.scalar('obj/recon_loss', tf.reduce_mean(cost))
                    tf.summary.scalar('obj/kl', tf.reduce_mean(total_kl))
                    tf.summary.scalar('weigh decay/total', wd_loss)

            # warm-up idea kl-term
            kl_coeff = self.kl_coeff_annealing(is_training)
            neg_elbo_per_sample = kl_coeff * total_kl + cost

            if k > 1:
                neg_elbo_per_sample = tf.reshape(neg_elbo_per_sample, [-1, k])
                neg_elbo_per_sample = tf.reduce_mean(neg_elbo_per_sample, axis=1)

            neg_elbo = tf.reduce_mean(neg_elbo_per_sample, name='neg_elbo')

            # compute importance weights
            if not is_training or (use_iw and k > 1):
                # log importance weight log p(z) - log q(z|x)
                log_iw = self.prior.log_prob(post_samples_concat, is_training)
                for i in range(len(posterior)):
                    log_iw -= posterior[i].log_prob(post_samples[i])

                # implement kl warm up for log_iw:
                log_iw = kl_coeff * log_iw
                # add p(x|z)
                log_iw -= cost
            else:
                log_iw = None

            # compute importance weighted loss
            if is_training and (use_iw and k > 1):
                log_iw_k = tf.reshape(log_iw, [-1, k])
                norm_w = tf.nn.softmax(log_iw_k)
                iw_loss_per_sample = tf.reduce_sum(tf.stop_gradient(norm_w) * log_iw_k, axis=1)
                iw_loss = - tf.reduce_mean(iw_loss_per_sample)
            else:
                iw_loss = None

            return iw_loss, neg_elbo, output, wd_loss, log_iw

    def kl_coeff_annealing(self, is_training):
        """ defines the coefficient used for annealing the KL term. It return 1 for the test graph but, a value
        between 0 and 1 for the training graph.
        
        Args:
            is_training: a boolean flag indicating whether the network is part of train or test graph. 

        Returns:
            kl_coeff: a scalar (non-trainable) tensor containing the kl coefficient.
        """
        global_step = get_global_step_var()
        if is_training:
            # anneal the KL coefficient in 30% iterations.
            max_epochs = 0.3 * self.config_train['num_iter']
            kl_coeff = tf.minimum(tf.to_float(global_step) / max_epochs, 1.)
            tf.summary.scalar('kl_coeff', kl_coeff)
        else:
            kl_coeff = 1.

        return kl_coeff

    def training(self, neg_elbo):
        """Sets up the training Ops.
        Creates an optimizer and applies the gradients to all trainable variables.

        Args:
            neg_elbo: neg_elbo tensor, from neg_elbo().

        Returns:
            train_op: The Op for training.
        """
        global_step = get_global_step_var()
        base_lr = self.config_train['lr']
        lr_values = [base_lr / 10, base_lr, base_lr / 3, base_lr / 10, base_lr / 33]
        boundaries = np.array([0.02, 0.6, 0.75, 0.95]) * self.config_train['num_iter']
        boundaries = [int(b) for b in boundaries]
        lr = tf.train.piecewise_constant(global_step, boundaries, lr_values)

        tf.summary.scalar('learning_rate', lr)
        optimizer = tf.train.AdamOptimizer(lr, epsilon=1e-3)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        print("Gradients TF Computes")
        for op in tf.trainable_variables():
            print(op)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(neg_elbo, global_step=global_step)

        return train_op
