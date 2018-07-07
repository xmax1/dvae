# Copyright 2018 D-Wave Systems Inc.
# DVAE# licensed to authorized users only under the applicable license
# agreement.  See LICENSE.

import tensorflow as tf
import numpy as np
import math

from smoothing_util import SmoothingDist


def sigmoid_cross_entropy_with_logits(logits, labels):
    """"
    See tensorflow.nn.sigmoid_cross_entropy_with_logits documentation.
    """
    return logits - logits * labels + tf.nn.softplus(-logits)


class DistUtil:
    def reparameterize(self, is_training):
        raise NotImplementedError

    def kl_dist_from(self, dist_util_obj, aux):
        raise NotImplementedError

    def entropy(self):
        raise NotImplementedError

    def log_prob(self, samples):
        raise NotImplementedError


class FactorialBernoulliUtil(DistUtil):
    num_param = 1

    def __init__(self, param, kw_param={}):
        """
        Set the logits of the factorial Bernoulli distribution.
        Args:
            param: params[0] is the logit of the probability of the random binary variables being 1.
            kw_param: not required.
        """
        assert isinstance(param, list), 'param should be a list.'
        assert len(param) == 1, 'param should have a length of one corresponding to logit_mu.'
        self.logit_mu = param[0]

    def reparameterize(self, is_training):
        """ 
        Samples from the bernoulli distribution. Can be used only during test.
        Args:
            is_training: a flag indicating whether we are building the training computation graph
            
        Returns:
            z: samples from bernoulli distribution
        """
        if is_training:
            raise NotImplementedError('Reparameterization of a bernoulli distribution is not differentiable.')
        else:
            q = tf.nn.sigmoid(self.logit_mu)
            rho = tf.random_uniform(shape=tf.shape(q), dtype=tf.float32)
            ind_one = tf.less(rho, q)
            z = tf.where(ind_one, tf.ones_like(q), tf.zeros_like(q))
            return z

    def log_ratio(self, zeta):
        """
        A dummy function for this class.
        Args:
            zeta: approximate post samples

        Returns:
            log_ratio: 0.
        """
        log_ratio = 0. * (2 * zeta - 1)
        return log_ratio

    def entropy(self):
        """
        Computes the entropy of the bernoulli distribution using:
            x - x * z + log(1 + exp(-x)),  where x is logits, and z=sigmoid(x).
        Returns: 
            ent: entropy
        """
        mu = tf.nn.sigmoid(self.logit_mu)
        ent = sigmoid_cross_entropy_with_logits(logits=self.logit_mu, labels=mu)
        return ent

    def log_prob_per_var(self, samples):
        """
        Compute the log probability of samples under distribution of this object.
            - (x - x * z + log(1 + exp(-x))),  where x is logits, and z is samples.
        Args:
            samples: matrix of size (num_samples * num_vars)

        Returns: 
            log_prob: a matrix of log_prob (num_samples * num_vars).
        """
        log_prob = - sigmoid_cross_entropy_with_logits(logits=self.logit_mu, labels=samples)
        return log_prob

    def log_prob(self, samples):
        """
        Call log_prob_per_var() and then compute the sum of log_prob of all the variables.
        Args:
            samples: matrix of size (num_samples * num_vars)

        Returns: 
            log_p: A vector of log_prob for each sample.
        """
        log_p = self.log_prob_per_var(samples)
        log_p = tf.reduce_sum(log_p, 1)
        return log_p


class Spike_and_Exp(FactorialBernoulliUtil):
    num_param = 1

    def __init__(self, param, kw_param):
        """
        Set the logits of the factorial Bernoulli distribution defined on the z components. 
        Args:
            param: params[0] is the logit of the probability of the random binary variables being 1.
            kw_param: a dictionary containing beta parameter of the spike-and-exp.
        """
        FactorialBernoulliUtil.__init__(self, param)
        self.beta = kw_param['beta']

    def reparameterize(self, is_training):
        """ 
        Samples from the spike-and-exp distribution.
        Samples from the inverse-CDF of a distribution consisting of a delta-spike at zero with magnitude
        1 - sigmoid(logistic_input), and otherwise the PDF beta/(e^beta - 1) * e^(beta * z)
        
        Args:
            is_training: a flag indicating whether we are building the training computation graph

        Returns:
            z: samples from bernoulli distribution
        """
        logistic_input = self.logit_mu
        approx_post_prob = tf.sigmoid(logistic_input)
        approx_post_prob = tf.clip_by_value(approx_post_prob, clip_value_min=1e-7, clip_value_max=1. - 1e-7)
        uniform_samples = tf.random_uniform(shape=tf.shape(approx_post_prob), minval=0.0, maxval=1.0)
        # (rho - 1)/q + 1 if rho > 1 - q; 0 otherwise
        rectifier = tf.to_float(tf.greater(uniform_samples, 1.0 - approx_post_prob))  # rho >= 1 - q
        rho_transformed = (((uniform_samples - 1.0) / approx_post_prob) + 1.0) * rectifier  # (rho + q - 1)/q
        # c_exp = tf.exp if self._inv_cdf_trainable else math.exp  # use TF operations on beta, rather than math ops
        c_exp = tf.exp if not isinstance(self.beta, float) else math.exp  # use TF operations on beta, rather than math ops
        samples = (1.0 / self.beta) * tf.log((c_exp(self.beta) - 1.0) * rho_transformed + 1.0) * rectifier

        return samples


class MixtureNormal(FactorialBernoulliUtil):
    num_param = 1

    def __init__(self, param, kw_param={}):
        """
        Set the logits of the factorial Bernoulli distribution defined on the z components. 
        Args:
            param: params[0] is the logit of the probability of the random binary variables being 1.
                   param[1] is the \delta \mu required for shifting means in shifted Gaussian.
            kw_param: a dictionary containing sigma parameter of normal components and the scale parameter
                      for delta mu for the case where the Gaussian are shifted (see DVAE# Sec. 4).
        """
        assert isinstance(param, list), 'param should be a list.'
        assert len(param) == MixtureNormal.num_param, \
            "param should have a length of one or two depending on kw_param['isotropic']"
        FactorialBernoulliUtil.__init__(self, [param[0]])

        self.sigma = kw_param['s']
        self.log_sigma = tf.log(self.sigma)
        if kw_param['isotropic']:     # indicates whether the distributions are shifted Gaussian
            self.delta_mu = 0.
        else:
            self.delta_mu = kw_param['delta_mu_scale'] * (tf.nn.sigmoid(param[1]) - 0.5)

    def conditional_sample(self, mean):
        """
        This function samples from r(\zeta|z + delta_mu) assuming that r is a Gaussian distribution with mean set to 
        z + delta_mu and sigma set to self.sigma.
        Args:
            mean: mean of the Gaussian distribution.
        Returns:
            zeta: samples generated from conditional
        """
        eps_shape = tf.shape(self.logit_mu)
        eps = tf.random_normal(eps_shape)
        zeta = mean + self.sigma * eps
        return zeta

    @staticmethod
    def normal_pdf(z, mu, sigma):
        pdf = 1. / (np.sqrt(2 * np.pi) * sigma) * tf.exp(-0.5 * (z - mu) * (z - mu) / (sigma * sigma))
        return pdf

    @staticmethod
    def normal_cdf(z, mu, sigma):
        return 0.5 * (1. + tf.erf((z - mu) / (np.sqrt(2.) * sigma)))

    def reparameterize(self, is_training):
        """"
        This function uses ancestral sampling to sample from mixture of two Gaussians centered at zero and one 
        (+\delta \mu for the case of shifted Gaussians). It then uses the implicit gradient idea to compute
        the gradient of samples with respect to \beta, logit_q, and \delta_mu. This idea is presented in DVAE# sec 3.4.
        
        Args:
            is_training: a flag indicating whether we are building the training computation graph

        Returns:
            zeta: samples from mixture of Gaussian distributions.
        """
        q = tf.nn.sigmoid(self.logit_mu)

        z = FactorialBernoulliUtil.reparameterize(self, is_training=False)
        mu = z + self.delta_mu
        zeta = self.conditional_sample(mu)

        mu0 = 0. + self.delta_mu
        mu1 = 1. + self.delta_mu
        pdf_0 = self.normal_pdf(zeta, mu0, self.sigma)
        pdf_1 = self.normal_pdf(zeta, mu1, self.sigma)
        cdf_0 = self.normal_cdf(zeta, mu0, self.sigma)
        cdf_1 = self.normal_cdf(zeta, mu1, self.sigma)

        grad_q = (cdf_0 - cdf_1) / (q * pdf_1 + (1 - q) * pdf_0)
        grad_q = tf.stop_gradient(grad_q)
        grad_term_q = grad_q * q
        grad_term_q -= tf.stop_gradient(grad_term_q)

        grad_s = ((1 - q) * pdf_0 * zeta + q * pdf_1 * (zeta - 1)) / self.sigma / ((1 - q) * pdf_0 + q * pdf_1)

        grad_s = tf.stop_gradient(grad_s)
        grad_term_s = grad_s * self.sigma
        grad_term_s -= tf.stop_gradient(grad_term_s)

        # the gradient of zeta w.r.t delta_mu is 1.
        grad_delta_mu = 1.
        grad_term_delta_mu = grad_delta_mu * self.delta_mu
        grad_term_delta_mu -= tf.stop_gradient(grad_term_delta_mu)
        
        zeta = tf.stop_gradient(zeta) + grad_term_q + grad_term_s + grad_term_delta_mu

        if is_training:
            # tf.summary.scalar('posterior/log_s', self.log_s)
            tf.summary.histogram('sigma', self.sigma)
            tf.summary.histogram('q', q)
            mean_q, var_q = tf.nn.moments(q, axes=[0])
            tf.summary.scalar('posterior/active_q', tf.reduce_sum(tf.to_float(tf.greater(var_q, 0.05))))
            tf.summary.histogram('var_q', var_q)
            tf.summary.histogram('mean_q', mean_q)
            tf.summary.histogram('delta_mu', self.delta_mu)
        return zeta

    def log_prob_per_var(self, samples):
        """
        Compute the log probability of samples under mixture of Gaussian distributions.
        Args:
            samples: matrix of size (num_samples * num_vars)

        Returns: 
            log_prob: a matrix of log_prob (num_samples * num_vars).
        """
        mu0 = 0. + self.delta_mu
        mu1 = 1. + self.delta_mu
        sigma2 = tf.square(self.sigma)
        log_sigma = self.log_sigma
        logit_q = self.logit_mu
        q = tf.nn.sigmoid(self.logit_mu)

        log_prob_mu0 = - 0.5 * tf.square(samples - mu0) / sigma2 - 0.5 * tf.log(2 * np.pi) - log_sigma

        q = tf.clip_by_value(q, 0., 1. - 1e-4)
        log_prob = tf.log(1. - q) + log_prob_mu0 + \
                   tf.nn.softplus(logit_q + 0.5 * (2 * samples * (mu1 - mu0) + tf.square(mu0) - tf.square(mu1)) / sigma2)

        return log_prob

    def log_prob(self, samples):
        """
        Call log_prob_per_var() and then compute the sum of log_prob of all the variables.
        Args:
            samples: matrix of size (num_samples * num_vars)

        Returns: 
            log_p:A vector of log_prob for each sample.
        """
        log_p = self.log_prob_per_var(samples)
        log_p = tf.reduce_sum(log_p, 1)
        return log_p


class MixtureGeneric(FactorialBernoulliUtil):
    num_param = 1

    def __init__(self, param, kw_param={}):
        """
        Creates a mixture of two overlapping distributions by setting the logits of the factorial Bernoulli distribution
        defined on the z components. This is a generic class that can work with any type of smoothing distribution
        that extends the SmoothingDist class.
        Args:
            param: params[0] is the logit of the probability of the random binary variables being 1.
            kw_param: a dictionary containing the key 'smoothing_dist' which is an object representing the overlapping
            distribution.
        """

        assert isinstance(param, list), 'param should be a list.'
        assert len(param) == MixtureGeneric.num_param, 'param should have a length of %d.' % MixtureGeneric.num_param
        assert isinstance(kw_param['smoothing_dist'], SmoothingDist)
        FactorialBernoulliUtil.__init__(self, [param[0]])
        self.smoothing_dist = kw_param['smoothing_dist']

    def reparameterize(self, is_training):
        """"
        This function uses ancestral sampling to sample from mixture of two overlapping distributions. 
        It then uses the implicit gradient idea to compute the gradient of samples with respect to logit_q. 
        This idea is presented in DVAE# sec 3.4. This function does not implement the gradient of samples with respect
        to beta or other parameters of the smoothing transformation.

        Args:
            is_training: a flag indicating whether we are building the training computation graph

        Returns:
            zeta: samples from mixture of overlapping distributions.
        """
        q = tf.nn.sigmoid(self.logit_mu)

        z = FactorialBernoulliUtil.reparameterize(self, is_training=False)
        shape = tf.shape(z)
        zeta = self.smoothing_dist.sample(shape)
        zeta = tf.where(tf.equal(z, 0.), zeta, 1. - zeta)

        pdf_0 = self.smoothing_dist.pdf(zeta)
        pdf_1 = self.smoothing_dist.pdf(1. - zeta)
        cdf_0 = self.smoothing_dist.cdf(zeta)
        cdf_1 = 1. - self.smoothing_dist.cdf(1. - zeta)

        grad_q = (cdf_0 - cdf_1) / (q * pdf_1 + (1 - q) * pdf_0)
        grad_q = tf.stop_gradient(grad_q)
        grad_term = grad_q * q
        grad_term -= tf.stop_gradient(grad_term)

        zeta = tf.stop_gradient(zeta) + grad_term

        return zeta

    def log_prob_per_var(self, samples):
        """
        Compute the log probability of samples under mixture of overlapping distributions.
        Args:
            samples: matrix of size (num_samples * num_vars)

        Returns: 
            log_prob: a matrix of log_prob (num_samples * num_vars).
        """
        q = tf.nn.sigmoid(self.logit_mu)
        pdf_0 = self.smoothing_dist.pdf(samples)
        pdf_1 = self.smoothing_dist.pdf(1. - samples)
        log_prob = tf.log(q * pdf_1 + (1 - q) * pdf_0)
        return log_prob

    def log_prob(self, samples):
        """
        Call log_prob_per_var() and then compute the sum of log_prob of all the variables.
        Args:
            samples: matrix of size (num_samples * num_vars)

        Returns: 
            log_p: A vector of log_prob for each sample.
        """
        log_p = self.log_prob_per_var(samples)
        log_p = tf.reduce_sum(log_p, 1)
        return log_p

    def log_ratio(self, zeta):
        """
        Compute log_ratio needed for gradients of KL (presented in DVAE++).
        Args:
            zeta: approximate post samples

        Returns:
            log_ratio: log r(\zeta|z=1) - log r(\zeta|z=0) 
        """
        log_pdf_0 = self.smoothing_dist.log_pdf(zeta)
        log_pdf_1 = self.smoothing_dist.log_pdf(1. - zeta)
        log_ratio = log_pdf_1 - log_pdf_0
        return log_ratio



