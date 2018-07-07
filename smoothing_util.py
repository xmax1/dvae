# Copyright 2018 D-Wave Systems Inc.
# DVAE# licensed to authorized users only under the applicable license
# agreement.  See LICENSE.

import tensorflow as tf
import numpy as np


class SmoothingDist:
    def pdf(self, zeta):
        """ Implements r(\zeta|z=0)"""
        raise NotImplementedError

    def cdf(self, zeta):
        """ Implements R(\zeta|z=0)"""
        raise NotImplementedError

    def sample(self, shape):
        """ Samples from r(\zeta|z=0)"""
        raise NotImplementedError

    def log_pdf(self, zeta):
        """ Computes log r(\zeta|z=0)"""
        raise NotImplementedError


class PowerLaw(SmoothingDist):
    """ This class implements the smoothing distribution class for power function."""
    def __init__(self, params):
        self._lambda = 1. / params['beta']

    def pdf(self, zeta):
        pdf = tf.pow(zeta + 1e-7, self._lambda - 1.) * self._lambda
        return pdf

    def cdf(self, zeta):
        cdf = tf.pow(zeta + 1e-7, self._lambda)
        return cdf

    def sample(self, shape):
        rho = tf.random_uniform(shape)
        zeta = tf.pow(rho, 1. / self._lambda)
        return zeta

    def log_pdf(self, zeta):
        log_pdf = (self._lambda - 1.) * tf.log(zeta + 1e-7) + tf.log(self._lambda)
        return log_pdf


class Normal(SmoothingDist):
    """ This class implements the smoothing distribution class for Gaussian smoothing."""
    def __init__(self, params):
        self.sigma = tf.sqrt(1. / params['beta'])

    def pdf(self, zeta):
        pdf = 1. / (np.sqrt(2 * np.pi) * self.sigma) * tf.exp(-0.5 * zeta * zeta / tf.square(self.sigma))
        return pdf

    def cdf(self, zeta):
        return 0.5 * (1. + tf.erf(zeta / (np.sqrt(2.) * self.sigma)))

    def sample(self, shape):
        rho = tf.random_normal(shape)
        return self.sigma * rho

    def log_pdf(self, zeta):
        pdf = -0.5 * zeta * zeta / tf.square(self.sigma) - 0.5 * tf.log(2 * np.pi) - tf.log(self.sigma)
        return pdf


class Exponential(SmoothingDist):
    """ This class implements the smoothing distribution class for Exponential smoothing."""
    def __init__(self, params):
        self.beta = params['beta']

    def pdf(self, zeta):
        return self.beta * tf.exp(- self.beta * zeta) / (1 - tf.exp(-self.beta))

    def cdf(self, zeta):
        return (1. - tf.exp(- self.beta * zeta)) / (1 - tf.exp(-self.beta))

    def sample(self, shape):
        rho = tf.random_uniform(shape)
        zeta = - tf.log(1. - (1. - tf.exp(-self.beta)) * rho) / self.beta
        return zeta

    def log_pdf(self, zeta):
        return tf.log(self.beta) - self.beta * zeta - tf.log(1 - tf.exp(-self.beta))


class ExponentialUniform(SmoothingDist):
    """ This class implements the smoothing distribution class for Uniform-Exponential smoothing."""
    def __init__(self, params):
        self.beta = params['beta']
        self.eps = params['eps']

    def pdf(self, zeta):
        pdf_r = self.beta * tf.exp(- self.beta * zeta) / (1 - tf.exp(-self.beta))
        return (1. - self.eps) * pdf_r + self.eps

    def cdf(self, zeta):
        cdf_r = (1. - tf.exp(- self.beta * zeta)) / (1 - tf.exp(-self.beta))
        return (1. - self.eps) * cdf_r + self.eps * zeta

    def sample(self, shape):
        rho1 = tf.random_uniform(shape)  # sample from epsilon
        rho2 = tf.random_uniform(shape)  # sample conditioned on epsilon
        zeta = - tf.log(1. - (1. - tf.exp(-self.beta)) * rho2) / self.beta
        zeta = tf.where(rho1 < self.eps, rho2, zeta)
        return zeta

    def log_pdf(self, zeta):
        return tf.log(self.pdf(zeta))
