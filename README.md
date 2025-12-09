📘 Variational Autoencoder with Arbitrary Conditioning (VAEAC)

VAEAC is a generative model designed for image inpainting with arbitrary masks. Unlike standard VAEs, which assume a fixed input structure, VAEAC learns a conditional latent distribution based on the visible pixels and the mask, allowing it to reconstruct missing regions regardless of their shape, size, or position.

Model Architecture (Brief)

Proposal Network (Posterior): Learns an approximate latent distribution using the full image during training.

Conditional Prior Network: Predicts the latent distribution using only the observed (unmasked) pixels and the mask.

Decoder (Generative Network): Takes a sampled latent vector along with the observed pixels and reconstructs the missing region.

The training objective aligns the prior with the posterior while teaching the decoder to fill in masked regions realistically. At inference, only the conditional prior and decoder are used, enabling diverse and context-aware inpainting.

Experiments

This implementation has been tested on:

MNIST

Fashion-MNIST

CelebA

Each dataset demonstrates VAEAC’s ability to handle different levels of complexity—from simple digits to high-dimensional face images.

✍️ Medium Article

A detailed explanation of the intuition, architecture, and training dynamics of VAEAC is available in my Medium article:

👉 Understanding Image Inpainting and the Variational Autoencoder with Arbitrary Conditioning
https://medium.com/@PREET9/understanding-image-inpainting-and-the-variational-autoencoder-with-arbitrary-conditioning-225b2552a9cc
