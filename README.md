## Variational Autoencoder with Arbitrary Conditioning (VAEAC)

VAEAC is a generative model designed for **image inpainting with arbitrary masks**. Unlike standard VAEs that assume a fixed input structure, VAEAC learns a **conditional latent distribution** based on the visible pixels and the mask. This enables the model to reconstruct missing regions regardless of their size, shape, or position.

###  Brief Architecture Overview

- **Proposal Network (Posterior):**  
  Learns an approximate latent distribution using the *full image* during training.

- **Conditional Prior Network:**  
  Predicts the latent distribution using only the *observed pixels* and the mask.

- **Decoder (Generative Network):**  
  Takes a sampled latent vector along with the observed pixels and reconstructs the missing region.

The training objective aligns the prior and posterior distributions while teaching the decoder to fill masked regions in a realistic and context-aware manner.

###  Datasets Used

This implementation has been tested on the following datasets:

- **MNIST**
- **Fashion-MNIST**
- **CelebA**

These experiments demonstrate VAEAC’s flexibility across simple grayscale digits to high-dimensional facial images.

###  Medium Article

A detailed explanation of the intuition, architecture, and training process is available in my Medium article:

 **Understanding Image Inpainting and the Variational Autoencoder with Arbitrary Conditioning**  
https://medium.com/@PREET9/understanding-image-inpainting-and-the-variational-autoencoder-with-arbitrary-conditioning-225b2552a9cc
