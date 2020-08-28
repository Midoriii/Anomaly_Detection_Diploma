# Scripts for running experiments on Metacentrum.cz
FAIR WARNING: Running Siamese network scripts requires a lot of resources (High-end GPU,
and mostly about 150GB of RAM) because of the size of the input images (768x768), which
mustn't be reduced, lest crucial details are lost. Since the input data is made of
'All with All' pairs, it's considerably sizeable.

## AutoencodersByType
Scripts for training hand-picked well-performing Autoencoder architectures. This
time the Autoencoders are trained on just BSE or SE images, not mixed together.
They are also using the extended variant of faulty images for performance evaluation,
containing also the faulty images with plugged central hole.

## AutoencodersExtendedFaultyData
Scripts for training hand-picked best-performing Autoencoder architectures. Trained
on mixed BSE and SE images but tested on all the faulty images, including the previously
omitted ones with plugged central hole.

## BasicAutoencoders
Scripts for training a wide range of Autoencoder architectures, each with a range of
epoch parameters. These Autoencoders are trained on BSE and SE images coupled together.
Performance is evaluated on all faulty images barring the ones with plugged central hole.

## BasicAutoencodersCleanedData
Scripts for training a wide range of Autoencoder architectures, each with a range of
epoch parameters. This time they are trained on reduced mixed set of BSE and SE images,
containing only the clearest and 'most correct' ones. Performance is evaluated on all
faulty images barring the ones with plugged central hole.

## ContrastiveLoss
Scripts for training hand-picked well-performing Siamese Network architectures. This
time using Contrastive loss instead of Binary Cross Entropy. The performance is also
evaluated on all of the faulty images, including the ones with plugged central hole.

## SiameseNets
Scripts for training a wide range of Siamese Network architectures. Because of the
essence of how Siamese Nets work, they are trained and evaluated on only one of
the image types, either BSE or SE. Evaluation is made on all the faulty data with
the exception of images with plugged central hole.

## SiameseNetsExtendedFaultyData
Scripts for training hand-picked well-performing Siamese Network architectures. This
time the evaluation is done on all of the faulty images, including the ones with plugged
central hole.

## TransposeConvAutoencoders
Scripts for training several handpicked Autoencoders which use Conv2DTranspose layer
instead of UpSampling2D layer. The difference lies in trainability of the layers,
with Conv2DTranspose layer being trainable. Performance evaluated on the same data
as the BasicAutoencoders folder scripts.

### The rest
Scripts for data preparation coupled with one output file containing concrete numbers
about the amount of pairs created.
