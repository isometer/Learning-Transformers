This repo is a crash course in the Transformer architecture and one application.

We'd like to build a transformer-based autoencoder that can identify and correct anomalies in physiological data.
We're currently using [this article](https://medium.com/@moussab.orabi/enable-transformers-with-anomaly-detection-in-high-order-multivariate-time-series-data-509a5df39151) as a guide, but we'll be looking elsewhere as needed.

Now using the [MIT-BIH Noise Stress Test Dataset](https://physionet.org/content/nstdb/1.0.0/) as a benchmark to help identify what is true signal and what is noise. These files are interpreted using the python WFDB (WaveForm DataBase) package.