# TF USEv5 API (no AVX)

This is a simple repo that serves v5 of the Universal Sentence Encoder as an API. In my use case, I have an older CPU, so I use a custom build of tensorflow that was built without AVX. Also I have a custom step to strip `tensorflow_text` from the langchain wrapper because I can't get it working on my AVX-incompatible platform. Pretty sure that import is not used anyway... might put in a PR to strip it from the source.