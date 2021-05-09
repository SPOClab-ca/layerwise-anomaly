# Layerwise Anomaly

This repository contains the source code and data for our ACL 2021 paper: *"How is BERT surprised? Layerwise detection of linguistic anomalies"* by Bai Li, Zining Zhu, Guillaume Thomas, Yang Xu, and Frank Rudzicz.

## Citation

If you use our work in your research, please cite:

Li, B., Zhu, Z., Thomas, G., Xu, Y., and Rudzicz, F. (2021) How is BERT surprised? Layerwise detection of linguistic anomalies. In *Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics (ACL)*.

```
@inproceedings{li2021layerwise,
  author = "Li, Bai and Zhu, Zining and Thomas, Guillaume and Xu, Yang and Rudzicz, Frank",
  title = "How is BERT surprised? Layerwise detection of linguistic anomalies",
  booktitle = "Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics (ACL)",
  publisher = "Association for Computational Linguistics",
  year = "2021",
}
```

## Dependencies

The project was developed with the following library versions. Running with other versions may crash or produce incorrect results.

* Python 3.7.5
* CUDA Version: 11.0
* torch==1.7.1
* transformers==4.5.1
* numpy==1.19.0
* pandas==0.25.3
* scikit-learn==0.22

## Setup Instructions

Todo

## Run unit tests

`PYTHONPATH=. pytest tests`
