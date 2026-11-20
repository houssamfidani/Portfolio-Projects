# Visual instance retrieval

A photograph of a painting serves as a query against **1,611 paintings**. A bag-of-visual-words index produces a shortlist, then geometric verification checks which image regions agree on scale and position. The notebook also applies SIFT matching to a planar image replacement.

[Open the notebook](visual_instance_retrieval.ipynb) · [Recorded rankings and inlier counts](results/recorded-results.json)

## Recorded result

| Original visual-word rank | Rank after geometry check | Inlier pairs |
|---:|---:|---:|
| 2 | 1 | 69 |
| 1 | 2 | 67 |
| 3 | 3 | 14 |

The first two candidates depict the query painting. Both survived the geometric check, with a clear gap to the closest incorrect candidate. This comparison covers one query and ten shortlisted images.

## Run

From this directory in Python 3.12 or later:

```bash
python -m pip install -r requirements.txt
python -m jupyterlab visual_instance_retrieval.ipynb
```

The first run downloads the scene images and a 168 MB painting index. Thumbnails use the image URLs in the index and are cached locally. Ranking uses the index and works even if individual thumbnails are unavailable.

## Sources

The images and visual-word index come from the ENS WILLOW materials. The experiments build on work by Andrea Vedaldi and Andrew Zisserman, adapted by Gül Varol, Ignacio Rocco, and Fernando Julio Cendra, and on material by David Fouhey adapted by Guillaume Le Moing. [Oxford VGG](https://www.robots.ox.ac.uk/~vgg/practicals/overview/index.html) · [David Fouhey](https://cs.nyu.edu/~fouhey/teaching/EECS442_W23)
