# COLMAP vocabulary trees

The `vocab_tree/` directory holds pre-trained FAISS-based visual-word vocabulary trees used by COLMAP (≥ 3.12) for image retrieval — they pre-filter candidate image pairs and dramatically speed up SfM on large datasets. They come from the official COLMAP mirror at [demuc.de/colmap](https://demuc.de/colmap/).

This repo ships only the **FAISS** family. The older FLANN + SIFT trees from COLMAP ≤ 3.11 are not included — the conda env's `pycolmap 3.13.0` uses the modern FAISS format, and the 256K FAISS tree is a direct upgrade over the classic 256K. If you must use a pre-3.12 COLMAP, download a classic tree from [demuc.de/colmap](https://demuc.de/colmap/) directly.

| File                                                              | Size  | Words | Descriptor                      | When to use                                                 |
| ----------------------------------------------------------------- | ----- | ----- | ------------------------------- | ----------------------------------------------------------- |
| `vocab_tree_faiss_flickr100K_words256K.bin`                       | 70 MB | 256 K | SIFT (FAISS)                    | Default for SIFT-based pipelines — drop-in upgrade over the classic 256K |
| `vocab_tree_faiss_flickr100K_words64K_aliked_n16rot.bin`          | 18 MB | 64 K  | ALIKED n16 (rotation-invariant) | For ALIKED-n16 features + COLMAP 3.12 deep retrieval        |
| `vocab_tree_faiss_flickr100K_words64K_aliked_n32.bin`             | 18 MB | 64 K  | ALIKED n32                      | For ALIKED-n32 features                                     |

Pass them to COLMAP with:

```bash
colmap vocab_tree_matcher \
    --VocabTreeMatching.vocab_tree_path vocab_tree/vocab_tree_faiss_flickr100K_words256K.bin \
    --database_path <db>
```

## Storage — Git LFS

These files total ~106 MB and are tracked via **Git LFS** (`.gitattributes` → `vocab_tree/*.bin filter=lfs`). Consequences for users:

- **Lazy download.** A regular `git clone` downloads only small LFS pointers (~190 B each); the real binaries are fetched only when a command touches them.
- **Fetch selectively:**

  ```bash
  git lfs pull --include "vocab_tree/vocab_tree_faiss_flickr100K_words256K.bin"
  ```

- **Fetch all three:**

  ```bash
  git lfs pull
  ```

- **Skip LFS entirely at clone time:**

  ```bash
  GIT_LFS_SKIP_SMUDGE=1 git clone git@github.com:behnamasadi/OpenCVProjects.git
  ```

### Prerequisite — install git-lfs once per machine

```bash
sudo apt install git-lfs     # Debian/Ubuntu
git lfs install              # registers the filter in ~/.gitconfig
```

On macOS: `brew install git-lfs`.
