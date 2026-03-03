# Common Task 2 — Jets as Graphs: GNN Quark/Gluon Classification
## ML4SCI GSoC 2026

---

## Overview

This task trains a **Dynamic Graph Convolutional Neural Network (DGCNN)** to classify particle jets as either **quark-initiated** or **gluon-initiated**, using a graph-based representation of the detector data.

The full pipeline transforms raw 3-channel detector images into graphs, then applies EdgeConv layers to classify each jet.

**Implementation:** Pure PyTorch — no external GNN library (PyTorch Geometric) required.  
**Dataset:** Quark/Gluon jet dataset, 30,000 events sampled from 139,306 total.  
**Final Test ROC-AUC: 0.7830 | Accuracy: 71.30%**

---

## Pipeline

```
125×125×3 Detector Image
        ↓
    Point Cloud         ← keep only non-zero pixels (~664 points/event avg)
        ↓
    k-NN Graph          ← connect K=16 nearest neighbours in (η, φ) space
        ↓
    DGCNN / EdgeConv    ← 3 message-passing layers
        ↓
    Global Pooling      ← MaxPool + MeanPool per graph
        ↓
    MLP Classifier      ← outputs P(quark), P(gluon)
```

---

## Step-by-Step Explanation

### Step 1 — Load Data (Cell 3)

30,000 jet events are loaded from the HDF5 file into RAM using chunked reads (5,000 at a time) to avoid memory overflow. The raw images have shape `(N, 3, 125, 125)` where the three channels are:

| Channel | Detector | Physical meaning |
|---------|----------|-----------------|
| ECAL | Electromagnetic Calorimeter | Energy from electrons and photons |
| HCAL | Hadronic Calorimeter | Energy from hadrons (pions, protons…) |
| Tracks | Tracker | Paths of charged particles |

**Output:** `X: (30000, 3, 125, 125)` — balanced: 15,168 gluon / 14,832 quark

---

### Step 2 — Image → Point Cloud (Cell 4)

Each 125×125×3 image is **sparse** — most pixels are zero (empty detector cells). Instead of processing the full grid, we extract only the **active pixels** (where any channel exceeds a noise threshold of 1e-5).

Each active pixel becomes a **point** with 5 features:

```
point = [η,  φ,  ECAL,  HCAL,  Track]
         ↑   ↑     ↑      ↑      ↑
       row  col   ch0    ch1    ch2
       /124 /124   log1p normalised to [0,1]
```

- **η (eta)** = row / 124 — proxy for pseudorapidity (polar angle from beam axis)
- **φ (phi)** = col / 124 — proxy for azimuthal angle
- Energy channels are `log1p`-compressed to handle the heavy-tailed energy distribution

**Output stats from 2,000 events:**
```
Mean: 664 points/event   Median: 641   Min: 215   Max: 1538
```

This confirms jets are dense enough (hundreds of active pixels) for meaningful graph structure.

---

### Step 3 — Point Cloud → k-NN Graph (Cell 5)

Each point cloud is converted into a graph by connecting each node to its **K=16 nearest neighbours** in (η, φ) space using a **KDTree** (O(N log N) vs brute-force O(N²)).

**Graph structure:**
```
Nodes       = active energy deposit points   (variable N per event)
Edges       = spatial proximity in (η, φ)    (each node → 16 neighbours)
Node feats  = [η, φ, ECAL, HCAL, Track]      (5 features)
Label       = 0 (gluon) or 1 (quark)
```

**Sample graph:** 884 nodes, 17,480 edges  
**Total graphs built:** 30,000 (0 skipped — all events had ≥ 5 active points)

**Why k-NN in (η, φ)?**  
Angular distance in (η, φ) is the natural proximity metric for jet particles — it corresponds to their angular separation as seen from the interaction point.

**Why K=16?**  
Standard value from the DGCNN paper. Too few edges (K=4) misses neighbourhood context; too many (K=32) introduces noise and slows computation.

---

### Step 4 — Graph Batching (Cell 7)

Variable-size graphs cannot be stacked into a single tensor. Instead, a batch of N graphs is merged into **one large disconnected graph**:

```
batch.x      = cat([g1.x, g2.x, ..., gN.x])   shape: (total_nodes, 5)
batch.src    = cat([g1.src + offset1, ...])     shape: (total_edges,)
batch.batch  = [0,0,...,0, 1,1,...,1, ...]      which graph each node belongs to
batch.y      = [label_1, label_2, ..., label_N] shape: (N,)
```

The `batch` index vector allows `global_pool` operations to correctly aggregate per-graph embeddings.

**Split:** 22,500 train (49.7% quark) | 4,500 val | 3,000 test — balanced splits

---

### Step 5 — EdgeConv Primitives (Cell 8)

Three core operations are implemented from scratch in pure PyTorch:

#### `scatter_max_pool(src_feat, dst_idx, num_nodes)`
For each target node, takes the **maximum** over all incoming edge features. Uses `scatter_reduce_('amax')` — native PyTorch 2.0+.

#### `global_max_pool(x, batch, B)` and `global_mean_pool(x, batch, B)`
Aggregate all node embeddings within each graph into a single graph-level vector. Used after the last EdgeConv layer to produce a fixed-size representation per jet.

#### `EdgeConvLayer(in_dim, out_dim)`
The core message-passing operation:
```
For each edge (src → dst):
    edge_feature = MLP( [x_dst  ‖  x_src − x_dst] )
                          ↑             ↑
                     global info    relative info

New x_dst = MaxPool over all incoming edge_feature
```

The `x_src − x_dst` term encodes **relative** position and energy differences between connected particles. This makes the layer:
- **Translation-invariant** — sensitive to angular separations, not absolute positions
- **Geometry-aware** — captures the direction and magnitude of energy flow between deposits

---

### Step 6 — DGCNN Architecture (Cell 9)

```
Input (N, 5)
  ↓  EdgeConv-1 : MLP(10→64→64)       LeakyReLU + BN
  ↓  EdgeConv-2 : MLP(128→128→128)    LeakyReLU + BN
  ↓  EdgeConv-3 : MLP(256→256→256)    LeakyReLU + BN
  ↓  GlobalMaxPool(256) + GlobalMeanPool(256) → cat → (512,)
  ↓  Linear(512→256) → ReLU → Dropout(0.4)
  ↓  Linear(256→128) → ReLU → Dropout(0.3)
  ↓  Linear(128→2)
  ↓  Softmax → P(gluon), P(quark)
```

**Why concatenate MaxPool + MeanPool?**
- MaxPool captures the most energetic node → peak jet energy
- MeanPool captures average jet structure → multiplicity and spread
- Together they give a richer graph-level descriptor

**Trainable parameters:** 334,850

---

### Step 7 — Training (Cell 10)

| Setting | Value | Reason |
|---------|-------|--------|
| Loss | CrossEntropyLoss | Standard for binary classification (includes softmax) |
| Optimiser | AdamW | Weight decay regularisation prevents overfitting |
| LR Schedule | CosineAnnealing | Smooth decay from 1e-3 → 1e-5 over 40 epochs |
| Gradient clip | max_norm=1.0 | Prevents exploding gradients on variable-size graphs |
| Early stopping | patience=8 on Val AUC | AUC is the evaluation metric — stop when it plateaus |

**Training log (selected epochs):**
```
  Ep    TrainLoss    ValLoss    ValAUC    ValAcc
  001     0.6477      0.5940    0.7562   68.98%  ✓
  005     0.5862      0.5837    0.7732   70.67%  ✓
  007     0.5808      0.5888    0.7777   71.07%  ✓
  011     0.5753      0.5809    0.7785   71.13%  ✓
  013     0.5729      0.6001    0.7783   70.89%
  ...
  Early stopping — Best Val AUC: 0.7785
```

Train and val loss stay close throughout — no significant overfitting.

---

### Step 8 — Evaluation (Cell 12)

```
════════════════════════════════════════════
  DGCNN (EdgeConv) — Test Set Results
════════════════════════════════════════════
  ROC-AUC   : 0.7830
  Accuracy  : 71.30%
════════════════════════════════════════════
```

**ROC-AUC = 0.7830** means the model correctly ranks a random quark jet above a random gluon jet 78.3% of the time — consistent with literature values for EdgeConv on this dataset (0.75–0.82).

**What the ROC curve shows:**  
At any operating threshold, the model achieves significantly better true-positive rate than random. The area under this curve (AUC) summarises the overall discriminating power across all thresholds.

**What the score distribution shows:**  
Quark and gluon P(quark) score distributions overlap substantially — this is expected. Quark/gluon discrimination is fundamentally ambiguous in QCD because both produce similar hadronic showers. The partial separation visible in the histogram is the learned physics signal.

---

## Physics Interpretation

**Why are quark and gluon jets different?**

| Property | Quark jet | Gluon jet |
|----------|-----------|-----------|
| QCD color factor | CF = 4/3 | CA = 3 |
| Particle multiplicity | Lower | Higher (ratio CA/CF = 9/4) |
| Angular spread | Narrower | Broader |
| Softer radiation | Less | More |

The DGCNN learns these differences via:
- **Angular spread:** broader node distributions in (η, φ) for gluons
- **Multiplicity:** more nodes per event for gluons (higher activity)
- **Energy topology:** EdgeConv's `x_j − x_i` captures energy gradients between deposits

---

## Limitation → Motivation for Specific Task 4

The k-NN graph construction used here connects each particle only to its **K nearest spatial neighbours**. This means the model only sees **local** jet substructure.

**Long-range correlations** — such as back-to-back energy deposits or wide-angle radiation patterns — are invisible to a local GNN.

**Specific Task 4 (Non-local GNN)** addresses this by:
1. Replacing k-NN edges with **attention over all particle pairs** (self-attention / transformer-style)
2. Allowing any two particles to communicate regardless of angular distance
3. Expected improvement: **+2–4% AUC** over the local DGCNN baseline

---

## File Outputs

| File | Description |
|------|-------------|
| `pc_sizes.png` | Histogram of point cloud sizes across events |
| `jet_graph_quark.png` | Quark jet visualised as a graph in (η, φ) |
| `jet_graph_gluon.png` | Gluon jet visualised as a graph in (η, φ) |
| `gnn_training.png` | Loss curves + Val AUC over epochs |
| `roc_curve.png` | ROC curve + score distribution on test set |
| `best_gnn.pth` | Best model checkpoint (saved at peak Val AUC) |
