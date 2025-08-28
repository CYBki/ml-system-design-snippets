
# Pinterest Visual Search System - Detailed Technical Analysis
## Core Problem and Solution Approach
Pinterest has billions of images and users want to discover content not just through text search, but through visual similarity. The team had to solve these fundamental challenges:

Scale Problem: Processing over 1 billion unique images
Cost Control: Building this with a small engineering team (approximately 6-7 people)
Speed: Providing real-time results to users
Accuracy: Minimizing irrelevant results

## System Architecture - Step-by-Step Operation
### 1. Feature Extraction
Pinterest extracts the following features from each image:
Deep Learning Features:

VGG-16 Model: 4096-dimensional feature vector from fc6 layer
AlexNet: Features from fc6 and fc8 layers
Custom Fine-tuning: Model trained on Pinterest's own data
These features are binarized (converted to 0s and 1s) - advantages in both speed and storage

Color Features:

Detect salient regions of the image (saliency detection)
Group colors in these regions using k-means in Lab color space and store them

### 2. Intelligent Object Detection System
There's a very clever approach here:
Stage 1 - Text Filtering:
Example: If a pin's description is "spring fashion, floral tote bag"
→ System predicts "bag" category
→ Runs only the bag detector (not all detectors!)

Stage 2 - Visual Detection:

Uses Deformable Part Models (DPM)
Extracts bounding box for each object
Considered successful if Intersection over Union (IoU) &gt; 0.3

Results:

Text only: 72.7% true positive but 6.7% false positive
Image only: 49.5% true positive, 2.5% false positive
Combined: 38.1% true positive, 0.5% false positive!

### 3. Incremental Fingerprinting System
This part is critical - thousands of new images arrive daily:
Epoch System:
2014-xx-xx/ → All images up to that date
2015-01-06/ → Only images added that day
2015-01-07/ → Only images added that day
...

Workflow:

New Image Detection: Collects MD5 hashes of newly uploaded images daily
Queue System: Enqueues jobs with PinLater (similar to Amazon SQS)
Distributed Processing:

Initial setup: Hundreds of 32-core machines, 1 day duration, 5TB data
Daily update: Only 5 machines sufficient!


Work Chunks: Each chunk divided to take ~30 minutes processing
Spot Instance Usage: If machine terminates, job moves to another worker

### 4. Search Infrastructure
Index Structure:
Shard 1 (Machine 1):
├── Token Index (On Disk)
│   └── Visual word → [image1_id, image5_id, ...]
└── Feature Tree (In Memory)
    └── Full feature vectors

Shard 2 (Machine 2):
├── Token Index
└── Feature Tree

... up to N machines

Search Process:

When query arrives: Image features extracted (~200-600ms)
Leaf Rankers: Each shard finds its K-nearest neighbors
Re-ranking: Score adjustment with metadata (pin descriptions, board info)
Root Ranker: Combines results from all shards for final ranking

## Two Main Applications Details
### Related Pins
Problem: 6% of pins on Pinterest had no or very few recommendations
Solution Steps:

First searches for near-duplicate images
If not found, retrieves similar images using VGG features
Metadata conformity threshold applied - fewer results but more relevant

Test Methodology:

1000 queries × 3000 images = 1.6 million image test set
Tagged each image with the query that retrieved it
Two images considered "relevant" if they share the same tag

Live Test:

Tested on 10% traffic for 3 months
Result: 2% increase in total repins

### Similar Looks
Goal: Find specific products in fashion images and show similar items
9 Categories Detected:

Shoes (873 samples) - Most successful
Dress (383 samples)
Glasses (238 samples)
Bag (468 samples)
Watch, pants, shorts, bikini, earrings

User Experience:

Red dot appears on detected object
User clicks the dot
Products similar to that object are listed

Performance:

80 million "clickable" objects detected
12% daily users clicked on dots
Each user clicked on average 0.55 similar products

## Logic Behind Technical Decisions
Why VGG-16?

Highest precision among tested models (30.2% @5)
Acceptable latency (642ms)

Why Two-Stage Detection?

Running all detectors is too expensive
Text already exists, why not use it?
Reduced false positives from 6.7% to 0.5%

Why Incremental Update?

Impossible to reprocess 1 billion images daily
Processing only new/changed images is sufficient
Cost optimization is critical

Why Binary Features?

Floats are 32 bits, binary is 1 bit - 32x less space
Hamming distance computed very fast (XOR operation)
Minimal performance loss

## System Strengths and Weaknesses
Strengths:

Built with open-source tools (Caffe, Hadoop, HBase)
Cost-effective (only 5 machines in steady-state)
Tested with real user data

Potential Improvement Areas:

CNN-based object detection (like Faster R-CNN)
Better utilization of Pinterest's social graph data
More interactive interfaces

## Key Performance Metrics



Model
Precision@5
Precision@10
Latency




AlexNet FC6
5.1%
4.0%
193ms


Pinterest FC6
23.4%
21.0%
234ms


GoogLeNet
22.3%
20.2%
1207ms


VGG 16-layer
30.2%
26.9%
642ms



## Object Detection Accuracy by Method



Method
True Positive
False Positive




Text Only
72.7%
6.7%


Image Only
49.5%
2.5%


Combined
38.1%
0.5%



This system is a beautiful example of pragmatic engineering approach - they've produced a working, scalable, and cost-effective solution that may not be perfect but delivers real value to users.
