
# Machine Learning Infrastructure and MLOps for Practitioners

### 1. Dream vs. Reality for ML Practitioners

**The Dream:**
- Data is provided
- Build optimal ML prediction system
- Deploy as scalable API or edge deployment
- System generates more data and improves itself

**The Reality:**
- Aggregate, process, clean, label, and version data
- Write and debug model code
- Provision compute resources
- Run many experiments and review results
- Discover mistakes and try different architectures
- Deploy the model
- Monitor production predictions

### 2. Three Main Tooling Categories

ML infrastructure can be broken down into three buckets:

#### **Data**
- Data sources
- Data lakes/warehouses
- Data processing
- Data exploration
- Data versioning
- Data labeling

#### **Training/Evaluation**
- Compute sources
- Resource management
- Software engineering
- Frameworks and distributed training libraries
- Experiment management
- Hyperparameter tuning

#### **Deployment**
- Continuous integration and testing
- Edge deployment
- Web deployment
- Monitoring
- Feature store

### 3. Software Engineering Tools

#### **Python Ecosystem**
- **Language Choice**: Python (general-purpose, easy to learn)
- **Scientific Libraries**: Pandas, NumPy, Scikit-Learn
- **IDE Options**: VS Code, Jupyter, PyCharm, Vim, Emacs

#### **Jupyter Notebooks Pros and Cons**

**Advantages:**
- Quick prototyping
- Exploratory analysis
- Companies like Netflix base entire workflows on them

**Disadvantages:**
- Version control difficulty (large JSON files)
- Primitive IDE features
- Hard to structure code reasonably
- Out-of-order execution artifacts
- Difficult for long/distributed tasks

#### **Streamlit**
- Framework for ML engineers to create beautiful apps
- Works like Python scripts
- Treats widgets as variables
- Has caching mechanism

### 4. Compute Hardware

#### **GPU Architectures (NVIDIA)**
- **Kepler/Maxwell**: Old, 2-4x slower
- **Pascal**: 1080 Ti cards, still useful
- **Volta/Turing**: Mixed precision support, tensor cores
- **Ampere**: Latest, 30% speed improvement

#### **Cloud vs On-Premise**

**Cloud Advantages:**
- Easy scalability
- DevOps convenience
- No maintenance

**On-Premise Advantages:**
- Long-term cost efficiency
- Full control

#### **Recommendations:**
- **Hobbyists**: Own machine + cloud
- **Startups**: Lambda Labs machine + cloud
- **Large Companies**: Powerful local machine + cloud

### 5. Resource Management Solutions

1. **Custom Scripts**: Simple but manual
2. **SLURM**: Cluster job scheduler
3. **Docker/Kubernetes**: Container-based solution
   - Kubeflow: Google's OSS project
4. **Custom ML Software**: AWS SageMaker, Paperspace Gradient

### 6. Frameworks and Distributed Training

#### **Deep Learning Frameworks**
- **TensorFlow**: Production-optimized, static graph (2015)
- **Keras**: Simple interface for TensorFlow
- **PyTorch**: Development-friendly, dynamic graph (2017)

**Current State:**
- Both have converged to similar capabilities
- PyTorch has 80%+ usage in academia
- TensorFlow added eager execution
- PyTorch subsumed Caffe2

#### **Why Frameworks Are Needed**
- **Auto-differentiation**: Automatic gradient computation
- **GPU Support**: CUDA integration
- **Abstraction**: Layers, optimizers, loss functions

#### **Distributed Training**

**Data Parallelism:**
- Split batch across GPUs
- Each GPU computes gradients
- Average at central node
- Linear speedup (~2x GPUs = 2x speed)

**Model Parallelism:**
- Split model weights across GPUs
- Complex, should be avoided if possible
- Gradient checkpointing as alternative

### 7. Experiment Management

#### **Low-Tech Solution**
- Manual tracking with Excel/spreadsheets
- Complex and error-prone

#### **Professional Solutions**
- **TensorBoard**: Comes with TensorFlow, simple
- **MLFlow**: Databricks OSS project
- **Paid Platforms**: Comet.ml, Weights & Biases, Neptune

### 8. Hyperparameter Tuning

#### **Software Solutions**
- **SigOpt**: API-based iterative optimization
- **Ray Tune**: Local software integrated with resource allocation
- **Weights & Biases**: YAML-based sweep configuration

#### **Benefits**
- Early stopping of poor-performing models
- Intelligent parameter sweeping
- Cost savings

### 9. "All-in-One" Solutions

#### **Core Features**
- Data labeling and querying services
- Model training and scaling
- Experiment tracking and versioning
- Development environments
- Model deployment and monitoring

#### **Examples**
- **Facebook FBLearner** (2016): One of the earliest examples
- **Cloud Vendors**: Google Cloud AI, AWS SageMaker, Azure ML
- **Startups**: Paperspace Gradient, Neptune, FloydHub
- **Open Source**: Determined AI

## Key Technical Insights

### Tesla's ML System Example
Tesla's self-driving system exemplifies the dream: data collection → training → evaluation → inference → deployment → more data collection in a continuous loop.

### Google's Technical Debt Paper
Shows that ML code is a tiny fraction of real-world ML systems - most effort goes into infrastructure for data processing, monitoring, configuration, and deployment.

### Framework Evolution
- TensorFlow started production-focused but hard to develop
- PyTorch started development-friendly but slow at scale
- Both have converged: TF added eager execution, PyTorch added compilation

### GPU Memory Importance
- Only data in GPU memory can be computed
- More GPU RAM = larger batches = faster training
- Tensor cores (Volta+) provide mixed-precision acceleration

### Resource Management Trade-offs
- Custom scripts: Simple but limited
- SLURM: Standard but requires setup
- Kubernetes: Powerful but complex
- All-in-one: Convenient but vendor lock-in

This comprehensive overview covers the entire ML infrastructure stack from development to deployment, highlighting both technical requirements and practical considerations for different organizational scales.
