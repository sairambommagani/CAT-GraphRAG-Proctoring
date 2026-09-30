# Python & Machine Learning Assessment
exam_length: 15

This is the syllabus for the assessment. The GraphRAG index (exam_graph/) turns it into a knowledge graph with the structure Section → Topic → Concept, plus prerequisite links. The adaptive exam's blueprint, the tags on every question and the AI question generator all come from this file.

To edit the syllabus, keep this format:
- `## Section`, followed by a `weight:` line (its share of the exam)
- `### Topic`, followed by an optional `requires:` line listing prerequisite topics
- the topic text, with the key concepts in **bold**

## Python Basics
weight: 0.2

### Types and Operators
Python is dynamically typed: a value carries its type, and a name can be bound to values of different types over time. The core built-in types are **int**, **float**, **str**, **bool** and **NoneType**. The **true division** operator `/` always returns a float, even for two ints (`5 / 2` is `2.5`), while **floor division** `//` rounds down to an integer for ints. The **`is` operator** checks object identity (whether two names point to the same object in memory), whereas `==` checks value equality.

### Collections and Indexing
requires: Types and Operators
A **list** is a mutable, ordered sequence; a **tuple** is an immutable one; a **dict** maps hashable keys to values, and `{}` creates an empty dict (an empty set needs `set()`). **Slicing** `seq[start:stop]` returns a new sequence from `start` up to but not including `stop`, so `[1, 2, 3][1:]` is `[2, 3]`. **zip** pairs elements from several iterables and stops at the shortest one. `sorted()` returns a new sorted list and accepts a **key function**, for example `key=lambda x: -x` sorts in descending order.

### References and Mutability
requires: Collections and Indexing
Assignment never copies: `b = a` makes both names refer to the same object, a situation called **aliasing**. Mutating the object through one name (for example `b.append(4)`) is visible through the other. To get an independent object use a **shallow copy** (`a.copy()`, `list(a)`, `a[:]`) or, for nested structures, a **deep copy** (`copy.deepcopy`).

### Functions and Generators
requires: Types and Operators
Functions are defined with the **def** keyword and are first-class objects. A **lambda** is a small anonymous function limited to a single expression. A **generator** is a function that uses **yield**; calling it returns an iterator that produces values lazily, one at a time, keeping its local state between values. This makes generators memory-efficient for large or infinite sequences.

## OOP
weight: 0.2

### Classes and Objects
requires: Functions and Generators
A **class** is defined with the `class` keyword and acts as a blueprint for objects. The **`__init__` method** initialises a new instance after it is created. Inside instance methods, **self** refers to the specific instance the method was called on, giving access to its attributes.

### Inheritance and Overriding
requires: Classes and Objects
**Inheritance** lets a child class reuse and extend a parent class. **Method overriding** means a child class defines a method with the same name as one in its parent, replacing the parent's behaviour for child instances. **super()** returns a proxy that delegates method calls to the next class in the method resolution order, typically to call the parent's version of an overridden method.

### Multiple Inheritance and MRO
requires: Inheritance and Overriding
With **multiple inheritance** a class can have several parents. The **method resolution order (MRO)**, computed with **C3 linearization**, determines the order in which classes are searched for an attribute. If two parents define the same method, the one that appears first in the MRO is used, which is usually the leftmost parent in the class definition.

### Encapsulation and Class Design
requires: Classes and Objects
**Encapsulation** bundles data with the methods that operate on it and hides internal details behind a public interface; Python signals non-public attributes by convention with a leading underscore. A **@staticmethod** receives neither the instance nor the class, while a **@classmethod** receives the class as its first argument (`cls`) and is often used for alternative constructors. Defining **`__slots__`** restricts instances to a fixed set of attributes and removes the per-instance `__dict__`, which saves memory.

## Data Structures & Algorithms
weight: 0.2

### Complexity and Arrays
requires: Collections and Indexing
**Big-O notation** describes how running time grows with input size. Accessing a Python list element by index is **O(1)** because lists are dynamic arrays. A **queue** is **FIFO** (first in, first out) and a **stack** is **LIFO** (last in, first out). A **hash table** (Python's dict and set) gives **O(1) average** lookup, degrading only with many collisions.

### Sorting
requires: Complexity and Arrays
**Merge sort** and **heapsort** run in **O(n log n)** time in the worst case. **Quicksort** is O(n log n) on average, but its **worst case is O(n²)**, which happens when the pivot choices are consistently poor, for example on already-sorted input with a first-element pivot.

### Trees and Heaps
requires: Complexity and Arrays
A **binary search tree (BST)** keeps smaller keys on the left and larger keys on the right. Search is O(log n) when the tree is balanced, but **O(n) in the worst case** for an unbalanced tree that degenerates into a list. An **in-order traversal** of a BST visits the keys in sorted order. A **binary heap** supports insert and extract-min in **O(log n)**, which makes it better than a sorted array (O(n) insertion) for a **priority queue**.

### Dynamic Programming and Union-Find
requires: Sorting, Trees and Heaps
**Dynamic programming** improves on naive recursion by storing the solutions of **overlapping subproblems** (**memoization** or tabulation) so that each one is computed only once. The **union-find** (disjoint set) structure with **path compression** and **union by rank** runs each operation in nearly constant **amortized** time, **O(α(n))**, where α is the inverse Ackermann function.

## ML Fundamentals
weight: 0.2

### Learning Paradigms
**Supervised learning** trains on labelled examples (input and target pairs) to predict targets. **Unsupervised learning** finds structure such as clusters in unlabelled data. The data is split into a **training set**, a **validation set** used to tune hyperparameters and choose models, and a **test set** held back for the final unbiased estimate.

### Overfitting and Regularization
requires: Learning Paradigms
**Overfitting** is when a model fits noise in the training data and performs well on training data but poorly on new data. The **bias-variance tradeoff** describes the balance between error from overly simple assumptions (bias) and error from sensitivity to the particular training sample (variance). **Regularization** reduces overfitting by penalising model complexity. **L1 regularization** tends to drive weights exactly to zero, which gives sparse models, while **L2 regularization** shrinks all weights smoothly towards zero.

### Optimization
requires: Learning Paradigms
**Gradient descent** minimises a loss function by repeatedly moving the parameters in the direction of the negative gradient. The **learning rate** sets the step size. Too high a learning rate makes training **diverge** or oscillate, and too low a rate makes it very slow.

### Deep Network Training
requires: Optimization, Overfitting and Regularization
In deep networks with **sigmoid** or **tanh** activations, gradients are multiplied through many layers of small derivatives and shrink exponentially. This is the **vanishing gradient problem**, and it stalls learning in early layers. **ReLU** activations reduce this problem. **Batch normalization** normalises the layer inputs within each mini-batch, which stabilises and speeds up training and allows higher learning rates.

## Model Evaluation
weight: 0.2

### Classification Metrics
requires: Learning Paradigms
**Accuracy** is the fraction of predictions that are correct. **Precision** is TP / (TP + FP), the share of predicted positives that are truly positive. **Recall** (sensitivity) is TP / (TP + FN), the share of actual positives that are found. On **imbalanced data** accuracy can be misleading: always predicting the majority class can score high while finding no positives. The **F1 score** is the harmonic mean of precision and recall.

### ROC and AUC
requires: Classification Metrics
The **ROC curve** plots the **true positive rate** against the **false positive rate** across all decision thresholds. The **AUC** summarises the curve: 1.0 is a perfect ranking, and **0.5 means no better than random guessing**.

### Validation Strategy
requires: Classification Metrics, Overfitting and Regularization
**k-fold cross-validation** trains and evaluates the model k times on different splits and averages the scores. This reduces the variance of the performance estimate compared with a single split. **Stratified k-fold** keeps the class proportions the same in every fold, which matters for imbalanced classes.

### Probability Calibration
requires: ROC and AUC
A model can rank examples well (a high AUC) and still output poorly **calibrated probabilities**: a predicted 0.8 may not mean an 80% chance. **Calibration** methods such as **Platt scaling** and **isotonic regression** remap scores so that predicted probabilities match observed frequencies. Neither accuracy nor AUC measures this.
