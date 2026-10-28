# Assignment 5: Demonstrate math, numpy, scipy, pandas, matplotlib
import math
import numpy as np
import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt

# ---------- 1. math module ----------
print("=== MATH MODULE ===")
print("ceil(4.2)      :", math.ceil(4.2))
print("floor(4.8)     :", math.floor(4.8))
print("factorial(5)   :", math.factorial(5))
print("sqrt(81)       :", math.sqrt(81))
print("fabs(-7.5)     :", math.fabs(-7.5))
print("copysign(5,-1) :", math.copysign(5, -1))

# ---------- 2. NumPy ----------
print("\n=== NUMPY ===")
arr = np.array([[1, 2, 3], [4, 5, 6]])
print("Array:\n", arr)
print("Shape:", arr.shape)
print("Reshaped:\n", arr.reshape(3, 2))
print("Transpose:\n", arr.T)
print("Add 10:\n", np.add(arr, 10))
print("Square root:\n", np.sqrt(arr))

# ---------- 3. SciPy ----------
print("\n=== SCIPY ===")
sample = [12, 15, 12, 18, 20, 15, 12, 25]
print("Mode:", stats.mode(sample, keepdims=True).mode[0])
print("Pearson correlation:", stats.pearsonr([1,2,3,4,5], [2,4,6,8,10]).statistic)

# ---------- 4. Pandas ----------
print("\n=== PANDAS ===")
# Using built-in sklearn iris as a CSV-like DataFrame
from sklearn.datasets import load_iris
iris = load_iris(as_frame=True)
df = iris.frame
print("First 5 rows:\n", df.head())
print("\nNull values per column:\n", df.isnull().sum())
print("\nSummary statistics:\n", df.describe())

# ---------- 5. NumPy & SciPy statistics on data ----------
print("\n=== STATISTICS ===")
vals = df['sepal length (cm)'].values
print("Mean (NumPy)   :", np.mean(vals))
print("Median (NumPy) :", np.median(vals))
print("Std Dev (NumPy):", np.std(vals))
print("Mode (SciPy)   :", stats.mode(vals, keepdims=True).mode[0])

# ---------- 6. Matplotlib Visualization ----------
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.plot(vals[:50], marker='o')
plt.title("Line Plot (first 50)")
plt.xlabel("Index"); plt.ylabel("Sepal Length")

plt.subplot(1, 3, 2)
df['species'] = iris.target
counts = df['species'].value_counts()
plt.bar(counts.index.astype(str), counts.values, color=['r', 'g', 'b'])
plt.title("Bar Chart: Species Count")

plt.subplot(1, 3, 3)
plt.scatter(df['sepal length (cm)'], df['petal length (cm)'],
            c=iris.target, cmap='viridis')
plt.title("Scatter: Sepal vs Petal")
plt.xlabel("Sepal Length"); plt.ylabel("Petal Length")

plt.tight_layout()
plt.show()
