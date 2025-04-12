# 🧭 Gridworld Pathfinding with Adaptive A*, Repeated A*, and Heuristic Evaluation

This project implements and compares advanced pathfinding strategies—Repeated Forward A*, Repeated Backward A*, and Adaptive A*—across multiple generated gridworld environments. It investigates tie-breaking strategies, heuristic consistency, and algorithmic efficiency using empirical results and visualization.

---

## 🧠 Algorithms Implemented

- ✅ Repeated Forward A*
- ✅ Repeated Backward A*
- ✅ Adaptive A*
- ✅ Tie-breaking strategies (smaller g vs. larger g)
- ✅ Heuristic consistency checker

---

## 📁 Project Structure

```
├── part0.py               # Gridworld generator using DFS
├── part2.py               # Repeated Forward A* with tie-breaking comparison
├── part2graph.py          # Visualizations comparing tie-breaking
├── part3.py               # Repeated Forward vs Backward A*
├── part5.py               # Repeated Forward vs Adaptive A*
├── visualisation.py       # Gridworld visualizer
├── *.csv                  # Result datasets for each experimental test
├── CS440_Assignment_1.pdf # Final report (Sahil, Rahul, Dennis)
```

---

## 📊 Key Results

| Algorithm               | Avg Expanded Nodes | Avg Runtime (s) |
|------------------------|--------------------|-----------------|
| Repeated Forward A*    | ~3009.98           | ~0.01073        |
| Adaptive A*            | ~3006.32           | ~0.01082        |
| Repeated Backward A*   | ~5437.98           | ~0.01392        |

---

## 🔍 Insights

- **Tie-breaking with larger g-values** reduces expanded nodes and runtime
- **Repeated Forward A\*** is generally more efficient than **Backward A\***
- **Adaptive A\*** shows slightly fewer node expansions, but marginal overhead due to heuristic updates
- Manhattan heuristics were proven consistent across all tested gridworlds

---

## 📦 How to Run

```bash
python3 part0.py               # Generate gridworlds
python3 part2.py               # Tie-breaking strategy test
python3 part3.py               # Forward vs Backward A*
python3 part5.py               # Forward vs Adaptive A*
python3 visualisation.py       # Save visualizations
```

---

## 📄 Report

Full analysis and experimental write-up is in:  
📄 `CS440_Assignment_1.pdf`

---

## 🙌 Authors

Sahil Sharma, Rahul Sankaralingam, Dennis Hemans  
Fall 2024 · CS440: Introduction to Artificial Intelligence

---

## 📜 License

For academic and research use only.
