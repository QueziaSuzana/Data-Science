
# 🧩 8-Puzzle Solver with Search Algorithms (A*, BFS, DFS)

This project implements the classic **8-Puzzle** game and solves its configurations using three search algorithms: **A\*** (with three heuristics), **BFS (Breadth-First Search)**, and **DFS (Depth-First Search)**. It was developed as part of a practical assignment for an Artificial Intelligence course.

---

## 🎯 Objectives

- Implement the 8-Puzzle game with a modular structure.
- Solve puzzle instances using informed and uninformed search algorithms.
- Evaluate and compare algorithm performance using different heuristics.
- Visualize results using a **Looker Studio dashboard**.

---

## 🛠️ Technologies and Tools

- `Python` (for algorithm implementation)
- `Google Sheets` (to store experimental results)
- `Looker Studio` (to build an interactive dashboard)

---

## 📦 Implemented Algorithms

- **A\*** (A-Star Search)
  - Heuristics:
    - `Manhattan Distance`
    - `Misplaced Tiles`
    - `Euclidean Distance`
- **BFS** (Breadth-First Search)
- **DFS** (Depth-First Search), with depth limits (d=10, d=20, d=30)

---

## 🔍 Heuristics and Justification

- **Misplaced Tiles**: simple and fast; counts the number of misplaced tiles.
- **Manhattan Distance**: sums horizontal and vertical distances; good balance of cost and accuracy.
- **Euclidean Distance**: more precise but computationally expensive.

---

## 📊 Experimental Results

Tests were run on both `standard` and `random` initial states, measuring:

- Execution time (seconds)
- Number of expanded nodes
- Solution depth
- Total number of moves

Summary of findings:

| Algorithm | Fastest | Fewest Nodes | Deepest Solution | Best Heuristic |
|-----------|---------|---------------|------------------|----------------|
| A*        | ✅ Yes  | ✅ Yes        | ✅ Yes           | Manhattan      |
| BFS       | Average | Medium         | Shallow          | —              |
| DFS       | ❌ Slow | ❌ Many nodes  | Inconsistent      | —              |

> **A\*** with **Manhattan** heuristic had the best overall performance.

---

## 📈 Looker Studio Dashboard

An interactive dashboard was created in **Looker Studio** to visualize and compare metrics for each algorithm and heuristic.

**Main visuals:**
- Time by algorithm
- Expanded nodes by method
- A* execution time by heuristic
- Solution depth and number of moves

![dash8puzzle](8Puzzle/Puzzle_-_Analysis_of_the_Experimental_Results.pdf)

---

## 📁 Project Structure

```bash
├── eight_puzzle_solver.py # A*, BFS, DFS implementations and Heuristic functions
├── interface_puzzle.py    # Execution script and export logic
├── Puzzle_-_Analysis_of_the_Experimental_Results            # Experimental results dashboard 
└── README.md              # This file
```

---

## 🧠 Conclusion

This project demonstrates how different search strategies behave when solving the same problem. The analysis confirms the efficiency of informed search, especially when using well-designed heuristics like **Manhattan Distance**.

---

## 📌 Author

Developed by [Quézia Silva] – [AI / PUC - Minas]
