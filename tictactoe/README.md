# Task 2: Tic-Tac-Toe AI with Minimax and Alpha-Beta Pruning

This directory contains the source code and walkthrough for an unbeatable Tic-Tac-Toe AI. The AI evaluates possible future moves using recursive search tree evaluations to guarantee it never loses.

## 🚀 How to Run

### **1. Command Line Interface (CLI)**
Play Tic-Tac-Toe against the AI directly in your terminal:
```bash
python tictactoe_ai.py
```

### **2. Jupyter Notebook**
Explore the detailed minimax algorithm implementation, game states search tree, alpha-beta optimizations, and code blocks showing AI blocking moves:
```bash
jupyter notebook tictactoe_notebook.ipynb
```

---

## 🛠️ Implementation Details

### **Core Algorithm: Minimax with Alpha-Beta Pruning**
Minimax is a backtracking decision-tree algorithm. It calculates all possible future moves for both the Maximizer (AI) and the Minimizer (Human) and chooses the path that guarantees the best possible outcome.

```text
       Max (AI)           [   O   ]  <- Best Move Chosen
                         /    |    \
       Min (Human)    [ -1 ] [ 0 ] [ +1 ]
                      /  \   /   \   /  \
       Max (AI)      ... ... ... ... ... ...
```

### **Alpha-Beta Pruning Optimization**
Without pruning, the algorithm traverses all possible subtrees. With Alpha-Beta Pruning, we keep track of:
- **Alpha ($\alpha$)**: The best score the Maximizer can guarantee.
- **Beta ($\beta$)**: The best score the Minimizer can guarantee.

If we find a branch where the opponent can force a worse outcome than a branch we already evaluated (i.e., $\beta \le \alpha$), we immediately **prune** that branch and stop evaluating its child nodes, reducing computation time and resource consumption.

### **Heuristic Scoring**
- AI Win: `+10 - depth` (We subtract depth so the AI prefers faster wins)
- Human Win: `depth - 10` (We add depth so the AI delays inevitable losses)
- Draw: `0`
