import tkinter as tk
from tkinter import ttk, messagebox
from eight_puzzle_solver import *

class PuzzleSolverApp:
    def __init__(self, root):
        self.root = root
        self.root.title("8-Puzzle Solver v2.0")
        self.root.geometry("600x700")
        
        # Estado inicial
        self.board = [[1, 2, 3], [4, 0, 6], [7, 5, 8]]
        self.solution_path = []
        self.current_step = 0
        
        self.create_widgets()
        self.display_board()
        
    def create_widgets(self):
        # Frame de configuração
        config_frame = ttk.LabelFrame(self.root, text="Configurações", padding=10)
        config_frame.pack(fill="x", padx=10, pady=5)
        
        # Controle de estado inicial
        ttk.Label(config_frame, text="Estado Inicial:").grid(row=0, column=0, sticky="w")
        self.state_var = tk.StringVar(value="padrão")
        states = ["padrão", "aleatório", "personalizado"]
        ttk.OptionMenu(config_frame, self.state_var, states[0], *states, 
                      command=self.update_state).grid(row=0, column=1, sticky="ew", padx=5)
        
        # Controle de algoritmo
        ttk.Label(config_frame, text="Algoritmo:").grid(row=1, column=0, sticky="w")
        self.algo_var = tk.StringVar(value="A*")
        algos = ["A*", "BFS", "DFS"]
        ttk.OptionMenu(config_frame, self.algo_var, algos[0], *algos).grid(row=1, column=1, sticky="ew", padx=5)
        
        # Controle de heurística
        self.heuristic_frame = ttk.Frame(config_frame)
        ttk.Label(self.heuristic_frame, text="Heurística:").pack(side="left")
        self.heuristic_var = tk.StringVar(value="manhattan")
        heuristics = ["manhattan", "misplaced", "euclidean"]
        ttk.OptionMenu(self.heuristic_frame, self.heuristic_var, heuristics[0], *heuristics).pack(side="left")
        self.heuristic_frame.grid(row=2, column=0, columnspan=2, sticky="w", pady=5)
        
        # Controle de profundidade para DFS
        self.depth_frame = ttk.Frame(config_frame)
        ttk.Label(self.depth_frame, text="Profundidade Máx:").pack(side="left")
        self.depth_var = tk.IntVar(value=30)
        ttk.Spinbox(self.depth_frame, from_=10, to=100, textvariable=self.depth_var, width=5).pack(side="left")
        self.depth_frame.grid(row=3, column=0, columnspan=2, sticky="w")
        self.depth_frame.grid_remove()
        
        # Botões de ação
        button_frame = ttk.Frame(self.root)
        ttk.Button(button_frame, text="Executar", command=self.run_solver).pack(side="left", padx=5)
        ttk.Button(button_frame, text="Reiniciar", command=self.reset).pack(side="left", padx=5)
        button_frame.pack(pady=10)
        
        # Área do tabuleiro
        self.board_frame = ttk.Frame(self.root)
        self.board_frame.pack(pady=10)
        
        # Controles de animação
        self.control_frame = ttk.Frame(self.root)
        self.prev_btn = ttk.Button(self.control_frame, text="< Anterior", command=self.prev_step, state="disabled")
        self.prev_btn.pack(side="left", padx=5)
        self.next_btn = ttk.Button(self.control_frame, text="Próximo >", command=self.next_step, state="disabled")
        self.next_btn.pack(side="left", padx=5)
        self.control_frame.pack(pady=5)
        
        # Área de resultados
        result_frame = ttk.LabelFrame(self.root, text="Resultados", padding=10)
        self.result_text = tk.Text(result_frame, height=8, width=70, state="disabled", font=('Courier', 10))
        scrollbar = ttk.Scrollbar(result_frame, command=self.result_text.yview)
        self.result_text['yscrollcommand'] = scrollbar.set
        self.result_text.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        result_frame.pack(fill="both", expand=True, padx=10, pady=5)
        
        # Atualizar visibilidade dos controles
        self.algo_var.trace_add('write', self.update_controls)
        self.update_controls()
    
    def update_controls(self, *args):
        if self.algo_var.get() == "A*":
            self.heuristic_frame.grid()
            self.depth_frame.grid_remove()
        elif self.algo_var.get() == "DFS":
            self.heuristic_frame.grid_remove()
            self.depth_frame.grid()
        else:
            self.heuristic_frame.grid_remove()
            self.depth_frame.grid_remove()
    
    def update_state(self, choice):
        if choice == "aleatório":
            self.board = generate_solvable_state()
            self.display_board()
        elif choice == "personalizado":
            self.customize_board()
    
    def customize_board(self):
        custom_window = tk.Toplevel(self.root)
        custom_window.title("Editar Tabuleiro")
        
        entries = []
        for i in range(3):
            row = []
            for j in range(3):
                entry = ttk.Entry(custom_window, width=3, font=('Arial', 14), justify='center')
                entry.grid(row=i, column=j, padx=5, pady=5)
                entry.insert(0, str(self.board[i][j]) if self.board[i][j] != 0 else "")
                row.append(entry)
            entries.append(row)
        
        def save_board():
            try:
                new_board = []
                for i in range(3):
                    row = []
                    for j in range(3):
                        val = entries[i][j].get()
                        row.append(int(val) if val else 0)
                    new_board.append(row)
                
                if not find_zero(new_board):
                    messagebox.showerror("Erro", "O tabuleiro deve ter um espaço vazio (0)")
                    return
                
                if not is_solvable(new_board):
                    messagebox.showwarning("Aviso", "Este tabuleiro não tem solução!")
                
                self.board = new_board
                self.display_board()
                custom_window.destroy()
            
            except ValueError:
                messagebox.showerror("Erro", "Digite apenas números de 0 a 8")
        
        ttk.Button(custom_window, text="Salvar", command=save_board).grid(row=3, columnspan=3, pady=10)
    
    def display_board(self, board=None):
        if board is None:
            board = self.board
            
        for widget in self.board_frame.winfo_children():
            widget.destroy()
            
        for i in range(3):
            for j in range(3):
                val = board[i][j]
                bg = "white" if val != 0 else "lightgray"
                label = tk.Label(self.board_frame, text=str(val) if val != 0 else "", 
                                width=4, height=2, relief="ridge",
                                font=("Arial", 16, "bold"), bg=bg)
                label.grid(row=i, column=j, padx=2, pady=2)
    
    def run_solver(self):
        algorithm = self.algo_var.get()
        
        try:
            if algorithm == "A*":
                result = a_star_search(self.board, self.heuristic_var.get())
            elif algorithm == "BFS":
                result = bfs_search(self.board)
            else:  # DFS
                result = dfs_search(self.board, self.depth_var.get())
            
            self.show_results(result)
            self.solution_path = result['path'] or []
            self.current_step = 0
            self.update_animation_controls()
            
            if self.solution_path:
                self.display_board(self.solution_path[0])
        
        except Exception as e:
            messagebox.showerror("Erro", str(e))
    
    def show_results(self, result):
        self.result_text.config(state="normal")
        self.result_text.delete(1.0, tk.END)
        
        if result['path'] is None:
            self.result_text.insert(tk.END, "Nenhuma solução encontrada!\n\n")
        else:
            self.result_text.insert(tk.END, 
                f"Método: {result['method']}\n"
                f"Tempo: {result['time']:.6f} segundos\n"
                f"Nós expandidos: {result['nodes']}\n"
                f"Profundidade: {result['depth']}\n"
                f"Total de movimentos: {len(result['path'])-1}\n\n")
            
            self.result_text.insert(tk.END, "Caminho da solução:\n")
            for i, state in enumerate(result['path']):
                self.result_text.insert(tk.END, f"Passo {i}:\n")
                for row in state:
                    self.result_text.insert(tk.END, f"{row}\n")
                self.result_text.insert(tk.END, "\n")
        
        self.result_text.config(state="disabled")
        self.result_text.see(1.0)
    
    def prev_step(self):
        if self.current_step > 0:
            self.current_step -= 1
            self.display_board(self.solution_path[self.current_step])
            self.update_animation_controls()
    
    def next_step(self):
        if self.current_step < len(self.solution_path) - 1:
            self.current_step += 1
            self.display_board(self.solution_path[self.current_step])
            self.update_animation_controls()
    
    def update_animation_controls(self):
        self.prev_btn.config(state="normal" if self.current_step > 0 else "disabled")
        self.next_btn.config(state="normal" if self.current_step < len(self.solution_path)-1 else "disabled")
    
    def reset(self):
        self.board = [[1, 2, 3], [4, 0, 6], [7, 5, 8]]
        self.solution_path = []
        self.current_step = 0
        self.display_board()
        self.result_text.config(state="normal")
        self.result_text.delete(1.0, tk.END)
        self.result_text.config(state="disabled")
        self.update_animation_controls()

if __name__ == "__main__":
    root = tk.Tk()
    app = PuzzleSolverApp(root)
    root.mainloop()